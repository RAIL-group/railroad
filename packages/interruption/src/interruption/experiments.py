"""
Running interruption experiments: a planner (planners.py) works through a
scenario's task sequence (scenarios.py) while tasks arrive.

    result = execute(scenario, planner)          # -> RunResult, no dashboard
    metrics = run_experiment(scenario, planner)  # also replays the run in the dashboard
"""
from typing import Optional
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from rich.console import Console

from railroad.core import (
    Action,
    Fluent,
    Goal,
    LiteralGoal,
    AndGoal,
    convert_goal_to_positive_preconditions,
    ff_heuristic,
    get_action_by_name,
)
from railroad.dashboard import PlannerDashboard
from railroad.dashboard._protocols import DashboardPlanner
from railroad.environment import SymbolicEnvironment
from railroad.environment.procthor.environment import ProcTHOREnvironment

from .constants import ACTION_PROB_DEBUG
from .dashboard_adapters import AstarDashboardPlanner
from .environments import KitchenProcTHOREnvironment
from .planner import InterruptionSearchProblem
from .planning_framework import Planner, RetryState, search_with_retry, search_without_retry
from .scenarios import Scenario
from .utilities import (
    get_action_cost,
    negative_fluent_preprocessing,
)


@dataclass
class ExperimentData:
    """
    One run's working state: the environment, the search problem for the current
    task, and the scenario's goals mapped to the scene's names.
    """
    env: KitchenProcTHOREnvironment
    search_problem: InterruptionSearchProblem
    planner: Planner
    neg_to_pos_mapping: dict[Fluent, Fluent]
    # Scenario.goal and .interrupting_task_dist in the scene's names
    goal: Goal
    interrupting_task_dist: tuple[list[Goal], list[float]]
    failure_cost: float = 500
    retry_state: RetryState = field(default_factory=RetryState)


@dataclass(frozen=True)
class RunResult:
    """What happened in one run of a scenario."""
    success: bool
    # the run's total time, failure penalties included
    cost: float
    # each executed action's name, marked " | Goal Complete" / " | Interrupt", and
    # "Failed Task" for each task the planner found no plan for
    actions: list[str]
    # the tasks in the order they arrived, in the scene's names
    task_sequence: list[Goal]
    # searches that failed and were retried (the solver fell back to a smaller
    # sub-task or to only the goal's objects), and of those, the goal-object ones
    search_retries: int
    object_filtered_searches: int
    # seconds spent planning and executing
    wall_time: float

    @property
    def had_failed_search(self) -> bool:
        """Whether a task went unplanned (without retries, it costs Scenario.failure_cost)."""
        return "Failed Task" in self.actions

    def metrics(self) -> dict:
        """The result as the benchmark records it (see benchmarks/bench_interruption.py)."""
        return {
            "success": self.success,
            "wall_time": self.wall_time,
            "plan_cost": self.cost,
            "actions_count": len(self.actions),
            "required_search_retry": self.search_retries > 0,
            "search_retries": self.search_retries,
            "required_object_filtering": self.object_filtered_searches > 0,
            "object_filtered_searches": self.object_filtered_searches,
            "had_failed_search": self.had_failed_search,
            "actions": self.actions,
            "log_html": "",
            "log_plot": b'',
        }


def execute(scenario: Scenario, planner: Planner) -> RunResult:
    """
    Runs scenario with planner (see planners.py); once every task has arrived,
    its myopic() planner solves the rest. No dashboard: see run_experiment.
    """
    return _execute(initialize_experiment_data(scenario, planner), scenario)


def run_experiment(
    scenario: Scenario,
    planner: Planner,
    *,
    record: bool = False,
    show_plot: bool = False,
    save_plot: str | None = None,
    save_video: str | None = None,
) -> dict:
    """
    execute, then replays the run in the railroad dashboard: live in the terminal,
    then show_plot / save_plot / save_video; or, with record, silently, adding the
    dashboard's HTML log ("log_html") and trajectory plot ("log_plot") to the
    result, as the benchmarks do. Returns RunResult.metrics(), whose wall_time
    includes the replay.
    """
    data = initialize_experiment_data(scenario, planner)
    start_time = time.perf_counter()
    result = _execute(data, scenario)

    dash_env = scenario.build_environment()
    task_sequence_goal = _get_task(result.task_sequence)
    recording_console = Console(record=True, force_terminal=True, width=120) if record else None
    dashboard = _setup_dashboard(
        convert_goal_to_positive_preconditions(task_sequence_goal, data.neg_to_pos_mapping),
        dash_env, recording_console,
        partial(AstarDashboardPlanner, heuristic_fn=ff_heuristic), record
    )
    execute_deterministic_replay(result.actions, dashboard, dash_env, scenario.failure_cost)

    metrics = result.metrics()
    metrics["wall_time"] = time.perf_counter() - start_time
    _finish_dashboard(
        metrics, dashboard, dash_env, recording_console, show_plot, save_plot, save_video
    )
    return metrics


def _execute(data: ExperimentData, scenario: Scenario) -> RunResult:
    # print out the actual action probabilities
    if ACTION_PROB_DEBUG:
        _show_action_probabilities(data)

    # set the experiment seed after loading the ProcTHOR scene since it sets the
    # seed to procthor_seed
    random.seed(scenario.seeds.experiment_seed)
    start_time = time.perf_counter()

    event_trace, task_sequence, search_retries, filtered_searches = (
        _get_task_sequence_event_trace(data, scenario)
    )
    success = (
        "Goal Complete" in event_trace[-1]
        if not scenario.augment_task
        else _get_task(task_sequence).evaluate(data.env.state.fluents)
    )
    return RunResult(
        success=success,
        cost=float(data.env.state.time),
        actions=event_trace,
        task_sequence=task_sequence,
        search_retries=search_retries,
        object_filtered_searches=filtered_searches,
        wall_time=time.perf_counter() - start_time,
    )


# helper functions
def _get_task_sequence_event_trace(
    data: ExperimentData,
    scenario: Scenario,
) -> tuple[list[str], list[Goal], int, int]:
    """
    Searchs for a plan and executes the plan in the environment for the specificed n length
    task sequence. Additionally, stores a trace of the envents, in the form a of list of action
    names, which can later be deterministically replayed by the AstarDashboardPlanner.
    Also returns the total number of failed searches across all calls to the solver,
    and how many of those were retried over only the goal's objects.
    """
    event_trace = []
    search_retries = 0
    filtered_searches = 0
    task_arrival_sequence = [data.goal]
    current_task = data.goal
    num_incomplete_goals = scenario.num_task_sequence
    iteration_count = 0

    # the termination condition of the loop depends on how solver failure's are handled:
    # if the solver is allowed to retry, then continue until all goals are complete.
    # if the solver is not allowed to retry, then the solver should only be called
    # num_task_sequence+1 times
    termination_condition = (
        num_incomplete_goals > 0 if scenario.retry_with_subgoals
        else iteration_count < scenario.num_task_sequence
    )
    while termination_condition:
        # use the appropriate solver to plan
        all_tasks_arrived = len(task_arrival_sequence) == scenario.num_task_sequence

        planner = data.planner
        retry_state = data.retry_state
        # if all the tasks have arrived, just solve the goals in the task as quick as possible
        if all_tasks_arrived:
            planner = data.planner.myopic()
            # start from the task-size limit the experiment's searches found, without
            # passing what the myopic searches find back to them
            retry_state = replace(data.retry_state)

        if scenario.retry_with_subgoals:
            plan, success, num_retries, num_filtered = search_with_retry(
                planner, data.env, data.search_problem, data.neg_to_pos_mapping,
                retry_state=retry_state,
            )
        else:
            plan, success, num_retries, num_filtered = search_without_retry(
                planner, data.env, data.search_problem, data.neg_to_pos_mapping
            )
        search_retries += num_retries
        filtered_searches += num_filtered

        # execute the plan in the environment
        completed_goals = _execution_loop(
            (current_task, success),
            plan,
            data,
            event_trace,
            all_tasks_arrived
        )
        # update the number of incomplete goals in the task sequence
        num_incomplete_goals-=len(completed_goals)

        # task arrival
        newly_arrived_goal = None
        if not all_tasks_arrived:
            next_task_idx = len(task_arrival_sequence)-1
            newly_arrived_goal = data.interrupting_task_dist[0][next_task_idx]
            task_arrival_sequence.append(newly_arrived_goal)

        # update the task
        incomplete_goals = _get_incomplete_tasks(
            scenario.augment_task, current_task, newly_arrived_goal, completed_goals
        )

        if len(incomplete_goals) > 0:
            current_task = _get_task(incomplete_goals)
            data.search_problem.goal = convert_goal_to_positive_preconditions(
                current_task, data.neg_to_pos_mapping
            )

        iteration_count+=1
        # check if another call to the solver should be made
        termination_condition = (
            num_incomplete_goals > 0 if scenario.retry_with_subgoals
            else iteration_count < scenario.num_task_sequence
        )

    return event_trace, task_arrival_sequence, search_retries, filtered_searches


def _execution_loop(
    goal_success: tuple[Goal, bool],
    plan: list[Action],
    data: ExperimentData,
    event_trace: list[str],
    last_task_flag: bool = False
) -> list[Goal]:
    """
    Helper function for the execution loop of task interruption experiments.
    Returns a list of all successfully completed goals.
    """
    goal, success = goal_success
    completed_goals = []

    if not success:
        event_trace.append("Failed Task")
        # update the state's time to reflect the failed task
        data.env.apply_time_penalty(data.failure_cost)
        return completed_goals

    # goals not yet credited; shrinks as they complete, so actions taken after the
    # task is done (e.g. an anticipatory planner's extra literal) can't re-count them
    pending = [goal] if isinstance(goal, LiteralGoal) else list(goal.children())

    # execution loop
    action_idx = 0
    while True:
        task_arrival_prob = 0
        try:
            converted_action = plan[action_idx]
        except IndexError:
            converted_action = None

        if converted_action is not None:
            action = get_action_by_name(data.env.get_actions(), converted_action.name)
            # perform the action
            data.env.act(action)
            # update the scene_graph
            data.env.update_scene_graph(action)
            event_trace.append(converted_action.name)

            task_arrival_prob = (
                data.search_problem.interruption_prob_fn
                if isinstance(data.search_problem.interruption_prob_fn, (float, int))
                else data.search_problem.interruption_prob_fn(get_action_cost(action))
            )

        # check if current task was completed successfully
        # NOTE: only append to an action's name if the task wasn't already complete
        completed_by_action = [
            sub_goal for sub_goal in pending if sub_goal.evaluate(data.env.state.fluents)
        ]

        completed_goals.extend(completed_by_action)

        if len(completed_by_action) > 0:
            # goal was already complete without taking any actions
            if len(event_trace) == 0:
                event_trace.append("Goal Complete")
            # goal was completed after taking an action
            else:
                event_trace[-1] += " | Goal Complete"

            pending = [g for g in pending if g not in completed_by_action]

        # check if interrupting task arrived
        if (
            converted_action is not None and not last_task_flag and
            random.random() < task_arrival_prob
        ):
            event_trace[-1] += " | Interrupt"
            break

        # update plan idx
        action_idx+=1
        if len(plan) in {0, action_idx}:
            break

    return completed_goals


def _get_incomplete_tasks(
    augment: bool, current_task: Goal, new_task: Optional[Goal], completed_goals: list[Goal]
) -> list[Goal]:
    """
    Helper function for updating the current task. Its primary function is to 
    remove goal that have already been completed and add the newly arrived goal to
    the task.
    Returns a list of incomplete goals.
    """
    if not augment:
        assert new_task is not None
        return [new_task]
    # task augmentation experiments
    goals = [current_task] if isinstance(current_task, LiteralGoal) else current_task.children()
    incomplete_goals = [goal for goal in goals if goal not in completed_goals]
    if new_task is not None:
        incomplete_goals.append(new_task)
    return incomplete_goals


def _get_task(goals: list[Goal]) -> Goal:
    return AndGoal(goals) if len(goals) > 1 else goals[0]


def initialize_experiment_data(scenario: Scenario, planner: Planner) -> ExperimentData:
    """
    A run's starting point: a fresh environment, the scenario's goals mapped to
    the scene's names, and the search problem for its first task, with negative
    fluents converted for the FF heuristic. The scenario itself is left unchanged.
    """
    env = scenario.build_environment()

    # convert generic name of objects/containers in goals to actual names of
    # objects/containers within the scene (ex. spoon_14)
    # duplicate objects/containers are randomly sampled
    scene_objects = _get_scene_objects_locations(
        scenario.seeds.duplicate_objects_containers_seed,
        env.scene.objects
    )
    scene_locations = _get_scene_objects_locations(
        scenario.seeds.duplicate_objects_containers_seed,
        {loc for loc in env.scene.locations if loc != "start_loc"}
    )
    goal = _map_goal_to_scene(scenario.goal, scene_objects, scene_locations)
    task_dist: tuple[list[Goal], list[float]] = ([], [])
    for task, prob in zip(*scenario.interrupting_task_dist):
        task_dist[0].append(_map_goal_to_scene(task, scene_objects, scene_locations))
        task_dist[1].append(prob)

    # fluent pre-processing for usage of FF heuristic
    converted_actions, _, converted_goals, mapping= negative_fluent_preprocessing(
        env.get_actions(), env.state, [goal] + task_dist[0]
    )

    return ExperimentData(
        env,
        InterruptionSearchProblem(
            converted_goals[0],
            converted_actions,
            (list(converted_goals[1:]), task_dist[1]),
            scenario.task_arrival_fn,
        ),
        planner,
        mapping,
        goal,
        task_dist,
        failure_cost=scenario.failure_cost,
    )


def _setup_dashboard(
    goal: Goal,
    env: SymbolicEnvironment,
    recording_console: Console | None,
    planner_factory: Callable[[list[Action]], DashboardPlanner] | None,
    record: bool
) -> PlannerDashboard:
    """
    The dashboard to replay a run in: started live in the terminal, or, when
    recording, left silent for its history to be exported.
    """
    dashboard = PlannerDashboard(
        goal,
        env,
        planner_factory=planner_factory,
        print_on_exit=not record,
        console=recording_console,
    )

    if not record:
        dashboard.start()
    return dashboard


def _finish_dashboard(
    metrics: dict,
    dashboard: PlannerDashboard,
    dash_env: SymbolicEnvironment,
    recording_console: Console | None,
    show_plot: bool,
    save_plot: str | None,
    save_video: str | None,
) -> None:
    """
    Ends the replay: records the dashboard's HTML log and trajectory plot into
    metrics when recording, or stops the live dashboard and shows its plots.
    """
    if recording_console:
        dashboard.print_history()
        metrics["log_html"] = recording_console.export_html(inline_styles=True)

        if isinstance(dash_env, ProcTHOREnvironment):
            try:
                location_coords = {
                    name: (float(coord[0]), float(coord[1]))
                    for name, coord in dash_env.scene.locations.items()
                }
                plot_image = dashboard.get_plot_image(location_coords=location_coords)
                if plot_image is not None:
                    metrics["log_plot"] = plot_image
            except Exception as e:
                print(f"Failed to render trajectory plot: {e}")
    else:
        dashboard.stop()
        dashboard.show_plots(show_plot=show_plot, save_video=save_video, save_plot=save_plot)


def execute_deterministic_replay(
    event_trace: list[str],
    dashboard: PlannerDashboard,
    dash_env: KitchenProcTHOREnvironment,
    failure_cost: float,
) -> None:
    """
    Executes the deterministic replay of an interruption task sequence for viewing in 
    the railroad dashboard.
    """
    adapter = AstarDashboardPlanner(dash_env.get_actions(), ff_heuristic)

    for converted_action_name in event_trace:
        action_name = (
            converted_action_name.split(" |")[0]
            if any(sub in converted_action_name for sub in ["Interrupt", "Goal Complete"])
            else converted_action_name
        )
        try:
            action = get_action_by_name(dash_env.get_actions(), action_name)
            dash_env.act(action)
        except ValueError:
            # action_name was either "Goal Complete" or "Failed Task"
            if action_name == "Failed Task":
                dash_env.apply_time_penalty(failure_cost)

        dashboard.update(adapter, converted_action_name)


# TODO - make this function able to handle more complex AND and OR goals (likely need to use goal.children())
def _map_goal_to_scene(
    goal: Goal,
    scene_objects: dict[str, str],
    scene_locations: dict[str, str]
) -> Goal:
    """
    Helper function that converts the generic name objects/containers used in 
    goals (ex. at spoon shelvingunit) to actual names of objects/containers within
    the scene (ex. spoon_14). An `at` goal keeps its location as a category
    instead (at-category spoon_14 shelvingunit), so it is satisfied by any
    shelving unit in the scene, not just one of them.
    """
    if isinstance(goal, LiteralGoal):
        fluent = goal.fluent()
        # map generic objection/container names to scene specific objects/container names
        args = []
        for arg in fluent.args:
            # handle edge cases
            # if the goal argument is not in the scene, return the original goal
            if arg not in scene_objects and arg not in scene_locations:
                return goal
            # if the argument is both an object and location, return the original goal
            if arg in scene_objects and arg in scene_locations:
                return goal
            args.append(scene_objects[arg] if arg in scene_objects else scene_locations[arg])

        name = fluent.name
        if name == "at" and len(fluent.args) == 2 and fluent.args[1] in scene_locations:
            name, args = "at-category", [args[0], fluent.args[1]]
        fluent_str = f"{name} {' '.join(args)}"
        if fluent.negated:
            fluent_str = f"not {fluent_str}"
        return LiteralGoal(Fluent(fluent_str))
    return goal


def _get_scene_objects_locations(seed: int, objects: set[str]) -> dict[str, str]:
    """
    Helper function for sampling a unique mapping from generic object/location name
    (ex. spoon) to its name in the scene (ex. spoon_14).
    """
    rng = random.Random(seed)
    full_objects_mapping = defaultdict(list)
    # sort to ensure determinism, since sets aren't ordered
    for obj in sorted(objects, key=lambda x: int(x.split("_")[-1])):
        full_objects_mapping[obj.split("_")[0]].append(obj)
    return {
        generic_name : rng.choice(scene_objects)
        for generic_name, scene_objects in full_objects_mapping.items()
    }


## debugging helper functions
def _show_action_probabilities(data: ExperimentData) -> None:
    actual_probabilities = []
    pick_flag = False
    place_flag = False

    for action in data.search_problem.actions:
        action_type = action.name.split(" ")[0]
        action_cost = get_action_cost(action)
        if action_type == "pick":
            if pick_flag:
                continue
            pick_flag = True
        if action_type == "place":
            if place_flag:
                continue
            place_flag = True

        action_prob_int = (
            data.search_problem.interruption_prob_fn(action_cost)
            if not isinstance(data.search_problem.interruption_prob_fn, (int, float))
            else data.search_problem.interruption_prob_fn
        )
        actual_probabilities.append((action.name, action_cost, action_prob_int))

    sorted_probs = sorted(actual_probabilities, key=lambda x: x[-1], reverse=True)
    for i, (name, cost, prob) in enumerate(sorted_probs, 1):
        print(f"{i}. Action - {name}: Cost -  {cost:.4f}; P_I(a) - {prob:.4f}")
