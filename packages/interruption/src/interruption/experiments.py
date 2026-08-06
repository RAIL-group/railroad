from typing import Sequence
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from rich.console import Console

from railroad.core import (
    Action,
    Fluent,
    Goal,
    LiteralGoal,
    convert_goal_to_positive_preconditions,
    convert_state_to_positive_preconditions,
    ff_heuristic,
    get_action_by_name,
)
from railroad.dashboard import PlannerDashboard
from railroad.dashboard._protocols import DashboardPlanner
from railroad.environment import SymbolicEnvironment
from railroad.environment.procthor.environment import ProcTHOREnvironment

from .dashboard_adapters import AstarDashboardPlanner
from .environments import construct_procthor_kitchen_environment
from .planner import astar_search
from .utilities import (
    TaskArrivalProb,
    get_action_cost,
    get_task_arrival_prob,
    handcrafted_interruption_value,
    negative_fluent_preprocessing,
)


@dataclass
class ExperimentSeeds:
    """
    Data struct that contains all the seeds needed to make an
    interruption experiment deterministic.
    """
    procthor_seed: int = 201
    experiment_seed: int = 240
    task_sample_seed: int = 75
    object_placement_seed: int | None = None
    duplicate_objects_containers_seed: int = 240


@dataclass
class ExperimentConfig:
    """
    Data struct that holds all relevant user-provided experiment
    configuration information. The goal and goals in the task 
    distribution use the generic names of objects/containers
    to find matching objects/containers within the scene. These
    names are then used to update the goals.
    """
    goal: Goal
    interrupting_task_dist: tuple[Sequence[Goal], list[float]]
    baseline_flag: bool
    task_arrival_model: TaskArrivalProb
    seeds: ExperimentSeeds
    num_task_sequence: int = 2
    benchmark_flag: bool = False


@dataclass
class ExperimentData:
    """
    Data struct for storage of experiment related data.
    """
    env: ProcTHOREnvironment
    interruption_prob_fn: float | Callable[[float], float]
    interruption_value_fn: Callable[[frozenset[Fluent]], float] | None
    converted_actions: list[Action]
    converted_goal: Goal
    converted_interrupting_task_dist: tuple[list[Goal], list[float]]
    neg_to_pos_mapping: dict[Fluent, Fluent]


@dataclass
class DashboardData:
    """
    Data struct for storage of dashboard related data.
    """
    dashboard: PlannerDashboard
    dash_env: ProcTHOREnvironment
    recording_console: Console | None
    start_time: float
    show_plot: bool = False
    save_plot: str | None = None
    save_video: str | None = None


def run_experiment(
    config: ExperimentConfig,
    show_plot: bool = False,
    save_plot: str | None = None,
    save_video: str | None = None
) -> dict:
    """
    Runs a single experiment, given the parameters specified in main.
    Returns the total cost of the execution sequence, the initial plan, and a trace of 
    the execution sequence.
    """
    experiment_data = initialize_experiment_data(config)

    # set the experiment seed after loading the ProcTHOR scene since it sets the
    # seed to procthor_seed
    random.seed(config.seeds.experiment_seed)

    recording_console = (
        Console(record=True, force_terminal=True, width=120) if config.benchmark_flag else None
    )

    start_time = time.perf_counter()

    event_trace, _ = _get_task_sequence_event_trace(experiment_data, config)

    # setup for deterministic replay for dashboard
    dash_env = construct_procthor_kitchen_environment(
        config.seeds.procthor_seed, config.seeds.object_placement_seed
    )

    pos_goal = convert_goal_to_positive_preconditions(
        config.goal, experiment_data.neg_to_pos_mapping
    )
    dashboard = _setup_dashboard(
        pos_goal, dash_env, recording_console,
        partial(AstarDashboardPlanner, heuristic_fn=ff_heuristic), config.benchmark_flag
    )
    execute_deterministic_replay(event_trace, dashboard, dash_env)
    dash_data = DashboardData(
        dashboard, dash_env, recording_console,
        start_time, show_plot, save_plot, save_video
    )
    return _get_results(event_trace, dash_data)


# helper functions
def _get_task_sequence_event_trace(
    data: ExperimentData,
    config: ExperimentConfig
) -> tuple[list[str], list[Goal]]:
    """
    Searchs for a plan and executes the plan in the environment for the specificed n length
    task sequence. Additionally, stores a trace of the envents, in the form a of list of action
    names, which can later be deterministically replayed by the AstarDashboardPlanner.
    """
    event_trace = []
    task_arrival_sequence = [config.goal]

    for i in range(config.num_task_sequence - 1):
        plan, _, _ = astar_search(
            convert_state_to_positive_preconditions(data.env.state, data.neg_to_pos_mapping),
            data.converted_goal,
            data.converted_actions,
            data.converted_interrupting_task_dist if not config.baseline_flag else None,
            ff_heuristic,
            data.interruption_prob_fn if not config.baseline_flag else 0,
            data.interruption_value_fn,
            0,
            num_steps=300000,
            print_trace=False
        )

        _execution_loop(plan, data, event_trace)

        # setup for next task in the sequence
        task_arrival_sequence.append(config.interrupting_task_dist[0][i])
        data.converted_goal = data.converted_interrupting_task_dist[0][i]

        # check if current task was completed successfully
        if task_arrival_sequence[i].evaluate(data.env.state.fluents):
            event_trace[-1] += " | Goal Complete"

    # complete the last task in the sequence under the assumption that no future tasks will come
    plan, _, _ = astar_search(
        convert_state_to_positive_preconditions(data.env.state, data.neg_to_pos_mapping),
        data.converted_goal,
        data.converted_actions,
        None,
        ff_heuristic,
        0,
        None,
        0,
        num_steps=300000,
        print_trace=False
    )

    _execution_loop(plan, data, event_trace, True)

    # check if last task was completed successfully
    if task_arrival_sequence[-1].evaluate(data.env.state.fluents):
        event_trace[-1] += " | Goal Complete"

    return event_trace, task_arrival_sequence


def _execution_loop(
    plan: list[Action],
    data: ExperimentData,
    event_trace: list[str],
    last_task_flag: bool = False
) -> None:
    """
    Helper function for the execution loop of task interruption experiments.
    """
    # execution loop
    for converted_action in plan:
        action = get_action_by_name(data.env.get_actions(), converted_action.name)
        # perform the action
        data.env.act(action)
        event_trace.append(converted_action.name)

        task_arrival_prob = (
            data.interruption_prob_fn
            if isinstance(data.interruption_prob_fn, (float, int))
            else data.interruption_prob_fn(get_action_cost(action))
        )
        # check if interrupting task arrived
        if not last_task_flag and random.random() < task_arrival_prob:
            event_trace[-1] += " | Interrupt"
            break


def initialize_experiment_data(config: ExperimentConfig) -> ExperimentData:
    """
    Helper function for initializing the ExperimentData struct, which entails
    the loading of the environment, setting up the task arrival prob function,
    setting up the interruption value function, and performing negative fluent
    preprocessing.
    """
    env = construct_procthor_kitchen_environment(
            config.seeds.procthor_seed, object_seed=config.seeds.object_placement_seed
    )

    # convert generic name of objects/containers in goals to actual names of
    # objects/containers within the scene (ex. spoon_14)
    # duplicate objects/containers are randomly sampled
    scene_objects = _get_scene_objects_locations(
        config.seeds.duplicate_objects_containers_seed,
        env.scene.objects
    )
    scene_locations = _get_scene_objects_locations(
        config.seeds.duplicate_objects_containers_seed,
        {loc for loc in env.scene.locations if loc != "start_loc"}
    )
    config.goal = _map_goal_to_scene(config.goal, scene_objects, scene_locations)
    # don't directly modify the generic interrupting task distribution
    task_dist = ([], [])
    for goal, prob in zip(*config.interrupting_task_dist):
        task_dist[0].append(_map_goal_to_scene(goal, scene_objects, scene_locations))
        task_dist[1].append(prob)
    config.interrupting_task_dist = task_dist

    # fluent pre-processing for usage of FF heuristic
    converted_actions, _, converted_goals, mapping= negative_fluent_preprocessing(
        env.get_actions(), env.state, [config.goal] + list(config.interrupting_task_dist[0])
    )

    # update config struct to store the new converted goal and task distribution
    converted_goal = converted_goals[0]
    converted_interrupting_task_dist = (list(converted_goals[1:]), config.interrupting_task_dist[1])

    interruption_prob_fn = partial(
        get_task_arrival_prob, config.task_arrival_model.rv_type,
        config.task_arrival_model.interruption_prob,
        config.task_arrival_model.distribution_type,
        max(get_action_cost(act) for act in converted_actions)
    )

    # for learned function
    interruption_value_fn = (
        partial(handcrafted_interruption_value, config.task_arrival_model.interruption_prob)
        if not config.baseline_flag
        else None
    )

    return ExperimentData(
        env, interruption_prob_fn, interruption_value_fn, converted_actions,
        converted_goal, converted_interrupting_task_dist,mapping
    )


def _setup_dashboard(
    goal: Goal,
    env: SymbolicEnvironment,
    recording_console: Console | None,
    planner_factory: Callable[[list[Action]], DashboardPlanner] | None,
    is_benchmark: bool
) -> PlannerDashboard:
    """
    Helper function for the setting up the dashboard conditionally based on if the 
    run_experiment function is used for a standalone experiment or as part of a benchmark.
    """
    dashboard = PlannerDashboard(
        goal,
        env,
        planner_factory=planner_factory,
        print_on_exit=not is_benchmark,
        console=recording_console,
    )

    if not is_benchmark:
        dashboard.start()
    return dashboard


def _get_results(
    event_trace: list[str],
    dash_data: DashboardData
) -> dict:
    """
    Helper function for obtaining the conditional necessary results based on if the 
    run_experiment function is used for a standalone experiment or as part of a benchmark.
    """
    result = {
        "success": "Goal Complete" in event_trace[-1],
        "wall_time": time.perf_counter() - dash_data.start_time,
        "plan_cost": float(dash_data.dash_env.state.time),
        "actions_count": len(event_trace),
        "actions": event_trace,
        "log_html": "",
        "log_plot": b''
    }

    if dash_data.recording_console: # if recording_console is not None, this indicates benchmark
        dash_data.dashboard.print_history()
        html_output = dash_data.recording_console.export_html(inline_styles=True)
        result["log_html"] = html_output

        if isinstance(dash_data.dash_env, ProcTHOREnvironment):
            try:
                location_coords = {
                    name: (float(coord[0]), float(coord[1]))
                    for name, coord in dash_data.dash_env.scene.locations.items()
                }
                plot_image = dash_data.dashboard.get_plot_image(location_coords=location_coords)
                if plot_image is not None:
                    result["log_plot"] = plot_image
            except Exception as e:
                print(f"Failed to render trajectory plot: {e}")
    else:
        dash_data.dashboard.stop()
        dash_data.dashboard.show_plots(
            show_plot=dash_data.show_plot,
            save_video=dash_data.save_video,
            save_plot=dash_data.save_plot
        )

    return result


def execute_deterministic_replay(
    event_trace: list[str],
    dashboard: PlannerDashboard,
    dash_env: SymbolicEnvironment
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
        action = get_action_by_name(dash_env.get_actions(), action_name)
        dash_env.act(action)
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
    the scene (ex. spoon_14).
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

        fluent_str = (
            f"not {fluent.name} {' '.join(args)}"
            if fluent.negated
            else f"{fluent.name} {' '.join(args)}"
        )
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
