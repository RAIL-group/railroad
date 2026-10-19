"""
Implementations of the interruption-based, myopic, and anticipatory planning planners
for ProcTHOR environments. These planner implementations all utilize the astar_search
function from planner.py
"""
import random
from enum import Enum
from typing import Optional, Callable
from itertools import product, combinations
import numpy as np
from shapely import geometry
from railroad.core import (
    Action, State, LiteralGoal, Goal, Fluent as F, convert_goal_to_positive_preconditions
    , ff_heuristic, convert_state_to_positive_preconditions, AndGoal
)
from railroad.environment.procthor.scene import ProcTHORScene
from railroad.environment.procthor.scenegraph import SceneGraph
from railroad.navigation.pathing import get_cost_and_path

from .constants import (
    LAMBDA_ADD, LAMBDA_MAX, LAMBDA_FF, AP_DEBUG, NUM_AUGMENTED_TASK_SAMPLES,
    AP_SEED
)
from .environments import KitchenProcTHOREnvironment
from .operators import BLOCKING_PREDICATES, only_clears_blocking_facts
from .planner import (
    InterruptionSearchProblem, PlannerConfig, SearchStats, astar_search,
    filter_actions_to_objects, goal_objects, held_objects,
)


class PlannerMode(Enum):
    """
    Enumeration for interruption-based and baseline planners A*
    Search settings.
    """
    MYOPIC = 1
    ANTICIPATORY_PLANNING = 2
    INTERRUPTION = 3
    INTERRUPTION_AP = 4


# wrapper heuristic functions
def ap_heuristic_fn(
    state: State,
    goal: Goal,
    actions: list[Action],
    v_ap: float = 0,
    weights: Optional[tuple[float, float]] = None,
    h_multi: float = 1
) -> float:
    """
    A wrapper of the ff_heuristic function for incorporating the expected
    value over a task distribution resulting from a learned function into
    the heuristic. When the weights argument is provided, the first element
    of the tuple is the weight of the ff-heuristic value and the second elemnt
    is the weight of the v_ap term. Additionally, when a value for weights is not
    provided, the wrapper simply returns the value of the ff-heuristic.
    """
    if weights is None:
        h_val = ff_heuristic(state, goal, actions, LAMBDA_ADD, LAMBDA_MAX, LAMBDA_FF)
        return h_multi * h_val
    ff_weight, v_ap_weight = weights
    # NOTE: temporary fix
    assert ff_weight == 1 and v_ap_weight == 1
    return h_multi * ff_heuristic(
        state, goal, actions, LAMBDA_ADD, LAMBDA_MAX, LAMBDA_FF
    ) + v_ap

def anticipatory_planner(
    initial_state: tuple[State, SceneGraph],
    interruption_problem: InterruptionSearchProblem,
    search_params: PlannerConfig,
    scene: ProcTHORScene,
    neg_to_pos_mapping: dict[F, F]
) -> tuple[list[Action], float, bool]:
    """
    Implementation of the anticipatory planning framework for large-scale environments from
    Talukder et al.
    """
    rng = random.Random(AP_SEED)

    best_plan = []
    best_value_sg = float("inf")
    best_value_total = float("inf")

    # set search_params to match a myopic planner
    assert search_params.interruption_value_fn is not None
    ev_model = search_params.interruption_value_fn
    search_params.interruption_value_fn = None

    # run myopic planner to get initial plan/value of s_g
    plan, value_sg, success, scene_graph_sg = astar_search(
        initial_state, interruption_problem, search_params
    )
    assert scene_graph_sg is not None

    # task was able to be solved
    if success:
        best_plan = plan
        best_value_sg = value_sg
        best_value_total = value_sg + ev_model(scene_graph_sg)

    # for debugging
    if AP_DEBUG:
        print(f"Total costs to reach augmented goal state: {best_value_sg:.4f}")
        print(f"V_AP of augmented goal state: {ev_model(scene_graph_sg):.4f}")
        print(f"V_s_g + V_AP = {best_value_total:.4f}")

    assert isinstance(interruption_problem.goal, Goal)
    non_augmented_goal = interruption_problem.goal

    selected_locations, selected_objects = focused_sampling(
        interruption_problem.goal,
        best_plan,
        scene.grid,
        scene.locations,
        scene.object_locations
    )

    # get sampled augmentated tasks
    sampled_augmented_tasks = _get_sampled_augmented_tasks(
        interruption_problem.goal, selected_locations, selected_objects, neg_to_pos_mapping
    )
    random_sampled_augmented_tasks = (
        rng.sample(sampled_augmented_tasks, k=NUM_AUGMENTED_TASK_SAMPLES)
        if len(sampled_augmented_tasks) > NUM_AUGMENTED_TASK_SAMPLES
        else sampled_augmented_tasks
    )

    for task in random_sampled_augmented_tasks:
        interruption_problem.goal = task
        # run myopic planner to get initial plan/value of s_g
        plan, value_sg, success, scene_graph_sg = astar_search(
            initial_state, interruption_problem, search_params
        )

        assert scene_graph_sg is not None
        value_total = value_sg + ev_model(scene_graph_sg)

        if AP_DEBUG:
            print(f"Total costs to reach augmented goal state: {value_sg:.4f}")
            print(f"V_AP of augmented goal state: {ev_model(scene_graph_sg):.4f}")
            print(f"V_s_g + V_AP = {value_total:.4f}")

        if success and value_total < best_value_total:
            best_value_total = value_total
            best_value_sg = value_sg
            best_plan = plan

    # restore the interruption_value_fn and non_augmented goal attributes for future tasks
    search_params.interruption_value_fn = ev_model
    interruption_problem.goal = non_augmented_goal

    return best_plan, best_value_sg, best_value_total != float("inf")


def focused_sampling(
    task: Goal,
    symbolic_plan: list[Action],
    occupancy_grid: np.ndarray,
    locations: dict[str, tuple[int, int]],
    objects_by_location: dict[str, set[str]],
) -> tuple[set[str], set[str]]:
    """
    Helper function for implementing focused sampling from Ridwan's paper
    using railroad instead of antplan package.
    Returns the selected containers and selected objects
    """
    task_relevant_objects = _get_task_relevant_objects(task)

    # get paths through the occupancy grid for src and dest locations in the symbolic plan
    paths = _get_occupany_grid_paths(symbolic_plan, occupancy_grid, locations)

    # check if any of the containers are on the path
    locations_close_to_path = set()
    objects_close_to_path = set()

    for name, pos in locations.items():
        if name != "start_loc" and check_intersects(paths, pos):
            locations_close_to_path.add(name)
            location_objects = objects_by_location.get(name, set())
            # exclude object that is already part of the current task
            # location_objects.discard(task.fluent().args[0])
            location_objects.difference_update(task_relevant_objects)
            objects_close_to_path |= location_objects

    return locations_close_to_path, objects_close_to_path


def _get_occupany_grid_paths(
    symbolic_plan: list[Action], occupancy_grid: np.ndarray, locations: dict[str, tuple[int, int]]
) -> list:
    """
    Helper function for focused_sampling.
    """
    # get paths through the occupancy grid for src and dest locations in the symbolic plan
    paths = []
    for action in symbolic_plan:
        components_of_action = action.name.split(" ")
        if components_of_action[0] == "move":
            # assumes that the last two words in a move action are the src and dest locations
            src_loc, dest_loc = [locations[loc_name] for loc_name in components_of_action[-2:]]

            # compute path
            _, points_on_path = get_cost_and_path(
                occupancy_grid,
                src_loc,
                dest_loc,
                use_soft_cost=True,
                unknown_as_obstacle=False,
                soft_cost_scale=12.0,
            )
            path = geometry.LineString(points_on_path.T)
            # might want to tune the distance, but fine for now
            paths.append(path.buffer(25))
    return paths


def check_intersects(paths, val):
    """
    Helper function that checks if a point intersects any of the paths.
    """
    for path in paths:
        if path.contains(geometry.Point(val)):
            return True
    return False


def _get_sampled_augmented_tasks(
    current_task: Goal,
    locations: set[str],
    objects: set[str],
    mapping: dict[F, F]
) -> list[Goal]:
    # NOTE: for this instance, differing iteration orders of the locations and objects
    # doesn't matter because all the augmented tasks will be evaluated eventually.
    augmented_tasks = []
    for obj, loc in product(objects, locations):
        augmented_tasks.append(
            convert_goal_to_positive_preconditions(current_task & F(f"at {obj} {loc}"), mapping)
        )
    return augmented_tasks


def _get_task_relevant_objects(task: Goal) -> set[str]:
    # get objects that are relevant to the current task
    task_relevant_objects = set()
    if isinstance(task, LiteralGoal):
        task_relevant_objects.add(task.fluent().args[0])
    else: # NOTE: written for AndGoals in mind
        for child_goal in task.children():
            assert isinstance(child_goal, LiteralGoal)
            task_relevant_objects.add(child_goal.fluent().args[0])
    return task_relevant_objects


# wrapper search functions
def _planning_state(state: State, neg_to_pos_mapping: dict[F, F]) -> State:
    """
    The state a plan starts from: the environment's state without blocking facts
    or their pending clean-ups (see operators.BLOCKING_PREDICATES), converted to
    positive preconditions. Blocking keeps a plan from moving twice in a row, but
    a new plan may need to start with a move even if the robot has just moved.
    """
    unblocked = State(
        state.time,
        {fluent for fluent in state.fluents if fluent.name not in BLOCKING_PREDICATES},
        [(time, effect) for time, effect in state.upcoming_effects
         if not only_clears_blocking_facts(effect)],
    )
    return convert_state_to_positive_preconditions(unblocked, neg_to_pos_mapping)


def search_without_retry(
    planner_mode: PlannerMode,
    environment: KitchenProcTHOREnvironment,
    search_problem: InterruptionSearchProblem,
    planner_parameters: PlannerConfig,
    neg_to_pos_mapping: dict[F, F],
    stats: SearchStats | None = None,
) -> tuple[list[Action], bool, int, int]:
    """
    Wrapper function of the planning algorithms used in the task sequence
    with continous task arrival experiments. Returns a tuple containing the
    plan, success_flag, the number of failed searches and the number of searches
    retried over goal objects (both always 0; see search_with_retry). stats, when
    given, is filled in by the A* search (not by the anticipatory planner).
    """
    initial_state = _planning_state(environment.state, neg_to_pos_mapping)

    if planner_mode == PlannerMode.ANTICIPATORY_PLANNING:
        plan, _, success = anticipatory_planner(
            (initial_state, environment.scene.scene_graph),
            search_problem,
            planner_parameters,
            environment.scene,
            neg_to_pos_mapping
        )
    else:
        state_scene_graph = (
            initial_state,
            None if planner_mode == PlannerMode.MYOPIC
            else environment.scene.scene_graph
        )

        plan, _, success, _ = astar_search(
            state_scene_graph, search_problem, planner_parameters, stats=stats
        )
    return plan, success, 0, 0


def search_with_retry(
    planner_mode: PlannerMode,
    environment: KitchenProcTHOREnvironment,
    search_problem: InterruptionSearchProblem,
    planner_parameters: PlannerConfig,
    neg_to_pos_mapping: dict[F, F],
    allow_decomposition: bool = True,
    search_stats: list[SearchStats] | None = None,
) -> tuple[list[Action], bool, int, int]:
    """
    A wrapper function for planning algorithms utilized as part of interruption
    experiments. If a task is too complex to be solved within the alotted search
    budget, the same sub-task is first retried over only the actions involving its
    own objects; if that also fails, the complex task is decomposed into a smaller
    sub-task to be solved. Gives up, unsuccessfully, once no single-goal sub-task
    can be solved, or straight after the goal-object retry when allow_decomposition
    is False (when only a plan for the whole task will do). Returns a tuple
    containing the plan, success_flag, the number of failed searches, and the number
    of searches retried over goal objects. search_stats, when given, gets a
    SearchStats per A* search.
    """
    initial_state = _planning_state(environment.state, neg_to_pos_mapping)
    num_goals = (
        _get_task_goal_count(search_problem.goal)
        if planner_parameters.max_task_complexity == -1
        else planner_parameters.max_task_complexity
    )
    initial_num_goals = num_goals
    task = search_problem.goal
    all_objects = environment.scene.objects
    failure_count = 0
    filtered_searches = 0

    while True:
        # find easiest to accomplish sub-task
        ranked_sub_goals = _decompose_task(
            initial_state, task,
            search_problem.actions,
            planner_parameters.heuristic_fn,
            min(num_goals, _get_task_goal_count(search_problem.goal))
        )
        search_problem.goal = ranked_sub_goals[0]

        stats = _new_stats(search_stats)
        plan, success, *_ = search_without_retry(
            planner_mode, environment, search_problem,
            planner_parameters, neg_to_pos_mapping, stats
        )

        # retry the same sub-task over only its own objects before shrinking it
        if not success:
            failure_count += 1
            filtered_searches += 1
            plan, success = _search_over_goal_objects(
                planner_mode, environment, search_problem,
                planner_parameters, neg_to_pos_mapping, all_objects,
                _new_stats(search_stats, filtered=True),
            )

        if success:
            break

        # solver didn't successfully find a plan
        failure_count += 1
        # a single goal can't be decomposed further; stopping here also keeps the
        # complexity limit at 1 or more, which _decompose_task requires
        if not allow_decomposition or num_goals == 1:
            break
        num_goals -= 1

    # restore the original task
    search_problem.goal = task

    # keep track of the maximum number of goals within a task the planner can solve
    if num_goals < initial_num_goals:
        planner_parameters.max_task_complexity = num_goals

    return plan, success, failure_count, filtered_searches


def _search_over_goal_objects(
    planner_mode: PlannerMode,
    environment: KitchenProcTHOREnvironment,
    search_problem: InterruptionSearchProblem,
    planner_parameters: PlannerConfig,
    neg_to_pos_mapping: dict[F, F],
    all_objects: set[str],
    stats: SearchStats | None = None,
) -> tuple[list[Action], bool]:
    """
    search_without_retry over only the actions involving the current goal's objects
    and any object the robot is holding, which it may need to put down first (see
    planner.filter_actions_to_objects). search_problem is shared across calls, so
    its actions are restored afterwards, like the goal in search_with_retry.
    """
    actions = search_problem.actions
    keep = goal_objects(search_problem.goal, all_objects) | held_objects(
        environment.state, all_objects
    )
    search_problem.actions = filter_actions_to_objects(actions, keep, all_objects)
    try:
        plan, success, *_ = search_without_retry(
            planner_mode, environment, search_problem, planner_parameters, neg_to_pos_mapping,
            stats,
        )
    finally:
        search_problem.actions = actions
    return plan, success


def _new_stats(
    search_stats: list[SearchStats] | None, filtered: bool = False
) -> SearchStats | None:
    """A SearchStats appended to search_stats, or None when stats aren't being collected."""
    if search_stats is None:
        return None
    stats = SearchStats(filtered=filtered)
    search_stats.append(stats)
    return stats


def _decompose_task(
    state: State,
    goal: Goal,
    actions: list[Action],
    heuristic_fn: Callable[[State, Goal, list[Action], float], float] | float,
    num_goals_in_subtask: int
) -> list[Goal]:
    """
    Helper function for decomposing a complex task into a list of sub-tasks,
    in order of estimated cost to-go. 
    """
    assert num_goals_in_subtask > 0
    # NOTE: no support for raw scalar heuristic functions
    assert not isinstance(heuristic_fn, (float, int))
    sub_tasks = combinations(
        [goal] if isinstance(goal, LiteralGoal) else goal.children(),
        num_goals_in_subtask
    )
    return sorted(
        [
            sub_task[0] if len(sub_task) == 1 else AndGoal(sub_task)
            for sub_task in sub_tasks
        ], key=lambda x: heuristic_fn(state, x, actions, 0)
    )


def _get_task_goal_count(goal: Goal):
    return 1 if isinstance(goal, LiteralGoal) else len(goal.children())
