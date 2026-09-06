"""
Implementations of the interruption-based, myopic, and anticipatory planning planners
for ProcTHOR environments. These planner implementations all utilize the astar_search
function from planner.py
"""
from typing import Optional
from itertools import product
import numpy as np
from shapely import geometry
from railroad.core import (
    Action, State, LiteralGoal, Goal, Fluent as F, convert_goal_to_positive_preconditions
    , ff_heuristic
)
from railroad.environment.procthor.scene import ProcTHORScene
from railroad.environment.procthor.scenegraph import SceneGraph
from railroad.navigation.pathing import get_cost_and_path

from .constants import LAMBDA_ADD, LAMBDA_MAX, LAMBDA_FF, AP_DEBUG
from .planner import InterruptionSearchProblem, PlannerConfig, astar_search

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
    else:
        ff_weight, v_ap_weight = weights
        h_val = ff_weight * ff_heuristic(
            state, goal, actions, LAMBDA_ADD, LAMBDA_MAX, LAMBDA_FF
        ) + v_ap_weight * v_ap
    return h_multi * h_val


# discount functions
def get_no_int_prob(interruption_probs: list[float]) -> float:
    """
    Returns the probability of an interrupting task not arriving,
    based on the level of the search tree.
    """
    no_int_prob = 1
    for prob in interruption_probs:
        no_int_prob*=(1 - prob)
    return no_int_prob


def get_no_int_discount(interruption_probs: list[float], discount_factor: float = 1) -> float:
    """
    Helper function for the case where action costs/heuristic values
    are not discounted by the probility a task arriving during the
    execution of an action.
    """
    return discount_factor ** len(interruption_probs)


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

    for task in sampled_augmented_tasks:
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
