"""
Implementations of the interruption-based, myopic, and anticipatory planning planners
for ProcTHOR environments. These planner implementations all utilize the astar_search
function from planner.py
"""
from itertools import product
import numpy as np
from shapely import geometry
from railroad.core import (
    Action, State, LiteralGoal, Goal, Fluent as F, convert_goal_to_positive_preconditions
)
from railroad.environment.procthor.scene import ProcTHORScene
from railroad.environment.procthor.scenegraph import SceneGraph
from railroad.navigation.pathing import get_cost_and_path

from .planner import InterruptionSearchProblem, PlannerConfig, astar_search


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


def get_no_int_discount(interruption_probs: list[float]) -> float:
    """
    Helper function for the case where action costs/heuristic values
    are not discounted by the probility a task arriving during the
    execution of an action.
    """
    return 1


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
    # set search_params to match a myopic planner
    assert search_params.interruption_value_fn is not None
    ev_model = search_params.interruption_value_fn
    search_params.interruption_value_fn = None

    # run myopic planner to get initial plan/value of s_g
    best_plan, best_value_sg, _, scene_graph_sg = astar_search(
        initial_state, interruption_problem, search_params
    )
    
    assert scene_graph_sg is not None
    best_value_total = best_value_sg + ev_model(scene_graph_sg)

    # for debugging
    print(f"Total costs to reach augmented goal state: {best_value_sg:.4f}")
    print(f"V_AP of augmented goal state: {ev_model(scene_graph_sg):.4f}")
    print(f"V_s_g + V_AP = {best_value_total:.4f}")

    # focused sampling; NOTE - right now only supports literal goals
    assert isinstance(interruption_problem.goal, LiteralGoal)
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
        plan, value_sg, _, scene_graph_sg = astar_search(
            initial_state, interruption_problem, search_params
        )

        assert scene_graph_sg is not None
        value_total = value_sg + ev_model(scene_graph_sg)

        # for debugging
        print(f"Total costs to reach augmented goal state: {value_sg:.4f}")
        print(f"V_AP of augmented goal state: {ev_model(scene_graph_sg):.4f}")
        print(f"V_s_g + V_AP = {value_total:.4f}")

        if value_total < best_value_total:
            best_value_total = value_total
            best_value_sg = value_sg
            best_plan = plan

    return best_plan, best_value_sg, True


def focused_sampling(
    task: LiteralGoal,
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
            location_objects.discard(task.fluent().args[0])
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
            # TODO - might want to tune the distance, but fine for now
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
