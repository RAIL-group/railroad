import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from railroad.core import (
    Action,
    Goal,
    LiteralGoal,
    State,
    convert_action_effects,
    convert_action_to_positive_preconditions,
    convert_goal_to_positive_preconditions,
    convert_state_to_positive_preconditions,
    create_positive_fluent_mapping,
    extract_negative_goal_fluents,
    extract_negative_preconditions,
    transition,
)
from railroad.core import Fluent as F
from railroad.environment.procthor.resources import get_procthor_10k_dir
from railroad.environment.procthor.scenegraph import SceneGraph
from railroad.environment.procthor.utils import get_generic_name


# global constants/enums
class RandomVariableType(Enum):
    """
    Enumeration for the valid types of a random variable. Ensures that the 
    user can only pass in a valid random variable type for get_interruption_prob function.
    """
    DISCRETE = 1
    CONTINUOUS = 2


class DistributionType(Enum):
    """
    Enumeration for the supported types of distributions for get_task_arrival_prob.
    """
    UNIFORM = 1
    EXPONENTIAL = 2

@dataclass
class TaskArrivalProb:
    """
    Data struct that encapsulates all the user-provided settings
    for the task arrival model.
    """
    interruption_prob: float
    rv_type: RandomVariableType
    distribution_type: DistributionType


# utility functions for interruption anticipatory planning
def get_action_cost(action: Action) -> float:
    """
    Gets the total cost (reward) of performing an action.
    """
    return action.effects[-1].time + action.extra_cost


def get_reward(action: Action, discount_factor: float, additional_reward: float) -> float:
    """
    Generic reward function.
    """
    return get_discounted_value(get_action_cost(action), discount_factor, additional_reward)


def get_discounted_value(value: float, discount_factor: float, additional_value: float) -> float:
    """
    Returns the discounted sum of the value and additional_value.
    """
    return discount_factor * (value + additional_value)


def get_next_state(
    state: State,
    action: Action,
    interrupting_prob_fn: Callable[[float], float] | float = 0
) -> tuple[State, float]:
    """
    Gets the state s' after performing an action a in s. This function
    assumes the state transition is deterministic. Additionally, returns
    the probability of an interrupting task arriving after the transition.
    """
    if isinstance(interrupting_prob_fn, (float, int)):
        interruption_prob = interrupting_prob_fn
    else:
        interruption_prob = interrupting_prob_fn(get_action_cost(action))
    outcomes = transition(state, action)
    assert len(outcomes) == 1
    next_state, prob = outcomes[0]
    assert prob == 1.0
    return next_state, interruption_prob


def negative_fluent_preprocessing(actions: list[Action], state: State, goals: list[Goal]):
    """
    Wrapper function to convert negative fluents to equivalent positive fluents. Important
    when using the FF heuristic.
    """

    # build negative fluent to equivalent positive fluent mapping
    negative_preconditions = extract_negative_preconditions(actions)
    for goal in goals:
        negative_preconditions = negative_preconditions | extract_negative_goal_fluents(goal)
    mapping = create_positive_fluent_mapping(negative_preconditions)

    # convert actions using mapping
    converted_actions = []
    for action in actions:
        action_pos_precond = convert_action_to_positive_preconditions(action, mapping)
        converted_action = convert_action_effects(action_pos_precond, mapping)
        converted_actions.append(converted_action)

    # convert state using mapping
    converted_state = convert_state_to_positive_preconditions(state, mapping)

    # convert goals using mapping
    converted_goals = []
    for goal in goals:
        converted_goal = convert_goal_to_positive_preconditions(goal, mapping)
        converted_goals.append(converted_goal)
    return converted_actions, converted_state, converted_goals, mapping


def get_task_arrival_prob(
    rv_type: RandomVariableType,
    arrival_prob: float,
    distribution_type: DistributionType | None = DistributionType.UNIFORM,
    time_for_prob: float = 100,
    action_time: float = -1,
) -> float:
    """
    Helper function that returns the probability of a task arriving after the execution
    of an action. Supports both per-action (treating the random variable as discrete) and
    per-time-unit (treating the random variable as continuous) probabilities.
    """
    if rv_type == RandomVariableType.DISCRETE or action_time == -1:
        return arrival_prob
    if (
        rv_type == RandomVariableType.CONTINUOUS and
        (
            distribution_type == DistributionType.UNIFORM or
            arrival_prob == 1
        )
    ):
        return min(arrival_prob * action_time, 1.0)
    # case: exponential distribution and arrival_prob != 1
    # arrival_prob is now parameter Beta for the exponential distribution
    beta = _calibrate_beta_parameter(arrival_prob, time_for_prob)
    return 1 - math.exp(-beta * action_time)


def _calibrate_beta_parameter(prob: float, a_t: float) -> float:
    """
    Helper function for computing the value of the beta parameter for the
    CDF of the exponential distribution such the provided time to complete
    an action will have the specified probability.
    Returns the computed Beta parameter when valid inputs provided
    (Prob: [0, 1) and a_t >= 0). Otherwise returns -1 on invalid inputs.
    """
    if prob < 0 or prob >= 1 or a_t < 0:
        return -1
    return -math.log(1 - prob) / a_t


def print_plan(actions: list[str]) -> None:
    """
    Helper function for printing out the best plan in a more
    readable format.
    """
    # print("Best Plan:")
    for i, action in enumerate(actions):
        print(f"{i}. {action}")

# TODO - use for task augmentation case. also add fluent checking
def get_augmented_task_dist(
    current_task: F | LiteralGoal,
    interrupting_task_dist: tuple[list[Goal], list[float]]
) -> tuple[list[Goal], list[float]]:
    """
    Helper function for task augmentation experiments. Given
    the passed in interrupting_task_dist, creates new future
    tasks that include the current task. Does not make changes
    in-place.
    NOTE - currently this function doesn't provide any checking for
    tasks with conflicting goal fluents.
    """
    augmented_tasks = []
    probs = []
    for task, prob in zip(*interrupting_task_dist):
        augmented_tasks.append(current_task & task)
        probs.append(prob)
    return (augmented_tasks, probs)


def randomize_task_distribution_order(
    task_distribution: tuple[Sequence[Goal], list[float]],
    seed: int
) -> tuple[Goal, tuple[list[Goal], list[float]]]:
    """
    Helper function that randomly selects a task from the task
    distribution to be the current task and re-orders the tasks
    in the task distribution.
    """
    tasks, probs = task_distribution
    rng = random.Random(seed)
    idxes = rng.sample(range(len(tasks)), k=len(tasks))
    return tasks[idxes[-1]], ([tasks[i] for i in idxes[:-1]], [probs[i] for i in idxes[:-1]])


# helper functions for ProcTHOR-10k dataset experiments
def filter_procthor_scenes(
    num_rooms: set[int] | None = None,
    room_types: set[str] | None = None,
    locations: set[str] | None = None,
    objects: set[str] | None = None
) -> list[int]:
    """
    Filters the scenes of the ProcTHOR-10k dataset based on 
    the number of rooms in the scene, if 1 or more of the rooms
    in the scene have the desired roomType, and/or the scene contains
    user-specified locations (containers) and objects, which must be
    lowercase strings.
    Returns a list of the indicies of scenes (seeds) that have the 
    desired criteria.
    """
    # load in scene representations of ProcTHOR-10k
    data_dir = get_procthor_10k_dir()
    with open(data_dir / 'data.jsonl', 'r', encoding="utf-8") as f:
        json_list = list(f)

    # when no filter criteria are provided, return a list of all the seeds
    if (
        num_rooms is None and
        room_types is None and
        locations is None and
        objects is None
    ):
        return list(range(len(json_list)))

    filtered_scene_seeds = []
    for seed, scene_json in enumerate(json_list):
        scene = json.loads(scene_json)
        rooms = scene["rooms"]
        containers = scene["objects"]
        if (
            _check_num_rooms(rooms, num_rooms) and
            _check_scene_room_types(rooms, room_types) and
            _check_scene_locations(containers, locations) and
            _check_scene_objects(containers, objects)
        ):
            filtered_scene_seeds.append(seed)

    return filtered_scene_seeds


def _check_num_rooms(rooms: list[dict[str, Any]], num_rooms: set[int] | None) -> bool:
    """
    Helper function for checking if the number of rooms in a ProcTHOR scene
    matches the desired number of rooms.
    Returns True if the user doesn't specify the desired number of rooms or
    if a match is found. Otherwise, returns False.
    """
    return len(rooms) in num_rooms if num_rooms is not None else True


def _check_scene_room_types(rooms: list[dict[str, Any]], room_types: set[str] | None) -> bool:
    """
    Helper function for checking if 1 or more of the rooms in a ProcTHOR
    scene is of the desired type. (E.g., kitchen, bedroom, etc.)
    Returns True if the user doesn't specify the desired number of rooms or
    if a match is found. Otherwise, returns False.
    """
    if room_types is None:
        return True
    return bool([True for room in rooms if room["roomType"] in room_types])


def _check_scene_locations(containers: list[dict[str, Any]], locations: set[str] | None) -> bool:
    """
    Helper function for checking if the ProcTHOR scene contains the 
    desired locations. (E.g., countertop, fridge, etc.)
    Returns True if the user doesn't specify the desired number of rooms or
    if all locations are present. Otherwise, returns False.
    """
    if locations is None:
        return True
    scene_locations = {get_generic_name(container["id"]) for container in containers}
    return locations.issubset(scene_locations)


def _check_scene_objects(containers: list[dict[str, Any]], objects: set[str] | None) -> bool:
    """
    Helper function for checking if the ProcTHOR scene contains the
    desired objects. (E.g., coffeemachine, egg, etc.)
    Returns True if the user doesn't specify the desired number of rooms or
    if all objects are present. Otherwise, returns False.
    """
    if objects is None:
        return True
    scene_objects = {
        get_generic_name(child["id"])
        for container in containers
        for child in container.get("children", [])
    }
    return objects.issubset(scene_objects)


# helper functions for debugging/testing behavior in ProcTHOR environments
def handcrafted_interruption_value(prob_int: float, state_fluents: frozenset[F]) -> float:
    """
    Function used to test the source of the growing planning time required
    when transitioning to ProcTHOR environments.
    """
    # good_fluent_sets = [
    #     {F("holding r1-left spoon_14")},#, F("holding r1-right pan_17")},
    #     {F("holding r1-right spoon_14")},#, F("holding r1-left pan_17")},
    #     {F("at pan_17 shelvingunit_6"), F("holding r1-left spoon_14")},
    #     {F("at pan_17 shelvingunit_6"), F("holding r1-right spoon_14")},
    # ]
    # good_state = bool([1 for fs in good_fluent_sets if fs.issubset(state_fluents)])
    # if prob_int >= 0.1 and good_state:
    #     return -500
    # return 500
    return 0


# helper functions for keeping scene graphs up to date with the state of an environment
def get_updated_scene_graph(
    scene_graph: SceneGraph,
    state: State,
    action: Action
) -> None:
    """
    Helper function for getting an updated scene graph that matches the
    environment's state after the robot took an action.
    Notes: this method assumes a single-robot scenario.
    Also, this function updates the scene_graph in-place (send in a copy
    if you don't want this behavior).
    """
    action_split = action.name.split(" ")
    action_type = action_split[0]
    robot_idx = scene_graph.robot_indices[0]

    if action_type in ["pick", "place"]:
        obj_idx = int(action_split[-1].split("_")[-1])
        loc_idx = int(action_split[-2].split("_")[-1])
        if action_type == "pick":
            scene_graph.delete_edge(loc_idx, obj_idx)
            scene_graph.add_edge(robot_idx, obj_idx)
            scene_graph.nodes[obj_idx]["position"] = scene_graph.nodes[robot_idx]["position"]
        else: # action_type == "place"
            scene_graph.delete_edge(robot_idx, obj_idx)
            scene_graph.add_edge(loc_idx, obj_idx)
            scene_graph.nodes[obj_idx]["position"] = scene_graph.nodes[loc_idx]["position"]
    else: # action_type == "move"
        new_loc_idx = int(action_split[-1].split("_")[-1])
        scene_graph.nodes[robot_idx]["position"] = scene_graph.nodes[new_loc_idx]["position"]
        # when the robot is holding one or more objects
        gripper_names = {fluent.args[0] for fluent in state.fluents if fluent.name == "hand-full"}
        _update_held_objects_position(state, scene_graph, robot_idx, gripper_names)


def _update_held_objects_position(
    state: State, scene_graph: SceneGraph, robot_idx: int, grippers: set[str]
) -> None:
    """
    Helper function for updating the position attribute of object nodes
    that are currently held by the robot.
    """
    for gripper in grippers:
        # check if gripper is holding an object
        if F(f"hand-full {gripper}") in state.fluents:
            obj_idxs = scene_graph.object_indices
            for idx in obj_idxs:
                obj = (
                    scene_graph.get_node_name_by_idx(idx) +
                    f"_{idx}"
                )
                if F(f"holding {gripper} {obj}") in state.fluents:
                    scene_graph.nodes[idx]["position"] = (
                        scene_graph.nodes[robot_idx]["position"]
                    )
