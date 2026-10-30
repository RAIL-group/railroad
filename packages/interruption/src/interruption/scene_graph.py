"""
Keeping a ProcTHOR scene graph up to date with the environment's state, and
re-dealing its objects over the scene's containers.
"""
import random
from typing import Sequence

from railroad.core import (
    Action,
    State,
)
from railroad.core import Fluent as F
from railroad.environment.procthor.scenegraph import SceneGraph


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

    if action_type.startswith(("pick-", "place-")):
        obj_idx = int(action_split[-1].split("_")[-1])
        loc_idx = int(action_split[-2].split("_")[-1])
        if action_type.startswith("pick-"):
            scene_graph.delete_edge(loc_idx, obj_idx)
            scene_graph.add_edge(robot_idx, obj_idx)
            scene_graph.nodes[obj_idx]["position"] = scene_graph.nodes[robot_idx]["position"]
        else: # action_type == "place-left" or "place-right"
            scene_graph.delete_edge(robot_idx, obj_idx)
            scene_graph.add_edge(loc_idx, obj_idx)
            scene_graph.nodes[obj_idx]["position"] = scene_graph.nodes[loc_idx]["position"]
    else: # action_type == "move"
        new_loc_idx = int(action_split[-1].split("_")[-1])
        scene_graph.nodes[robot_idx]["position"] = scene_graph.nodes[new_loc_idx]["position"]
        # when the robot is holding one or more objects
        _update_held_objects_position(state, scene_graph, robot_idx, action_split[1])


def _update_held_objects_position(
    state: State, scene_graph: SceneGraph, robot_idx: int, robot_name: str
) -> None:
    """
    Helper function for updating the position attribute of object nodes
    that are currently held by the robot. `robot_name` is the environment's
    name for the robot, as the holding-in-* fluents use it (e.g. "robot1"); the
    scene graph's own node for the robot is named differently ("robot").
    """
    for idx in scene_graph.object_indices:
        obj = (
            scene_graph.get_node_name_by_idx(idx) +
            f"_{idx}"
        )
        robot_holding_object = (
            F(f"holding-in-left {robot_name} {obj}") in state.fluents or
            F(f"holding-in-right {robot_name} {obj}") in state.fluents
        )

        if robot_holding_object:
            scene_graph.nodes[idx]["position"] = (
                scene_graph.nodes[robot_idx]["position"]
            )


def get_object_container_slots(scene_graph: SceneGraph) -> list[int]:
    """
    The container occupancy of a scene graph, as a sorted list with one entry
    per object: the index of the container it rests in. Objects not resting
    in a container (e.g. held by the robot) contribute no slot.
    """
    container_indices = set(scene_graph.container_indices)
    slots = []
    for obj_idx in scene_graph.object_indices:
        parent = scene_graph.get_parent_node_idx(obj_idx)
        if parent in container_indices:
            slots.append(parent)
    return sorted(slots)


def permute_object_containers(
    scene_graph: SceneGraph, slots: Sequence[int], rng: random.Random
) -> dict[int, tuple[int, int]]:
    """
    Reassigns every object by randomly permuting an occupancy: `slots`, the
    container occupancy to reproduce (see get_object_container_slots; usually
    a snapshot of the scene's original occupancy, so a shuffle resets drifted
    counts rather than preserving them), with the robot's hands as extra
    slots. If the robot is holding k objects, k randomly chosen entries of
    `slots` are replaced by k robot entries: the same number of objects stay
    held afterwards, though which ones may change, and the container counts
    are `slots` minus those k. Mutates scene_graph in place, mirroring
    get_updated_scene_graph's pick and place edits (edge, and position = the
    container's or the robot's).

    Always moves at least one object. Returns {object idx: (old parent idx,
    new parent idx)} for the objects that moved, a parent being a container or
    the robot. Raises ValueError, before touching the graph, if there is more
    than one robot, if any object is neither in a container nor held, if
    `slots` doesn't have one entry per object or names a non-container, or if
    the resulting occupancy names fewer than two distinct parents (every
    permutation would be identical).
    """
    container_indices = set(scene_graph.container_indices)
    robot_indices = scene_graph.robot_indices
    if len(robot_indices) > 1:
        raise ValueError("permute_object_containers assumes a single robot")
    obj_indices = sorted(scene_graph.object_indices)  # sorted: result depends only on rng
    current = []
    for obj_idx in obj_indices:
        parent = scene_graph.get_parent_node_idx(obj_idx)
        if parent not in container_indices and parent not in robot_indices:
            raise ValueError(f"object node {obj_idx} is neither in a container nor held")
        current.append(parent)
    if len(slots) != len(obj_indices):
        raise ValueError(f"{len(slots)} slots for {len(obj_indices)} objects")
    if not set(slots) <= container_indices:
        raise ValueError("slots name nodes that are not containers")

    num_held = sum(parent in robot_indices for parent in current)
    assigned = sorted(slots)
    for _ in range(num_held):
        assigned.pop(rng.randrange(len(assigned)))
    assigned += robot_indices[:1] * num_held
    if len(set(assigned)) < 2:
        raise ValueError("occupancy names fewer than two distinct parents; nothing to permute")

    while True:  # terminates: >=2 distinct entries means an ordering other than `current` exists
        rng.shuffle(assigned)
        if assigned != current:
            break

    moves = {}
    for obj_idx, old, new in zip(obj_indices, current, assigned):
        if old == new:
            continue
        scene_graph.delete_edge(old, obj_idx)
        scene_graph.add_edge(new, obj_idx)
        scene_graph.nodes[obj_idx]["position"] = scene_graph.nodes[new]["position"]
        moves[obj_idx] = (old, new)
    return moves
