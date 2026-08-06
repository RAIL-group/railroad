import types

from interruption.environments import KitchenProcTHOREnvironment
from railroad.core import Fluent as F, Action
from railroad.environment.procthor.scene import ProcTHORScene
from railroad.environment.procthor.scenegraph import SceneGraph
from railroad.environment.procthor.thor_interface import ThorInterface

# 5-element one-hot node type schema: apartment=0, robot=1, room=2, container=3, object=4
APARTMENT, ROBOT, ROOM, CONTAINER, OBJECT = range(5)


def _node_type(type_idx: int) -> list[int]:
    type_vec = [0, 0, 0, 0, 0]
    type_vec[type_idx] = 1
    return type_vec


def _node(name: str, type_idx: int, position) -> dict:
    return {"id": name, "type": _node_type(type_idx), "position": position, "name": name}


def _bare_kitchen_env(
    scene_graph: SceneGraph,
    fluents: set[F] | None = None,
    grippers: set[str] | None = None
) -> KitchenProcTHOREnvironment:
    """
    Construct a KitchenProcTHOREnvironment without running __init__.

    `objects_by_type` and `state` are read-only properties on the base
    Environment/SymbolicEnvironment classes -- objects_by_type is backed
    by `_objects_by_type`, and `state` is assembled on each access from
    `_fluents`/`_time`/`_active_skills`, so those underlying attributes
    are set directly here rather than the properties themselves.
    """
    env = object.__new__(KitchenProcTHOREnvironment)
    env.scene = object.__new__(ProcTHORScene)
    env.scene._thor = object.__new__(ThorInterface)
    env.scene._thor.scene_graph = scene_graph
    env._objects_by_type = {"gripper": grippers if grippers is not None else set()}
    env._fluents = fluents if fluents is not None else set()
    env._time = 0.0
    env._active_skills = []
    return env


def _build_scene_graph():
    """
    Test scene graph structure:
    apartment -- robot
              -- kitchen (room) -- countertop (container) -- spoon (object)
                                -- shelvingunit (container)
    """
    sg = SceneGraph()
    apt_idx = sg.add_node(_node("apt", APARTMENT, (0, 0)))
    robot_idx = sg.add_node(_node("robot0", ROBOT, (0, 0)))
    sg.add_edge(apt_idx, robot_idx)
    room_idx = sg.add_node(_node("kitchen", ROOM, (2, 2)))
    sg.add_edge(apt_idx, room_idx)
    countertop_idx = sg.add_node(_node("countertop", CONTAINER, (1, 1)))
    sg.add_edge(room_idx, countertop_idx)
    shelvingunit_idx = sg.add_node(_node("shelvingunit", CONTAINER, (5, 5)))
    sg.add_edge(room_idx, shelvingunit_idx)
    spoon_idx = sg.add_node(_node("spoon", OBJECT, (1, 1)))
    sg.add_edge(countertop_idx, spoon_idx)

    idx = types.SimpleNamespace(
        robot=robot_idx, countertop=countertop_idx,
        shelvingunit=shelvingunit_idx, spoon=spoon_idx,
    )
    return sg, idx


def test_update_scene_graph_move_updates_robot_position():
    sg, idx = _build_scene_graph()
    env = _bare_kitchen_env(sg)

    action = Action(
        set(),
        [],
        f"move robot1 start_loc shelvingunit_{idx.shelvingunit}"
    )

    env.update_scene_graph(action)

    assert sg.nodes[idx.robot]["position"] == (5, 5)


def test_update_scene_graph_move_also_updates_held_object_position():
    sg, idx = _build_scene_graph()
    # spoon is currently held (robot-adjacent, no container edge)
    sg.delete_edge(idx.countertop, idx.spoon)
    sg.add_edge(idx.robot, idx.spoon)

    env = _bare_kitchen_env(
        sg,
        fluents={F("hand-full r1-left"), F(f"holding r1-left spoon_{idx.spoon}")},
        grippers={"r1-left"},
    )

    action = Action(
        set(),
        [],
        f"move robot1 start_loc shelvingunit_{idx.shelvingunit}"
    )

    env.update_scene_graph(action)

    assert sg.nodes[idx.robot]["position"] == (5, 5)
    assert sg.nodes[idx.spoon]["position"] == (5, 5)


def test_update_scene_graph_pick_moves_edge_from_container_to_robot():
    sg, idx = _build_scene_graph()
    env = _bare_kitchen_env(sg)

    action = Action(
        set(),
        [],
        f"pick robot1 r1-left countertop_{idx.countertop} spoon_{idx.spoon}"
    )

    env.update_scene_graph(action)

    assert idx.spoon not in sg.get_adjacent_nodes_idx(idx.countertop)
    assert idx.spoon in sg.get_adjacent_nodes_idx(idx.robot)
    assert sg.get_parent_node_idx(idx.spoon) == idx.robot


def test_update_scene_graph_pick_updates_object_position_to_robot_position():
    sg, idx = _build_scene_graph()
    env = _bare_kitchen_env(sg)

    action = Action(
        set(),
        [],
        f"pick robot1 r1-left countertop_{idx.countertop} spoon_{idx.spoon}"
    )

    env.update_scene_graph(action)

    assert sg.nodes[idx.spoon]["position"] == sg.nodes[idx.robot]["position"]


def test_update_scene_graph_place_moves_edge_from_robot_to_container():
    sg, idx = _build_scene_graph()
    sg.delete_edge(idx.countertop, idx.spoon)
    sg.add_edge(idx.robot, idx.spoon)  # currently held

    env = _bare_kitchen_env(sg)

    action = Action(
        set(),
        [],
        f"place robot1 r1-left shelvingunit_{idx.shelvingunit} spoon_{idx.spoon}"
    )

    env.update_scene_graph(action)

    assert idx.spoon not in sg.get_adjacent_nodes_idx(idx.robot)
    assert idx.spoon in sg.get_adjacent_nodes_idx(idx.shelvingunit)
    assert sg.get_parent_node_idx(idx.spoon) == idx.shelvingunit


def test_update_scene_graph_place_updates_object_position_to_container_position():
    sg, idx = _build_scene_graph()
    sg.delete_edge(idx.countertop, idx.spoon)
    sg.add_edge(idx.robot, idx.spoon)
    sg.nodes[idx.spoon]["position"] = sg.nodes[idx.robot]["position"]  # as if just carried by robot

    env = _bare_kitchen_env(sg)

    action = Action(
        set(),
        [],
        f"place robot1 r1-left shelvingunit_{idx.shelvingunit} spoon_{idx.spoon}"
    )

    env.update_scene_graph(action)

    assert sg.nodes[idx.spoon]["position"] == sg.nodes[idx.shelvingunit]["position"]


def test_update_scene_graph_pick_then_place_round_trip():
    sg, idx = _build_scene_graph()
    env = _bare_kitchen_env(sg)

    pick_action = Action(
        set(),
        [],
        name=f"pick robot1 r1-left countertop_{idx.countertop} spoon_{idx.spoon}"
    )


    env.update_scene_graph(pick_action)
    assert sg.get_parent_node_idx(idx.spoon) == idx.robot

    place_action = Action(
        set(),
        [],
        name=f"place robot1 r1-left shelvingunit_{idx.shelvingunit} spoon_{idx.spoon}"
    )

    env.update_scene_graph(place_action)

    assert sg.get_parent_node_idx(idx.spoon) == idx.shelvingunit
    assert idx.spoon not in sg.get_adjacent_nodes_idx(idx.robot)


def test_update_scene_graph_pick_move_place_full_sequence():
    """
    Full realistic sequence: pick the spoon off the countertop, walk it
    to the shelvingunit, then place it there. Checks that the two
    different position-sync paths (pick/place's own inline position set,
    and move's _update_held_objects_position) stay consistent with each
    other across the whole sequence.

    Note: update_scene_graph never mutates fluents itself -- in the real
    system, pick/place's effects (hand-full/holding) would be applied to
    state separately as part of action execution, before/after this
    method runs. Since we're calling update_scene_graph directly, the
    "holding" fluents are set up front to simulate that window.
    """
    sg, idx = _build_scene_graph()
    env = _bare_kitchen_env(
        sg,
        fluents={F("hand-full r1-left"), F(f"holding r1-left spoon_{idx.spoon}")},
        grippers={"r1-left"},
    )

    # 1. pick the spoon off the countertop
    pick_action = Action(
        set(),
        [],
        name=f"pick robot1 r1-left countertop_{idx.countertop} spoon_{idx.spoon}"
    )

    env.update_scene_graph(pick_action)

    assert sg.get_parent_node_idx(idx.spoon) == idx.robot
    assert sg.nodes[idx.spoon]["position"] == sg.nodes[idx.robot]["position"] == (0, 0)

    # 2. move to the shelvingunit -- robot AND the held spoon should both relocate
    move_action = Action(
        set(),
        [],
        name=f"move robot1 start_loc shelvingunit_{idx.shelvingunit}"
    )


    env.update_scene_graph(move_action)

    assert sg.nodes[idx.robot]["position"] == (5, 5)
    assert sg.nodes[idx.spoon]["position"] == (5, 5)
    # still held, graph edge unchanged by move
    assert sg.get_parent_node_idx(idx.spoon) == idx.robot

    # 3. place the spoon on the shelvingunit
    place_action = Action(
        set(),
        [],
        name=f"place robot1 r1-left shelvingunit_{idx.shelvingunit} spoon_{idx.spoon}"
    )


    env.update_scene_graph(place_action)

    assert sg.get_parent_node_idx(idx.spoon) == idx.shelvingunit
    assert idx.spoon not in sg.get_adjacent_nodes_idx(idx.robot)
    assert sg.nodes[idx.spoon]["position"] == sg.nodes[idx.shelvingunit]["position"] == (5, 5)


def test_update_held_objects_position_updates_held_object():
    sg, idx = _build_scene_graph()
    env = _bare_kitchen_env(
        sg,
        fluents={F("hand-full r1-left"), F(f"holding r1-left spoon_{idx.spoon}")},
        grippers={"r1-left"},
    )

    env._update_held_objects_position(idx.robot)

    assert sg.nodes[idx.spoon]["position"] == sg.nodes[idx.robot]["position"]


def test_update_held_objects_position_ignores_gripper_that_is_not_full():
    sg, idx = _build_scene_graph()
    env = _bare_kitchen_env(sg, fluents=set(), grippers={"r1-left"})  # no hand-full fluent

    env._update_held_objects_position(idx.robot)

    assert sg.nodes[idx.spoon]["position"] == (1, 1)  # unchanged, still on countertop


def test_update_held_objects_position_only_updates_the_held_object():
    sg, idx = _build_scene_graph()
    cup_idx = sg.add_node(_node("cup", OBJECT, (5, 5)))
    sg.add_edge(idx.shelvingunit, cup_idx)

    env = _bare_kitchen_env(
        sg,
        fluents={
            F("hand-full r1-left"),
            F(f"holding r1-left spoon_{idx.spoon}"),
            # cup is not held
        },
        grippers={"r1-left"},
    )

    env._update_held_objects_position(idx.robot)

    assert sg.nodes[idx.spoon]["position"] == sg.nodes[idx.robot]["position"]
    assert sg.nodes[cup_idx]["position"] == (5, 5)


def test_update_held_objects_position_handles_multiple_grippers():
    sg, idx = _build_scene_graph()
    cup_idx = sg.add_node(_node("cup", OBJECT, (5, 5)))
    sg.add_edge(idx.shelvingunit, cup_idx)

    env = _bare_kitchen_env(
        sg,
        fluents={
            F("hand-full r1-left"),
            F(f"holding r1-left spoon_{idx.spoon}"),
            F("hand-full r1-right"),
            F(f"holding r1-right cup_{cup_idx}"),
        },
        grippers={"r1-left", "r1-right"},
    )

    env._update_held_objects_position(idx.robot)

    assert sg.nodes[idx.spoon]["position"] == sg.nodes[idx.robot]["position"]
    assert sg.nodes[cup_idx]["position"] == sg.nodes[idx.robot]["position"]
