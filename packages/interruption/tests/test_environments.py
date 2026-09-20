import random
import types
from collections import Counter

import pytest

from interruption.environments import KitchenProcTHOREnvironment, get_alfred_task_distribution
from interruption.utilities import (
    _update_held_objects_position,
    get_object_container_slots,
    permute_object_containers,
)
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
        f"move robot0 start_loc shelvingunit_{idx.shelvingunit}"
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
        fluents={F("left-hand-full robot0"), F(f"holding-in-left robot0 spoon_{idx.spoon}")},
    )

    action = Action(
        set(),
        [],
        f"move robot0 start_loc shelvingunit_{idx.shelvingunit}"
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
        f"pick-left robot0 countertop_{idx.countertop} spoon_{idx.spoon}"
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
        f"pick-left robot0 countertop_{idx.countertop} spoon_{idx.spoon}"
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
        f"place-left robot0 shelvingunit_{idx.shelvingunit} spoon_{idx.spoon}"
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
        f"place-left robot0 shelvingunit_{idx.shelvingunit} spoon_{idx.spoon}"
    )

    env.update_scene_graph(action)

    assert sg.nodes[idx.spoon]["position"] == sg.nodes[idx.shelvingunit]["position"]


def test_update_scene_graph_pick_then_place_round_trip():
    sg, idx = _build_scene_graph()
    env = _bare_kitchen_env(sg)

    pick_action = Action(
        set(),
        [],
        name=f"pick-left robot0 countertop_{idx.countertop} spoon_{idx.spoon}"
    )


    env.update_scene_graph(pick_action)
    assert sg.get_parent_node_idx(idx.spoon) == idx.robot

    place_action = Action(
        set(),
        [],
        name=f"place-left robot0 shelvingunit_{idx.shelvingunit} spoon_{idx.spoon}"
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
        fluents={F("left-hand-full robot0"), F(f"holding-in-left robot0 spoon_{idx.spoon}")},
    )

    # 1. pick the spoon off the countertop
    pick_action = Action(
        set(),
        [],
        name=f"pick-left robot0 countertop_{idx.countertop} spoon_{idx.spoon}"
    )

    env.update_scene_graph(pick_action)

    assert sg.get_parent_node_idx(idx.spoon) == idx.robot
    assert sg.nodes[idx.spoon]["position"] == sg.nodes[idx.robot]["position"] == (0, 0)

    # 2. move to the shelvingunit -- robot AND the held spoon should both relocate
    move_action = Action(
        set(),
        [],
        name=f"move robot0 start_loc shelvingunit_{idx.shelvingunit}"
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
        name=f"place-left robot0 shelvingunit_{idx.shelvingunit} spoon_{idx.spoon}"
    )


    env.update_scene_graph(place_action)

    assert sg.get_parent_node_idx(idx.spoon) == idx.shelvingunit
    assert idx.spoon not in sg.get_adjacent_nodes_idx(idx.robot)
    assert sg.nodes[idx.spoon]["position"] == sg.nodes[idx.shelvingunit]["position"] == (5, 5)


def test_update_held_objects_position_updates_held_object():
    sg, idx = _build_scene_graph()
    env = _bare_kitchen_env(
        sg,
        fluents={F("left-hand-full robot0"), F(f"holding-in-left robot0 spoon_{idx.spoon}")},
    )

    _update_held_objects_position(
        env.state,
        env.scene._thor.scene_graph,
        idx.robot,
        "robot0",
    )

    assert sg.nodes[idx.spoon]["position"] == sg.nodes[idx.robot]["position"]


def test_update_held_objects_position_ignores_gripper_that_is_not_full():
    sg, idx = _build_scene_graph()
    env = _bare_kitchen_env(sg, fluents=set())  # no hand-full fluent

    _update_held_objects_position(
        env.state,
        env.scene._thor.scene_graph,
        idx.robot,
        "robot0",
    )

    assert sg.nodes[idx.spoon]["position"] == (1, 1)  # unchanged, still on countertop


def test_update_held_objects_position_only_updates_the_held_object():
    sg, idx = _build_scene_graph()
    cup_idx = sg.add_node(_node("cup", OBJECT, (5, 5)))
    sg.add_edge(idx.shelvingunit, cup_idx)

    env = _bare_kitchen_env(
        sg,
        fluents={
            F("left-hand-full robot0"),
            F(f"holding-in-left robot0 spoon_{idx.spoon}"),
            # cup is not held
        },
    )

    _update_held_objects_position(
        env.state,
        env.scene._thor.scene_graph,
        idx.robot,
        "robot0",
    )

    assert sg.nodes[idx.spoon]["position"] == sg.nodes[idx.robot]["position"]
    assert sg.nodes[cup_idx]["position"] == (5, 5)


def test_update_held_objects_position_handles_multiple_grippers():
    sg, idx = _build_scene_graph()
    cup_idx = sg.add_node(_node("cup", OBJECT, (5, 5)))
    sg.add_edge(idx.shelvingunit, cup_idx)

    env = _bare_kitchen_env(
        sg,
        fluents={
            F("left-hand-full robot0"),
            F(f"holding-in-left robot0 spoon_{idx.spoon}"),
            F("right-hand-full robot0"),
            F(f"holding-in-right robot0 cup_{cup_idx}"),
        },
    )

    _update_held_objects_position(
        env.state,
        env.scene._thor.scene_graph,
        idx.robot,
        "robot0",
    )

    assert sg.nodes[idx.spoon]["position"] == sg.nodes[idx.robot]["position"]
    assert sg.nodes[cup_idx]["position"] == sg.nodes[idx.robot]["position"]


def test_held_object_follows_the_robot_when_the_graph_names_it_differently_than_the_env():
    """In the real scene the graph's robot node is "robot" while the environment
    and its fluents call the robot "robot1"; the held object must follow anyway."""
    sg, idx = _build_scene_graph()
    sg.nodes[idx.robot]["name"] = "robot"
    sg.delete_edge(idx.countertop, idx.spoon)
    sg.add_edge(idx.robot, idx.spoon)  # held
    env = _bare_kitchen_env(
        sg,
        fluents={F("left-hand-full robot1"), F(f"holding-in-left robot1 spoon_{idx.spoon}")},
    )

    env.update_scene_graph(Action(set(), [], f"move robot1 start_loc shelvingunit_{idx.shelvingunit}"))

    assert sg.nodes[idx.robot]["position"] == (5, 5)
    assert sg.nodes[idx.spoon]["position"] == (5, 5)


def test_update_held_objects_position_uses_the_given_robot_name_not_the_graph_nodes():
    sg, idx = _build_scene_graph()
    sg.nodes[idx.robot]["name"] = "robot"
    env = _bare_kitchen_env(sg, fluents={F(f"holding-in-right robot1 spoon_{idx.spoon}")})

    _update_held_objects_position(env.state, sg, idx.robot, "robot1")

    assert sg.nodes[idx.spoon]["position"] == sg.nodes[idx.robot]["position"]


@pytest.mark.parametrize(
        'objects, locations',
        [(
            {
                'knife_19', 'egg_22', 'peppershaker_9', 'papertowelroll_17',
                'tomato_11', 'pen_14', 'potato_13', 'tomato_10', 'pencil_15',
                'egg_21', 'spoon_16', 'apple_23', 'spraybottle_24',
                'pan_18', 'tomato_12', 'tomato_20'
            },
            {
                'fridge_4', 'garbagecan_5', 'countertop_3', 'stool_6',
                'shelvingunit_7', 'stool_8', 'start_loc'
            }
        )]
)
def test_get_alfred_task_distribution_ordering(objects, locations):
    baseline, _ = get_alfred_task_distribution(objects, locations)
    for seed in range(1000):
        reordered_objects = _reordered_set(objects, seed)
        reordered_locations = _reordered_set(locations, seed + 100)
        result, _ = get_alfred_task_distribution(reordered_objects, reordered_locations)
        assert result == baseline, f"order differs for shuffle seed {seed}"


def _reordered_set(items: set[str], seed: int) -> set[str]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    result = set()
    for item in shuffled:
        result.add(item)
    return result


@pytest.mark.parametrize(
        'objects, locations',
        [(
            {
                'knife_19', 'egg_22', 'peppershaker_9', 'papertowelroll_17',
                'tomato_11', 'pen_14', 'potato_13', 'tomato_10', 'pencil_15',
                'egg_21', 'spoon_16', 'apple_23', 'spraybottle_24',
                'pan_18', 'tomato_12', 'tomato_20'
            },
            {
                'fridge_4', 'garbagecan_5', 'countertop_3', 'stool_6',
                'shelvingunit_7', 'stool_8', 'start_loc'
            }
        )]
)
def test_get_alfred_task_distribution_size_matching(objects, locations):
    baseline, probs = get_alfred_task_distribution(objects, locations)

    assert len(baseline) == len(probs)

    baseline, probs = get_alfred_task_distribution(
        objects,
        locations,
        one_object_per_taskdist=True
    )

    assert len(baseline) == len(probs)


def _multi_object_graph():
    """
    Test scene graph: three containers, three objects.
    countertop: spoon, fork -- shelvingunit: cup -- fridge: (empty)
    """
    sg = SceneGraph()
    apt_idx = sg.add_node(_node("apt", APARTMENT, (0, 0)))
    robot_idx = sg.add_node(_node("robot0", ROBOT, (0, 0)))
    sg.add_edge(apt_idx, robot_idx)
    room_idx = sg.add_node(_node("kitchen", ROOM, (2, 2)))
    sg.add_edge(apt_idx, room_idx)
    container = {}
    for name, position in [("countertop", (1, 1)), ("shelvingunit", (5, 5)), ("fridge", (9, 9))]:
        container[name] = sg.add_node(_node(name, CONTAINER, position))
        sg.add_edge(room_idx, container[name])
    obj = {}
    for name, in_container in [
        ("spoon", "countertop"), ("fork", "countertop"), ("cup", "shelvingunit")
    ]:
        position = sg.nodes[container[in_container]]["position"]
        obj[name] = sg.add_node(_node(name, OBJECT, position))
        sg.add_edge(container[in_container], obj[name])
    return sg, types.SimpleNamespace(robot=robot_idx, room=room_idx, container=container, obj=obj)


def _occupancy(sg: SceneGraph) -> Counter:
    return Counter(sg.get_parent_node_idx(i) for i in sg.object_indices)


def _move_object(sg: SceneGraph, obj_idx: int, new_parent: int) -> None:
    parent = sg.get_parent_node_idx(obj_idx)
    assert parent is not None
    sg.delete_edge(parent, obj_idx)
    sg.add_edge(new_parent, obj_idx)


def test_get_object_container_slots_is_the_sorted_occupancy():
    sg, idx = _multi_object_graph()
    c = idx.container
    assert get_object_container_slots(sg) == sorted(
        [c["countertop"], c["countertop"], c["shelvingunit"]]
    )


def test_get_object_container_slots_skips_held_objects():
    sg, idx = _multi_object_graph()
    _move_object(sg, idx.obj["spoon"], idx.robot)
    c = idx.container
    assert get_object_container_slots(sg) == sorted([c["countertop"], c["shelvingunit"]])


def test_permute_object_containers_restores_original_counts_after_drift():
    sg, idx = _multi_object_graph()
    slots = get_object_container_slots(sg)
    for obj_idx in idx.obj.values():  # every task done: all objects piled into the fridge
        _move_object(sg, obj_idx, idx.container["fridge"])

    permute_object_containers(sg, slots, random.Random(0))

    assert _occupancy(sg) == Counter(slots)


@pytest.mark.parametrize("seed", range(10))
def test_permute_object_containers_always_moves_something(seed):
    sg, _ = _multi_object_graph()
    slots = get_object_container_slots(sg)
    before = _occupancy(sg)

    moves = permute_object_containers(sg, slots, random.Random(seed))

    assert moves
    assert _occupancy(sg) == before  # same slots as the current arrangement: counts unchanged


def test_permute_object_containers_reports_exactly_the_moved_objects_and_updates_graph():
    sg, idx = _multi_object_graph()
    slots = get_object_container_slots(sg)
    for obj_idx in idx.obj.values():
        _move_object(sg, obj_idx, idx.container["fridge"])
    parents_before = {i: sg.get_parent_node_idx(i) for i in sg.object_indices}

    moves = permute_object_containers(sg, slots, random.Random(3))

    for obj_idx in sg.object_indices:
        parent = sg.get_parent_node_idx(obj_idx)
        if obj_idx in moves:
            assert moves[obj_idx] == (parents_before[obj_idx], parent)
            assert parent != parents_before[obj_idx]
        else:
            assert parent == parents_before[obj_idx]
        assert sg.nodes[obj_idx]["position"] == sg.nodes[parent]["position"]


def test_permute_object_containers_is_deterministic_for_a_seed():
    sg_a, _ = _multi_object_graph()
    sg_b, _ = _multi_object_graph()
    slots = get_object_container_slots(sg_a)

    permute_object_containers(sg_a, slots, random.Random(11))
    permute_object_containers(sg_b, slots, random.Random(11))

    assert sorted(sg_a.edges) == sorted(sg_b.edges)


@pytest.mark.parametrize("case", ["orphan", "too_few_slots", "non_container_slot", "single_slot"])
def test_permute_object_containers_raises_without_mutating(case):
    sg, idx = _multi_object_graph()
    c = idx.container
    slots = get_object_container_slots(sg)
    if case == "orphan":  # neither in a container nor held
        sg.delete_edge(c["countertop"], idx.obj["spoon"])
    elif case == "too_few_slots":
        slots = slots[:-1]
    elif case == "non_container_slot":
        slots = [idx.room, c["countertop"], c["shelvingunit"]]
    else:
        slots = [c["fridge"]] * 3
    edges, positions = list(sg.edges), {i: n["position"] for i, n in sg.nodes.items()}

    with pytest.raises(ValueError):
        permute_object_containers(sg, slots, random.Random(0))

    assert sg.edges == edges
    assert {i: n["position"] for i, n in sg.nodes.items()} == positions


def test_randomize_object_locations_keeps_every_placement_record_consistent():
    sg, idx = _multi_object_graph()

    def name(i: int) -> str:
        return f"{sg.get_node_name_by_idx(i)}_{i}"

    def by_location() -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for obj_idx in sg.object_indices:
            result.setdefault(name(sg.get_parent_node_idx(obj_idx)), set()).add(name(obj_idx))
        return result

    def fluents_for(locations: dict[str, set[str]]) -> set[F]:
        return {
            F(f"at {obj} {loc}") for loc, objs in locations.items() for obj in objs
        } | {F("free robot0")}

    slots = get_object_container_slots(sg)
    env = _bare_kitchen_env(sg, fluents=fluents_for(by_location()))
    env._objects_by_type["robot"] = {"robot0"}
    env.scene._object_locations = by_location()
    env._objects_at_locations = by_location()
    env._original_container_slots = slots
    # drift as pick/place would: the graph and the fluents follow, but the
    # ground-truth maps stay as constructed (so they are stale when we shuffle)
    for obj_idx in idx.obj.values():
        _move_object(sg, obj_idx, idx.container["fridge"])
    env.fluents.clear()
    env.fluents.update(fluents_for(by_location()))
    assert env.scene.object_locations != by_location()

    env.randomize_object_locations(random.Random(5))

    expected = by_location()
    assert env.fluents == fluents_for(expected)
    assert env.scene.object_locations == expected
    assert env.objects_at_locations == expected
    assert _occupancy(sg) == Counter(env._original_container_slots)


@pytest.mark.parametrize("seed", range(8))
def test_permute_object_containers_keeps_the_number_of_held_objects(seed):
    sg, idx = _multi_object_graph()
    c = idx.container
    slots = get_object_container_slots(sg)  # countertop x2, shelvingunit
    _move_object(sg, idx.obj["spoon"], idx.robot)  # robot holds the spoon
    sg.nodes[idx.obj["spoon"]]["position"] = sg.nodes[idx.robot]["position"]

    moves = permute_object_containers(sg, slots, random.Random(seed))

    held = [i for i in sg.object_indices if sg.get_parent_node_idx(i) == idx.robot]
    assert len(held) == 1
    in_containers = _occupancy(sg)
    in_containers.pop(idx.robot)
    assert sum(in_containers.values()) == 2
    assert all(in_containers[k] <= Counter(slots)[k] for k in in_containers)  # a slot was traded for the hand
    for obj_idx in sg.object_indices:
        parent = sg.get_parent_node_idx(obj_idx)
        assert sg.nodes[obj_idx]["position"] == sg.nodes[parent]["position"]
    assert moves  # always moves something


def test_permute_object_containers_can_change_which_object_is_held():
    held_across_seeds = set()
    for seed in range(20):
        sg, idx = _multi_object_graph()
        slots = get_object_container_slots(sg)
        _move_object(sg, idx.obj["spoon"], idx.robot)
        permute_object_containers(sg, slots, random.Random(seed))
        held_across_seeds |= {i for i in sg.object_indices if sg.get_parent_node_idx(i) == idx.robot}
    assert len(held_across_seeds) > 1


@pytest.mark.parametrize("seed", range(6))
def test_randomize_object_locations_keeps_hands_and_fluents_consistent_while_holding(seed):
    sg, idx = _multi_object_graph()
    slots = get_object_container_slots(sg)

    def name(i: int) -> str:
        return f"{sg.get_node_name_by_idx(i)}_{i}"

    def by_location() -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for obj_idx in sg.object_indices:
            parent = sg.get_parent_node_idx(obj_idx)
            if parent != idx.robot:
                result.setdefault(name(parent), set()).add(name(obj_idx))
        return result

    spoon, fork, cup = idx.obj["spoon"], idx.obj["fork"], idx.obj["cup"]
    at_start = {F(f"at {name(cup)} {name(idx.container['shelvingunit'])}")}
    env = _bare_kitchen_env(sg, fluents={F("free robot0")})
    env._objects_by_type["robot"] = {"robot0"}
    env.scene._object_locations = by_location()
    env._objects_at_locations = by_location()
    env._original_container_slots = slots
    # pick both onto the robot, as actions would: edges, and the fluents
    for held_idx in (spoon, fork):
        _move_object(sg, held_idx, idx.robot)
    env.fluents.update(
        at_start | {
            F(f"holding-in-left robot0 {name(spoon)}"), F(f"holding-in-right robot0 {name(fork)}"),
            F("left-hand-full robot0"), F("right-hand-full robot0"),
        }
    )

    env.randomize_object_locations(random.Random(seed))

    held = [i for i in sg.object_indices if sg.get_parent_node_idx(i) == idx.robot]
    assert len(held) == 2
    a, b = held
    assert {f for f in env.fluents if f.name.startswith("holding-in")} in (
        {F(f"holding-in-left robot0 {name(a)}"), F(f"holding-in-right robot0 {name(b)}")},
        {F(f"holding-in-left robot0 {name(b)}"), F(f"holding-in-right robot0 {name(a)}")},
    )
    assert {f for f in env.fluents if f.name == "at"} == {
        F(f"at {name(o)} {loc}")
        for loc, objs in by_location().items() for o in sg.object_indices if name(o) in objs
    }
    assert {F("free robot0"), F("left-hand-full robot0"), F("right-hand-full robot0")} <= env.fluents
    assert env.scene.object_locations == by_location()
    assert env.objects_at_locations == by_location()
