"""
Tests for local (patched) changes to the vendored `railroad` package.
See patches/railroad/ at the repo root for the corresponding diffs.
"""
import types

import numpy as np
import pytest
from railroad.environment.procthor import thor_interface as thor_interface_module
from railroad.environment.procthor.scenegraph import SceneGraph
from railroad.environment.procthor.thor_interface import ThorInterface

# New 5-element one-hot node type schema (was 4 elements, no ROBOT):
# apartment=0, robot=1, room=2, container=3, object=4
APARTMENT, ROBOT, ROOM, CONTAINER, OBJECT = range(5)


def _node_type(type_idx: int) -> list[int]:
    type_vec = [0, 0, 0, 0, 0]
    type_vec[type_idx] = 1
    return type_vec


def _node(node_id: str, type_idx: int, position=(0, 0)) -> dict:
    return {
        "id": node_id,
        "type": _node_type(type_idx),
        "position": list(position),
        "name": node_id,
    }


def test_scenegraph_type_indices_with_five_type_schema():
    sg = SceneGraph()
    idx_apt = sg.add_node(_node("apt", APARTMENT))
    idx_robot = sg.add_node(_node("robot0", ROBOT))
    idx_room = sg.add_node(_node("bedroom0", ROOM))
    idx_cnt = sg.add_node(_node("bed0", CONTAINER))
    idx_obj = sg.add_node(_node("pillow0", OBJECT))

    assert set(sg.robot_indices) == {idx_robot}
    assert set(sg.room_indices) == {idx_room}
    assert set(sg.container_indices) == {idx_cnt}
    assert set(sg.object_indices) == {idx_obj}
    # apartment has no dedicated *_indices property, sanity check it's
    # still excluded from all four
    assert idx_apt not in (sg.robot_indices + sg.room_indices + sg.container_indices + sg.object_indices)


def test_get_parent_node_idx_robot_parent_is_apartment():
    sg = SceneGraph()
    idx_apt = sg.add_node(_node("apt", APARTMENT))
    idx_robot = sg.add_node(_node("robot0", ROBOT))
    sg.add_edge(idx_apt, idx_robot)

    assert sg.get_parent_node_idx(idx_robot) == idx_apt


def test_get_parent_node_idx_room_parent_is_apartment():
    sg = SceneGraph()
    idx_apt = sg.add_node(_node("apt", APARTMENT))
    idx_room = sg.add_node(_node("room0", ROOM))
    sg.add_edge(idx_apt, idx_room)

    assert sg.get_parent_node_idx(idx_room) == idx_apt


def test_get_parent_node_idx_container_parent_is_room():
    sg = SceneGraph()
    idx_room = sg.add_node(_node("room0", ROOM))
    idx_cnt = sg.add_node(_node("cnt0", CONTAINER))
    sg.add_edge(idx_room, idx_cnt)

    assert sg.get_parent_node_idx(idx_cnt) == idx_room


def test_get_parent_node_idx_object_parent_is_container():
    sg = SceneGraph()
    idx_cnt = sg.add_node(_node("cnt0", CONTAINER))
    idx_obj = sg.add_node(_node("obj0", OBJECT))
    sg.add_edge(idx_cnt, idx_obj)

    assert sg.get_parent_node_idx(idx_obj) == idx_cnt


def test_get_parent_node_idx_object_held_by_robot_falls_back_to_robot():
    """
    New behavior: an object with no adjacent container (e.g. picked up)
    resolves its parent to an adjacent robot node instead of None.
    """
    sg = SceneGraph()
    idx_robot = sg.add_node(_node("robot0", ROBOT))
    idx_obj = sg.add_node(_node("obj0", OBJECT))
    sg.add_edge(idx_robot, idx_obj)

    assert sg.get_parent_node_idx(idx_obj) == idx_robot


def test_get_parent_node_idx_object_prefers_container_over_robot():
    """
    The robot fallback should only trigger when no container parent was
    found -- if an object is (unusually) adjacent to both, the container
    relationship wins.
    """
    sg = SceneGraph()
    idx_robot = sg.add_node(_node("robot0", ROBOT))
    idx_cnt = sg.add_node(_node("cnt0", CONTAINER))
    idx_obj = sg.add_node(_node("obj0", OBJECT))
    sg.add_edge(idx_cnt, idx_obj)
    sg.add_edge(idx_robot, idx_obj)

    assert sg.get_parent_node_idx(idx_obj) == idx_cnt


def test_get_parent_node_idx_object_with_no_container_or_robot_is_none():
    sg = SceneGraph()
    idx_obj = sg.add_node(_node("obj0", OBJECT))

    assert sg.get_parent_node_idx(idx_obj) is None


def test_get_parent_node_idx_apartment_has_no_parent():
    sg = SceneGraph()
    idx_apt = sg.add_node(_node("apt", APARTMENT))

    assert sg.get_parent_node_idx(idx_apt) is None


def test_get_object_free_graph_preserves_robot_nodes():
    sg = SceneGraph()
    idx_robot = sg.add_node(_node("robot0", ROBOT))
    idx_obj = sg.add_node(_node("obj0", OBJECT))
    sg.add_edge(idx_robot, idx_obj)

    sg_free = sg.get_object_free_graph()

    assert set(sg_free.robot_indices) == {idx_robot}
    assert sg_free.object_indices == []


def test_object_indices_raises_for_legacy_four_element_type_vector():
    """
    Documents a compatibility break: any caller still constructing nodes
    with the old 4-element type vector (no ROBOT slot) will raise
    IndexError on object_indices, since index 4 is now out of range.
    Anything still building nodes with 4-element type lists needs
    updating to the 5-element schema.
    """
    sg = SceneGraph()
    sg.add_node({"id": "obj0", "type": [0, 0, 0, 1], "position": [0, 0], "name": "obj0"})

    with pytest.raises(IndexError):
        sg.object_indices


# --- ThorInterface: object-location randomization + robot scene-graph node ---

def _bare_thor_interface() -> ThorInterface:
    """Construct a ThorInterface without running __init__ (no AI2-THOR, no dataset)."""
    return object.__new__(ThorInterface)


def _pickupable_object(object_id, asset_id, parent_receptacle, position=None, rotation=None):
    return {
        "objectId": object_id,
        "assetId": asset_id,
        "pickupable": True,
        "parentReceptacles": [parent_receptacle],
        "position": position or {"x": 0, "y": 0, "z": 0},
        "rotation": rotation or {"x": 0, "y": 0, "z": 0},
    }


def test_update_object_locations_groups_pickupable_objects_by_container():
    ti = _bare_thor_interface()
    ti.scene = {"objects": [{"id": "cnt0", "children": []}]}
    event = types.SimpleNamespace(metadata={"objects": [
        _pickupable_object("spoon_1", "Spoon_1", "cnt0_receptacle"),
        _pickupable_object("cup_1", "Cup_1", "cnt0_receptacle"),
    ]})

    ti._update_object_locations(event)

    children = ti.scene["objects"][0]["children"]
    assert {c["id"] for c in children} == {"spoon_1", "cup_1"}
    assert all(c["kinematic"] is False for c in children)


def test_update_object_locations_ignores_non_pickupable_objects():
    ti = _bare_thor_interface()
    ti.scene = {"objects": [{"id": "cnt0"}]}
    non_pickupable = _pickupable_object("light_1", "Light_1", "cnt0_receptacle")
    non_pickupable["pickupable"] = False
    event = types.SimpleNamespace(metadata={"objects": [non_pickupable]})

    ti._update_object_locations(event)

    assert "children" not in ti.scene["objects"][0]


def test_update_object_locations_clears_stale_children_for_untouched_containers():
    """
    Every container's `children` is deleted up front and only rebuilt if
    the sim reports pickupable objects there this call -- a container
    that previously had children but got nothing placed on it this time
    ends up with no `children` key at all, not its old (stale) list.
    """
    ti = _bare_thor_interface()
    ti.scene = {"objects": [
        {"id": "cnt0", "children": [{"id": "stale_obj", "position": {}, "rotation": {}}]},
        {"id": "cnt1", "children": []},
    ]}
    event = types.SimpleNamespace(metadata={"objects": [
        _pickupable_object("spoon_1", "Spoon_1", "cnt1_receptacle"),
    ]})

    ti._update_object_locations(event)

    assert "children" not in ti.scene["objects"][0]
    assert {c["id"] for c in ti.scene["objects"][1]["children"]} == {"spoon_1"}


def test_randomized_scene_check_save_load_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(thor_interface_module, "get_procthor_10k_dir", lambda: tmp_path)

    ti = _bare_thor_interface()
    ti.seed = 3
    ti.object_seed = 7
    ti.scene = {"objects": [{"id": "cnt0"}]}

    assert ti._check_for_randomized_scene() is False

    ti._save_randomized_scene(path=str(tmp_path / "randomized_scenes"))

    assert ti._check_for_randomized_scene() is True
    assert ti._load_randomized_objects_scene() == ti.scene


def test_save_randomized_scene_default_path_can_diverge_from_check(monkeypatch, tmp_path):
    """
    _save_randomized_scene's default `path` is a hardcoded relative
    string ('./resources/procthor-10k/randomized_scenes'), independent
    of get_procthor_10k_dir(), which _check_for_randomized_scene and
    _load_randomized_objects_scene both use. If get_procthor_10k_dir()
    ever resolves somewhere other than that hardcoded relative path
    (e.g. a custom resources base dir), a save with the default path and
    a subsequent check silently miss each other, and the cache never
    hits -- every run re-randomizes and re-saves.
    """
    configured_elsewhere = tmp_path / "configured_elsewhere"
    monkeypatch.setattr(thor_interface_module, "get_procthor_10k_dir", lambda: configured_elsewhere)
    monkeypatch.chdir(tmp_path)  # hardcoded relative default resolves here, not under configured_elsewhere

    ti = _bare_thor_interface()
    ti.seed = 3
    ti.object_seed = 7
    ti.scene = {"objects": []}

    ti._save_randomized_scene()  # uses its own hardcoded default path

    assert ti._check_for_randomized_scene() is False

def _fake_render_top_down_from_controller(orthographic: bool):
    return (np.zeros((2, 2), dtype=np.uint8), (5, 7))

def test_save_and_get_cache_filename_includes_object_seed_when_set(monkeypatch, tmp_path):
    ti = _bare_thor_interface()
    ti.seed = 3
    ti.object_seed = 7
    monkeypatch.setattr(ti, "_get_reachable_positions_from_controller", lambda: [])
    monkeypatch.setattr(ti, "_render_top_down_from_controller", _fake_render_top_down_from_controller)

    ti._save_and_get_cache(path=str(tmp_path))

    assert (tmp_path / "scene_3_7.pkl").exists()
    assert not (tmp_path / "scene_3.pkl").exists()


def test_save_and_get_cache_filename_omits_object_seed_when_none(monkeypatch, tmp_path):
    ti = _bare_thor_interface()
    ti.seed = 3
    ti.object_seed = None
    monkeypatch.setattr(ti, "_get_reachable_positions_from_controller", lambda: [])
    monkeypatch.setattr(ti, "_render_top_down_from_controller", _fake_render_top_down_from_controller)

    ti._save_and_get_cache(path=str(tmp_path))

    assert (tmp_path / "scene_3.pkl").exists()
    assert not (tmp_path / "scene_3_7.pkl").exists()


def test_load_cache_returns_none_when_file_missing_for_object_seed(tmp_path):
    ti = _bare_thor_interface()
    ti.seed = 3
    ti.object_seed = 7

    assert ti._load_cache(path=str(tmp_path)) is None


def test_get_scene_graph_adds_robot_node_connected_to_apartment(monkeypatch):
    monkeypatch.setattr(thor_interface_module.utils, "get_edges_for_connected_graph", lambda *a, **k: [])

    ti = _bare_thor_interface()
    ti.rooms = [{"id": "Room|1", "roomType": "Kitchen", "position": (0, 0)}]
    ti.containers = [{
        "id": "CounterTop|1|2",
        "position": (1, 1),
        "children": [{"id": "Spoon|1|2|3", "position": (1, 1)}],
    }]
    ti.robot_pose = (5, 5)
    ti.scene = {"doors": []}
    ti.occupancy_grid = np.ones((2, 2))

    graph = ti._get_scene_graph()

    apt_idx = next(idx for idx, n in graph.nodes.items() if n["id"] == "Apartment|0")
    assert len(graph.robot_indices) == 1
    robot_idx = graph.robot_indices[0]
    assert graph.nodes[robot_idx]["position"] == (5, 5)
    assert graph.get_parent_node_idx(robot_idx) == apt_idx


def test_get_scene_graph_node_types_match_five_slot_schema_and_hierarchy(monkeypatch):
    monkeypatch.setattr(thor_interface_module.utils, "get_edges_for_connected_graph", lambda *a, **k: [])

    ti = _bare_thor_interface()
    ti.rooms = [{"id": "Room|1", "roomType": "Kitchen", "position": (0, 0)}]
    ti.containers = [{
        "id": "CounterTop|1|2",
        "position": (1, 1),
        "children": [{"id": "Spoon|1|2|3", "position": (1, 1)}],
    }]
    ti.robot_pose = (5, 5)
    ti.scene = {"doors": []}
    ti.occupancy_grid = np.ones((2, 2))

    graph = ti._get_scene_graph()

    apt_idx = next(idx for idx, n in graph.nodes.items() if n["id"] == "Apartment|0")
    robot_idx = graph.robot_indices[0]
    room_idx = graph.room_indices[0]
    cnt_idx = graph.container_indices[0]
    obj_idx = graph.object_indices[0]

    assert graph.nodes[apt_idx]["type"] == [1, 0, 0, 0, 0]
    assert graph.nodes[robot_idx]["type"] == [0, 1, 0, 0, 0]
    assert graph.nodes[room_idx]["type"] == [0, 0, 1, 0, 0]
    assert graph.nodes[cnt_idx]["type"] == [0, 0, 0, 1, 0]
    assert graph.nodes[obj_idx]["type"] == [0, 0, 0, 0, 1]

    assert graph.get_parent_node_idx(room_idx) == apt_idx
    assert graph.get_parent_node_idx(cnt_idx) == room_idx
    assert graph.get_parent_node_idx(obj_idx) == cnt_idx
