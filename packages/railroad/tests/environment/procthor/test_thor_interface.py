"""Tests for ThorInterface."""

import pytest
import numpy as np

from railroad.environment.procthor.resources import REMAP_DIR_ENV_VAR
from railroad.environment.procthor.thor_interface import ThorInterface


@pytest.fixture
def thor_interface():
    """Create ThorInterface for testing."""
    return ThorInterface(seed=0, resolution=0.05)


@pytest.mark.timeout(30)
def test_thor_interface_initialization(thor_interface):
    """Test ThorInterface initializes correctly."""
    assert len(thor_interface.scene_graph.nodes) > 0
    assert len(thor_interface.scene_graph.edges) > 0
    assert thor_interface.occupancy_grid.size > 0

    # Check grid values
    unique_vals = np.unique(thor_interface.occupancy_grid)
    assert 1 in unique_vals and 0 in unique_vals


@pytest.mark.timeout(30)
def test_thor_interface_robot_pose(thor_interface):
    """Test robot pose is extracted."""
    pose = thor_interface.robot_pose
    assert isinstance(pose, tuple)
    assert len(pose) == 2


@pytest.mark.timeout(30)
def test_thor_interface_known_costs(thor_interface):
    """Test known costs are computed."""
    assert 'initial_robot_pose' in thor_interface.known_cost
    # Check symmetric
    for id1, costs in thor_interface.known_cost.items():
        for id2, cost in costs.items():
            if id1 != id2:
                assert thor_interface.known_cost[id2][id1] == cost


@pytest.mark.timeout(30)
def test_thor_interface_target_objects(thor_interface):
    """Test target object info extraction."""
    info = thor_interface.get_target_objs_info(num_objects=1)
    assert 'name' in info
    assert 'idxs' in info
    assert 'type' in info
    assert 'container_idxs' in info


def _bare_thor_interface() -> ThorInterface:
    """Construct a ThorInterface without running __init__, so
    _deduplicate_objects can be tested on synthetic containers without
    loading real scene data or spinning up a Controller."""
    return ThorInterface.__new__(ThorInterface)


def test_remap_lookup_is_disabled_when_env_var_unset(monkeypatch):
    """With no opt-in, no remap is ever consulted - so a leftover file from an
    earlier run can't silently change what an unrelated caller loads."""
    monkeypatch.delenv(REMAP_DIR_ENV_VAR, raising=False)
    ti = _bare_thor_interface()
    ti.seed = 5

    assert ti._remapped_scene_path() is None
    assert ti._check_for_remapped_scene() is False


def test_remap_lookup_only_reads_the_opted_in_directory(monkeypatch, tmp_path):
    (tmp_path / "scene_5.json").write_text('{"marker": "remapped"}')
    monkeypatch.setenv(REMAP_DIR_ENV_VAR, str(tmp_path))

    ti = _bare_thor_interface()
    ti.seed = 5
    assert ti._check_for_remapped_scene() is True
    assert ti._load_remapped_scene() == {"marker": "remapped"}

    # keyed by seed within the directory: a different seed has no remap
    other = _bare_thor_interface()
    other.seed = 6
    assert other._check_for_remapped_scene() is False


def test_remap_lookup_ignores_a_different_run_directory(monkeypatch, tmp_path):
    """Two runs use two directories; a run only ever sees its own remaps."""
    run_a, run_b = tmp_path / "run_a", tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "scene_5.json").write_text('{"run": "a"}')

    monkeypatch.setenv(REMAP_DIR_ENV_VAR, str(run_b))
    ti = _bare_thor_interface()
    ti.seed = 5
    assert ti._check_for_remapped_scene() is False


def test_deduplicate_objects_keeps_every_container_and_first_object_per_category():
    """Duplicate containers stay; an object category already seen in an
    earlier container is dropped from later ones."""
    ti = _bare_thor_interface()
    ti.containers = [
        {"id": "Chair|1", "children": [{"id": "Knife|1"}]},
        {"id": "Chair|2", "children": [{"id": "Knife|2"}, {"id": "Pan|1"}]},
        {"id": "Painting|1"},
    ]

    ti._deduplicate_objects()

    assert [c["id"] for c in ti.containers] == ["Chair|1", "Chair|2", "Painting|1"]
    assert [ch["id"] for ch in ti.containers[0]["children"]] == ["Knife|1"]
    assert [ch["id"] for ch in ti.containers[1]["children"]] == ["Pan|1"]
    assert "children" not in ti.containers[2]


def test_deduplicate_objects_leaves_the_source_containers_untouched():
    """The scene a cache is generated from must keep every object."""
    ti = _bare_thor_interface()
    source = [
        {"id": "Chair|1", "children": [{"id": "Knife|1"}]},
        {"id": "Chair|2", "children": [{"id": "Knife|2"}]},
    ]
    ti.containers = list(source)

    ti._deduplicate_objects()

    assert [ch["id"] for ch in source[1]["children"]] == ["Knife|2"]
    assert ti.containers[1]["children"] == []
