"""Tests for ThorInterface."""

import pytest
import numpy as np

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
    _deduplicate_containers can be tested on synthetic containers without
    loading real scene data or spinning up a Controller."""
    return ThorInterface.__new__(ThorInterface)


def test_deduplicate_containers_keeps_most_populated_and_dedupes_children():
    """Duplicate containers of the same type should collapse to the instance
    with the most children; child objects of a type already kept by an
    earlier-processed container should be dropped, not just deduped within
    their own container."""
    ti = _bare_thor_interface()
    ti.containers = [
        {"id": "CounterTop|1", "children": [{"id": "Knife|1"}]},
        {"id": "Fridge|1", "children": [{"id": "Knife|2"}, {"id": "Pan|1"}]},
        {"id": "CounterTop|2", "children": [
            {"id": "Knife|3"}, {"id": "Pan|2"}, {"id": "Egg|1"},
        ]},
    ]

    ti._deduplicate_containers()

    container_types = [c["id"].split("|")[0].lower() for c in ti.containers]
    assert sorted(container_types) == ["countertop", "fridge"]

    countertop = next(c for c in ti.containers if c["id"] == "CounterTop|2")
    assert [child["id"] for child in countertop["children"]] == ["Knife|3", "Pan|2", "Egg|1"]

    # Both of fridge's children ("knife", "pan") were already claimed by the
    # countertop, which is processed first (its type appeared first in the
    # original list) -- so fridge is left with no children of its own.
    fridge = next(c for c in ti.containers if c["id"] == "Fridge|1")
    assert fridge["children"] == []


def test_deduplicate_containers_tie_keeps_first_seen():
    """Equal child counts should keep the first-seen instance of a type."""
    ti = _bare_thor_interface()
    ti.containers = [
        {"id": "Fridge|1", "children": [{"id": "Knife|1"}]},
        {"id": "Fridge|2", "children": [{"id": "Pan|1"}]},
    ]

    ti._deduplicate_containers()

    assert len(ti.containers) == 1
    assert ti.containers[0]["id"] == "Fridge|1"


def test_deduplicate_containers_leaves_childless_containers_untouched():
    """Containers with no 'children' key (e.g. non-receptacle objects)
    should survive dedup without gaining a spurious empty children list."""
    ti = _bare_thor_interface()
    ti.containers = [
        {"id": "Painting|1"},
        {"id": "Fridge|1", "children": [{"id": "Knife|1"}]},
    ]

    ti._deduplicate_containers()

    painting = next(c for c in ti.containers if c["id"] == "Painting|1")
    assert "children" not in painting
