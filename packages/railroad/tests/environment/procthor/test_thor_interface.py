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
