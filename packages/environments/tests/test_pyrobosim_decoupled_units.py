import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import time as time_module

from railroad._bindings import State, Fluent
from railroad.core import Fluent as F
from railroad.environment import SkillStatus
from railroad.environment.pyrobosim import (
    DecoupledPyRoboSimEnvironment,
    PyRoboSimScene,
    get_default_pyrobosim_world_file_path
)

@pytest.fixture
def mock_env():
    world_file = get_default_pyrobosim_world_file_path()
    scene = MagicMock(spec=PyRoboSimScene)
    scene.world_file = world_file
    scene.locations = {"table0": (1, 1), "my_desk": (2, 2), "robot1_loc": (0, 0)}
    scene.robots = [MagicMock(name="robot1")]
    scene.robots[0].name = "robot1"
    scene.robots[0].location = MagicMock()
    scene.robots[0].location.name = "kitchen"
    scene.world = MagicMock()
    scene.world.rooms = []

    initial_fluents = {F("at", "robot1", "robot1_loc"), F("free", "robot1")}
    initial_state = State(0.0, initial_fluents)

    # Patch PyRoboSimBridge so it doesn't start a real process
    with patch("railroad.environment.pyrobosim.pyrobosim.PyRoboSimBridge") as MockBridge:
        mock_bridge = MockBridge.return_value
        mock_bridge.get_latest_status.return_value = None

        env = DecoupledPyRoboSimEnvironment(
            scene=scene,
            state=initial_state,
            objects_by_type={"robot": {"robot1"}, "location": {"table0", "my_desk", "robot1_loc"}, "object": {"banana0"}},
            operators=[],
            show_plot=False
        )
        # Manually set loc_to_room for consistency tests
        env._loc_to_room = {"table0": "kitchen", "my_desk": "bedroom", "robot1_loc": "kitchen"}
        return env

def test_sync_state_robot_proxy_fluent(mock_env):
    """Test that _sync_state respects proxy fluents like robot1_loc."""
    # 1. Simulator says robot is in room 'kitchen'
    status = {
        "time": 1.0,
        "robots": {
            "robot1": {
                "location": "kitchen",
                "location_type": "room",
                "busy": False,
                "holding": None,
                "current_id": None,
                "last_completed_id": -1
            }
        },
        "objects": {}
    }

    # Use mock_bridge to provide status
    mock_env._bridge.get_latest_status.return_value = status
    mock_env._on_act_loop_iteration(0.1)

    # Should STILL have 'robot1_loc' because it's a specific location known to scene
    # and it is in 'kitchen' room.
    assert F("at", "robot1", "robot1_loc") in mock_env.fluents
    # Should NOT have 'kitchen' room fluent because we already have a specific location
    assert F("at", "robot1", "kitchen") not in mock_env.fluents

    # 2. Simulator says robot is at a specific location 'table0'
    status["robots"]["robot1"]["location"] = "table0"
    status["robots"]["robot1"]["location_type"] = "location"

    mock_env._bridge.get_latest_status.return_value = status
    mock_env._on_act_loop_iteration(0.1)

    # Should have 'table0'
    assert F("at", "robot1", "table0") in mock_env.fluents
    # Should NOT have 'robot1_loc' anymore
    assert F("at", "robot1", "robot1_loc") not in mock_env.fluents

def test_sync_state_holding_objects(mock_env):
    """Test that _sync_state correctly handles object containment."""
    # Add object at table0 initially
    mock_env.fluents.add(F("at", "banana0", "table0"))

    status = {
        "time": 1.0,
        "robots": {
            "robot1": {
                "location": "table0",
                "location_type": "location",
                "busy": False,
                "holding": "banana0",
                "current_id": None,
                "last_completed_id": 0
            }
        },
        "objects": {
            "banana0": {
                "location": "robot1", # In pyrobosim obj.parent is the robot
                "location_type": "robot"
            }
        }
    }

    mock_env._bridge.get_latest_status.return_value = status
    mock_env._on_act_loop_iteration(0.1)

    assert F("holding", "robot1", "banana0") in mock_env.fluents
    assert F("hand-full", "robot1") in mock_env.fluents
    # Object should not have an 'at' fluent if it's held
    assert not any(f.name == "at" and f.args[0] == "banana0" for f in mock_env.fluents)

def test_busy_handshake(mock_env):
    """Test that local busy tracking prevents stale status from marking robot free."""
    # Dispatch a move command
    mock_env.execute_skill("robot1", "move", "robot1", "robot1_loc", "table0")

    # Should be busy locally (free removed)
    assert F("free", "robot1") not in mock_env.fluents
    assert mock_env._last_dispatched_id["robot1"] == 0

    # Receive a status that is technically "new" but simulator hasn't started move yet
    status = {
        "time": time_module.time() + 0.1,
        "robots": {
            "robot1": {
                "location": "kitchen",
                "location_type": "room",
                "busy": False, # Still False in simulator
                "holding": None,
                "current_id": None,
                "last_completed_id": -1 # Hasn't seen cmd 0 yet
            }
        },
        "objects": {}
    }

    mock_env._bridge.get_latest_status.return_value = status
    mock_env._on_act_loop_iteration(0.1)

    # Should STILL be busy (free robot1 should not be added back)
    assert F("free", "robot1") not in mock_env.fluents

    # Now simulator confirms it's busy with cmd 0
    status["robots"]["robot1"]["busy"] = True
    status["robots"]["robot1"]["current_id"] = 0

    mock_env._bridge.get_latest_status.return_value = status
    mock_env._on_act_loop_iteration(0.1)

    # Still busy in fluents
    assert F("free", "robot1") not in mock_env.fluents

def test_sync_state_room_transition(mock_env):
    """Test that _sync_state clears specific location when robot enters a different room."""
    # Setup: robot1 at table0 (in kitchen), and env knows table0 is in kitchen
    mock_env.fluents.add(F("at", "robot1", "table0"))

    # 1. Simulator says robot is now in 'bedroom' room (idle)
    status = {
        "time": 1.0,
        "robots": {
            "robot1": {
                "location": "bedroom",
                "location_type": "room",
                "busy": False,
                "holding": None,
                "current_id": None,
                "last_completed_id": -1
            }
        },
        "objects": {}
    }

    mock_env._bridge.get_latest_status.return_value = status
    mock_env._on_act_loop_iteration(0.1)

    # Should DISCARD 'table0' because it's in 'kitchen', not 'bedroom'
    assert F("at", "robot1", "table0") not in mock_env.fluents
    # Should ADD 'bedroom' room fluent
    assert F("at", "robot1", "bedroom") in mock_env.fluents

def test_search_skill_handshake(mock_env):
    """Test that search skill correctly completes its handshake."""
    # Dispatch a search command
    mock_env.execute_skill("robot1", "search", "robot1", "table0", "banana0")

    # Should be busy locally
    assert F("free", "robot1") not in mock_env.fluents
    assert mock_env._last_dispatched_id["robot1"] == 0

    # Simulator confirms it's busy with search
    status = {
        "time": 1.0,
        "robots": {
            "robot1": {
                "location": "table0",
                "location_type": "location",
                "busy": True,
                "holding": None,
                "current_id": 0,
                "last_completed_id": -1
            }
        },
        "objects": {}
    }
    mock_env._bridge.get_latest_status.return_value = status
    mock_env._on_act_loop_iteration(0.1)
    assert mock_env.get_executed_skill_status("robot1", "search") == SkillStatus.RUNNING

    # Now simulator confirms search is complete
    status["robots"]["robot1"]["busy"] = False
    status["robots"]["robot1"]["current_id"] = None
    status["robots"]["robot1"]["last_completed_id"] = 0

    mock_env._bridge.get_latest_status.return_value = status
    mock_env._on_act_loop_iteration(0.1)

    assert mock_env.get_executed_skill_status("robot1", "search") == SkillStatus.DONE
    assert F("free", "robot1") in mock_env.fluents
