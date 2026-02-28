import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import time as time_module

from railroad._bindings import State, Fluent
from railroad.core import Fluent as F
from railroad.environment import SkillStatus
from railroad.environment.pyrobosim import (
    PyRoboSimEnvironment,
    PyRoboSimScene,
    get_default_pyrobosim_world_file_path
)

@pytest.fixture
def mock_env():
    world_file = get_default_pyrobosim_world_file_path()

    # Mock scene with a mock client
    scene = MagicMock(spec=PyRoboSimScene)
    scene.world_file = world_file
    scene.locations = {"table0": (1, 1), "my_desk": (2, 2), "robot1_loc": (0, 0)}

    mock_client = MagicMock()
    mock_client.get_robot_status.return_value = {"current_task_id": -1, "last_completed_id": -1}
    scene.client = mock_client

    initial_fluents = {F("at", "robot1", "robot1_loc"), F("free", "robot1")}
    initial_state = State(0.0, initial_fluents)

    env = PyRoboSimEnvironment(
        scene=scene,
        state=initial_state,
        objects_by_type={"robot": {"robot1"}, "location": {"table0", "my_desk", "robot1_loc"}, "object": {"banana0"}},
        operators=[],
    )
    return env

def test_execute_skill_dispatches_with_id(mock_env):
    """Test that execute_skill calls client with incrementing IDs."""
    mock_env.execute_skill("robot1", "move", "robot1", "robot1_loc", "table0")

    assert mock_env._last_dispatched_ids["robot1"] == 0
    mock_env._client.call_service.assert_called_with("execute", "robot1", "move", ("robot1", "robot1_loc", "table0"), 0)

    mock_env.execute_skill("robot1", "pick", "robot1", "table0", "banana0")
    assert mock_env._last_dispatched_ids["robot1"] == 1
    mock_env._client.call_service.assert_called_with("execute", "robot1", "pick", ("robot1", "table0", "banana0"), 1)

def test_get_executed_skill_status_handshake(mock_env):
    """Test the robust handshake logic in get_executed_skill_status."""
    # 1. Dispatch ID 0
    mock_env.execute_skill("robot1", "move", "robot1", "robot1_loc", "table0")

    # Case A: Client has no status yet
    mock_env._client.get_robot_status.return_value = {}
    assert mock_env.get_executed_skill_status("robot1", "move") == SkillStatus.RUNNING

    # Case B: Server picked up the task but hasn't finished (current_task_id == 0)
    mock_env._client.get_robot_status.return_value = {"current_task_id": 0, "last_completed_id": -1}
    assert mock_env.get_executed_skill_status("robot1", "move") == SkillStatus.RUNNING

    # Case C: Server finished the task (last_completed_id == 0)
    mock_env._client.get_robot_status.return_value = {"current_task_id": -1, "last_completed_id": 0}
    assert mock_env.get_executed_skill_status("robot1", "move") == SkillStatus.DONE

    # Case D: Server finished a LATER task (last_completed_id > 0)
    mock_env._client.get_robot_status.return_value = {"current_task_id": -1, "last_completed_id": 5}
    assert mock_env.get_executed_skill_status("robot1", "move") == SkillStatus.DONE

def test_stop_robot_dispatches(mock_env):
    """Test that stop_robot calls the appropriate service."""
    mock_env.stop_robot("robot1")
    mock_env._client.call_service.assert_called_with("stop", "robot1")
