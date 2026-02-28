import threading
import time as time_module
import pytest
import numpy as np
from typing import Dict, Set, Any, List, Optional, Tuple, Callable

from railroad._bindings import State, Fluent, Action
from railroad.core import Operator, Fluent as F
from railroad.environment import (
    PhysicalEnvironment,
    PhysicalScene,
    SkillStatus,
    LocationRegistry
)
from railroad.operators import (
    construct_move_operator_blocking,
    construct_pick_operator_blocking,
    construct_place_operator_blocking
)


def _run_async(func: Callable) -> Callable:
    """Decorator to run a function in a daemon thread, matching pyrobosim pattern."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(
            target=func,
            args=args,
            kwargs=kwargs,
            daemon=True
        )
        thread.start()
        return thread
    return wrapper


class SamplePhysicalScene(PhysicalScene):
    """A simple physical scene for testing with larger distances."""
    def __init__(self):
        self._locations = {
            "start": np.array([0.0, 0.0]),
            "shelf1": np.array([3.0, 0.0]),
            "shelf2": np.array([0.0, 4.0]),
            "bin": np.array([-3.0, -4.0]), # Distance from start is 5.0
            "table": np.array([2.0, 2.0])
        }
        self._objects = {"cube1", "cube2", "ball1"}
        self._object_locations = {
            "shelf1": {"cube1"},
            "shelf2": {"cube2"},
            "bin": {"ball1"}
        }

    @property
    def locations(self) -> Dict[str, Any]:
        return self._locations

    @property
    def objects(self) -> Set[str]:
        return self._objects

    @property
    def object_locations(self) -> Dict[str, Set[str]]:
        return self._object_locations

    def get_move_cost_fn(self) -> Callable[[str, str, str], float]:
        def move_cost(robot, loc_from, loc_to):
            p1 = self._locations.get(loc_from)
            p2 = self._locations.get(loc_to)
            if p1 is None or p2 is None:
                return float("inf")
            return float(np.linalg.norm(p1 - p2))
        return move_cost


class SamplePhysicalEnvironment(PhysicalEnvironment):
    """A physical environment with async actions implemented directly in the class."""
    def __init__(self, scene, state, objects_by_type, operators, location_registry=None):
        super().__init__(
            scene=scene,
            state=state,
            objects_by_type=objects_by_type,
            operators=operators,
            location_registry=location_registry
        )
        self._is_busy: Dict[str, bool] = {name: False for name in objects_by_type.get("robot", set())}
        self._stop_requested: Dict[str, bool] = {name: False for name in objects_by_type.get("robot", set())}
        self.execution_log: List[str] = []

    def execute_skill(self, robot_name: str, skill_name: str, *args: Any, **kwargs: Any) -> None:
        """Execute a skill, dispatching to async handlers."""
        self.execution_log.append(f"execute {skill_name} on {robot_name} with {args}")

        if skill_name == "move":
            loc_from = args[0]
            loc_to = args[1]
            self._move(robot_name, loc_from, loc_to)
        elif skill_name == "pick":
            obj_name = args[1] if len(args) > 1 else args[0]
            self._pick(robot_name, obj_name)
        elif skill_name == "place":
            self._place(robot_name)

        # Small sleep to ensure the thread has a chance to start and set _is_busy=True
        time_module.sleep(0.05)

    def get_executed_skill_status(self, robot_name: str, skill_name: str) -> SkillStatus:
        if self._is_busy.get(robot_name, False):
            return SkillStatus.RUNNING
        return SkillStatus.DONE

    def stop_robot(self, robot_name: str) -> None:
        if self._is_busy.get(robot_name, False):
            self.execution_log.append(f"stop {robot_name}")
            self._stop_requested[robot_name] = True

    def add_object_at_location(self, obj: str, location: str) -> None:
        self.scene.object_locations.setdefault(location, set()).add(obj)

    def remove_object_from_location(self, obj: str, location: str) -> None:
        if location in self.scene.object_locations:
            self.scene.object_locations[location].discard(obj)

    # --- Async Handlers ---

    def _simulate_task(self, robot_name: str, duration: float):
        """Internal helper to simulate hardware execution in a thread."""
        self._is_busy[robot_name] = True
        self._stop_requested[robot_name] = False

        start_time = time_module.time()
        while time_module.time() - start_time < duration:
            if self._stop_requested[robot_name]:
                break
            time_module.sleep(0.01)

        self._is_busy[robot_name] = False
        self._stop_requested[robot_name] = False

    @_run_async
    def _move(self, robot_name: str, loc_from: str, loc_to: str):
        duration = self.scene.get_move_cost_fn()(robot_name, loc_from, loc_to)
        self._simulate_task(robot_name, duration)

    @_run_async
    def _pick(self, robot_name: str, object_name: str):
        self._simulate_task(robot_name, 0.3)

    @_run_async
    def _place(self, robot_name: str):
        self._simulate_task(robot_name, 0.3)


def test_physical_environment_basic_flow():
    """Test basic move-pick-move-place flow in PhysicalEnvironment with async actions."""
    scene = SamplePhysicalScene()

    move_op = construct_move_operator_blocking(scene.get_move_cost_fn())
    pick_op = construct_pick_operator_blocking(0.3)
    place_op = construct_place_operator_blocking(0.3)

    initial_fluents = {
        F("at", "robot1", "start"),
        F("free", "robot1"),
        F("at", "ball1", "bin")
    }
    initial_state = State(0.0, initial_fluents)

    objects_by_type = {
        "robot": {"robot1"},
        "location": set(scene.locations.keys()),
        "object": set(scene.objects)
    }

    env = SamplePhysicalEnvironment(
        scene=scene,
        state=initial_state,
        objects_by_type=objects_by_type,
        operators=[move_op, pick_op, place_op]
    )

    # 1. Move to bin (Distance = 5.0)
    move_action = next(a for a in env.get_actions() if a.name == "move robot1 start bin")
    start_wall_time = time_module.time()
    state = env.act(move_action)
    elapsed = time_module.time() - start_wall_time

    assert F("at", "robot1", "bin") in state.fluents
    assert elapsed >= 5.0

    # 2. Pick ball1
    pick_action = next(a for a in env.get_actions() if a.name == "pick robot1 bin ball1")
    state = env.act(pick_action)
    assert F("holding", "robot1", "ball1") in state.fluents

    # 3. Move to table (Distance from bin (-3,-4) to table (2,2) is sqrt(5^2 + 6^2) = 7.81)
    move_to_table = next(a for a in env.get_actions() if a.name == "move robot1 bin table")
    start_wall_time = time_module.time()
    state = env.act(move_to_table)
    elapsed = time_module.time() - start_wall_time
    assert F("at", "robot1", "table") in state.fluents
    assert elapsed >= 7.8

    # 4. Place ball1
    place_action = next(a for a in env.get_actions() if a.name == "place robot1 table ball1")
    state = env.act(place_action)
    assert F("at", "ball1", "table") in state.fluents
    assert F("free", "robot1") in state.fluents


def test_physical_environment_interrupt():
    """Test that physical moves can be interrupted with async actions."""
    scene = SamplePhysicalScene()
    registry = LocationRegistry(scene.locations)
    move_op = construct_move_operator_blocking(registry.move_time_fn(velocity=1.0))

    initial_state = State(0.0, {F("at", "robot1", "start"), F("free", "robot1")})

    env = SamplePhysicalEnvironment(
        scene=scene,
        state=initial_state,
        objects_by_type={
            "robot": {"robot1", "robot2"},
            "location": set(scene.locations.keys())
        },
        operators=[move_op],
        location_registry=registry
    )
    # Robot 2 is free to trigger interruption of Robot 1's long move
    env.fluents.add(F("at", "robot2", "start"))
    env.fluents.add(F("free", "robot2"))

    # Move to shelf2 (Distance = 4.0)
    move_action = next(a for a in env.get_actions() if a.name == "move robot1 start shelf2")
    state = env.act(move_action)

    assert F("at", "robot1", "robot1_loc") in state.fluents
    assert "robot1_loc" in registry
    coords = registry.get("robot1_loc")
    assert coords is not None
    # shelf2 is at (0, 4), so x should be 0 and y should be between 0 and 4
    assert coords[0] == 0.0
    assert 0.0 < coords[1] < 4.0


def test_physical_environment_state_during_execution():
    """Verify that state fluents are NOT updated until the physical action completes."""
    scene = SamplePhysicalScene()
    move_op = construct_move_operator_blocking(scene.get_move_cost_fn())

    initial_state = State(0.0, {F("at", "robot1", "start"), F("free", "robot1")})
    env = SamplePhysicalEnvironment(
        scene=scene,
        state=initial_state,
        objects_by_type={"robot": {"robot1"}, "location": set(scene.locations.keys())},
        operators=[move_op]
    )

    # Move to shelf1 (Distance = 3.0)
    move_action = next(a for a in env.get_actions() if a.name == "move robot1 start shelf1")

    act_thread_result = {}

    def run_act():
        act_thread_result["final_state"] = env.act(move_action)

    thread = threading.Thread(target=run_act)
    thread.start()

    # Give the thread a moment to start and trigger the move
    time_module.sleep(0.5)

    # WHILE MOVING:
    current_state = env.state
    assert F("at", "robot1", "start") not in current_state.fluents
    assert F("at", "robot1", "shelf1") not in current_state.fluents
    assert F("free", "robot1") not in current_state.fluents

    # Wait for completion (duration is 3.0s)
    thread.join(timeout=5.0)

    # AFTER MOVING:
    final_state = act_thread_result["final_state"]
    assert F("at", "robot1", "shelf1") in final_state.fluents
    assert F("free", "robot1") in final_state.fluents
    assert final_state.time >= 3.0
