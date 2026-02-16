import time
import pytest
from pathlib import Path

from railroad._bindings import State, Fluent
from railroad.core import Fluent as F
from railroad.environment.pyrobosim import (
    PyRoboSimScene,
    DecoupledPyRoboSimEnvironment,
    get_default_pyrobosim_world_file_path
)
from railroad.operators import (
    construct_move_operator_blocking,
    construct_pick_operator_blocking,
    construct_place_operator_blocking,
    construct_no_op_operator
)

def test_pyrobosim_decoupled_execution():
    """Test that the decoupled environment can execute actions and sync state."""
    world_file = get_default_pyrobosim_world_file_path()
    scene = PyRoboSimScene(world_file)

    # Define operators
    move_op = construct_move_operator_blocking(scene.get_move_cost_fn())
    pick_op = construct_pick_operator_blocking(0.5)
    place_op = construct_place_operator_blocking(0.5)

    # Initial state: robot1 at kitchen_loc (initial pose), free.
    # We want to pick banana0 from table0 and move to bedroom.
    initial_fluents = {
        F("at", "robot1", "robot1_loc"),
        F("free", "robot1"),
        F("at", "banana0", "table0")
    }
    initial_state = State(0.0, initial_fluents)

    objects_by_type = {
        "robot": {"robot1"},
        "location": set(scene.locations.keys()) | {"bedroom", "kitchen", "bathroom"},
        "object": {"banana0", "apple0"}
    }

    env = DecoupledPyRoboSimEnvironment(
        scene=scene,
        state=initial_state,
        objects_by_type=objects_by_type,
        operators=[move_op, pick_op, place_op],
        show_plot=False # GUI off for CI
    )

    try:
        # 1. Move to table0
        actions = env.get_actions()
        move_to_table = next(a for a in actions if a.name == "move robot1 robot1_loc table0")
        print(f"\nExecuting: {move_to_table.name}")
        env.act(move_to_table)
        print(f"Fluents after move: {env.fluents}")
        assert F("at", "robot1", "table0") in env.fluents

        # 2. Pick banana0
        pick_action = next(a for a in env.get_actions() if a.name == "pick robot1 table0 banana0")
        env.act(pick_action)
        assert F("holding", "robot1", "banana0") in env.fluents
        assert F("at", "banana0", "table0") not in env.fluents

        # 3. Move to bedroom
        move_to_bedroom = next(a for a in env.get_actions() if a.name == "move robot1 table0 bedroom")
        env.act(move_to_bedroom)
        assert F("at", "robot1", "bedroom") in env.fluents

        # 4. Place banana0
        bedroom_locs = [l for l in scene.locations.keys() if "desk" in l or "bed" in l]
        target_loc = bedroom_locs[0] if bedroom_locs else "my_desk"

        # Move to target_loc first if needed
        move_to_desk = next(a for a in env.get_actions() if a.name == f"move robot1 bedroom {target_loc}")
        env.act(move_to_desk)

        place_action = next(a for a in env.get_actions() if a.name == f"place robot1 {target_loc} banana0")
        env.act(place_action)

        assert F("at", "banana0", target_loc) in env.fluents
        assert F("free", "robot1") in env.fluents

    finally:
        # Ensure process is cleaned up
        if hasattr(env, "_client"):
            env._client.stop()


def test_pyrobosim_decoupled_concurrency():
    """Test that multiple robots can act concurrently in decoupled mode using no_op."""
    world_file = get_default_pyrobosim_world_file_path()
    scene = PyRoboSimScene(world_file)
    noop_op = construct_no_op_operator(1.0) # 1 second duration

    initial_fluents = {
        F("at", "robot1", "robot1_loc"),
        F("free", "robot1"),
        F("at", "robot2", "robot2_loc"),
        F("free", "robot2")
    }
    initial_state = State(0.0, initial_fluents)

    env = DecoupledPyRoboSimEnvironment(
        scene=scene,
        state=initial_state,
        objects_by_type={
            "robot": {"robot1", "robot2"},
            "location": set(scene.locations.keys()),
            "object": set()
        },
        operators=[noop_op],
        show_plot=False
    )

    try:
        # 1. Dispatch no_op for robot1
        actions = env.get_actions()
        noop1 = next(a for a in actions if a.name == "no_op robot1")

        start_time = time.time()
        state1 = env.act(noop1)
        # act() should return immediately because robot2 is free
        assert time.time() - start_time < 2.0

        # robot1 should be busy (free removed)
        assert F("free", "robot1") not in state1.fluents
        # robot2 should be free
        assert F("free", "robot2") in state1.fluents

        # 2. Dispatch no_op for robot2
        noop2 = next(a for a in actions if a.name == "no_op robot2")

        state2 = env.act(noop2)
        # Now both busy. act() waits for ONE to finish.
        # Since they started 1s apart (approx), robot1 finishes first.

        # robot1 should be free now
        assert F("free", "robot1") in env.fluents

        # Wait for robot2
        time.sleep(1.5)
        # Accessing state triggers _update_skills which checks physical status
        _ = env.state
        assert F("free", "robot2") in env.fluents

    finally:
        if hasattr(env, "_client"):
            env._client.stop()
