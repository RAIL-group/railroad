import time
import pytest
from pathlib import Path

from railroad._bindings import State, Fluent
from railroad.core import Fluent as F
from railroad.environment.pyrobosim import (
    PyRoboSimScene,
    PyRoboSimEnvironment,
    get_default_pyrobosim_world_file_path
)
from railroad.operators import (
    construct_move_operator_blocking,
    construct_pick_operator_blocking,
    construct_place_operator_blocking
)

def test_pyrobosim_regular_execution():
    """Test that the environment can execute actions and state is updated."""
    world_file = get_default_pyrobosim_world_file_path()
    scene = PyRoboSimScene(world_file, show_plot=False)

    # Define operators
    move_op = construct_move_operator_blocking(scene.get_move_cost_fn())
    pick_op = construct_pick_operator_blocking(0.5)
    place_op = construct_place_operator_blocking(0.5)

    # Initial state: robot1 at kitchen_loc (initial pose), free.
    initial_fluents = {
        F("at", "robot1", "robot1_loc"),
        F("free", "robot1"),
        F("at", "banana0", "table0")
    }
    initial_state = State(0.0, initial_fluents)

    objects_by_type = {
        "robot": {"robot1", "robot2"},
        "location": set(scene.locations.keys()) | {"bedroom", "kitchen", "bathroom"},
        "object": {"banana0", "apple0"}
    }

    env = PyRoboSimEnvironment(
        scene=scene,
        state=initial_state,
        objects_by_type=objects_by_type,
        operators=[move_op, pick_op, place_op],
    )

    try:
        # 1. Move to table0
        actions = env.get_actions()
        move_to_table = next(a for a in actions if a.name == "move robot1 robot1_loc table0")
        env.act(move_to_table)
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
        scene.close()
