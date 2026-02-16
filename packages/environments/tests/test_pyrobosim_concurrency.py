import time
import pytest
from pathlib import Path
import logging

from railroad.core import Fluent as F, State, get_action_by_name
from railroad.environment.pyrobosim import (
    DecoupledPyRoboSimEnvironment,
    PyRoboSimScene,
    get_default_pyrobosim_world_file_path
)
from railroad import operators

def test_pyrobosim_concurrency():
    """Test that multiple robots can move concurrently in decoupled mode.

    Verifies concurrency by comparing the time taken for a single robot move
    against the time taken for two robots to perform the same move simultaneously.
    """
    # Disable pyrobosim logs to keep output clean
    logging.disable(logging.INFO)

    world_file = get_default_pyrobosim_world_file_path()
    scene = PyRoboSimScene(world_file)
    # Use standard move operator
    move_op = operators.construct_move_operator_blocking(scene.get_move_cost_fn())

    def get_env(robots, record=False):
        initial_fluents = {F("at", r, f"{r}_loc") for r in robots} | {F("free", r) for r in robots}
        state = State(0.0, initial_fluents)
        return DecoupledPyRoboSimEnvironment(
            scene=scene, state=state,
            objects_by_type={"robot": set(robots), "location": set(scene.locations.keys()), "object": set()},
            operators=[move_op], show_plot=False, record_plots=record
        )

    # PHASE 1: Single Robot Move (robot2 to counter0)
    env1 = get_env(["robot2"], record=True)
    try:
        all_actions = env1.get_actions()
        m = next(a for a in all_actions if a.name == "move robot2 robot2_loc counter0")

        t_start = time.time()
        env1.act(m, do_interrupt=False)

        # Wait for completion
        while F("free", "robot2") not in env1.fluents:
            _ = env1.state
            time.sleep(0.1)

        duration_single = time.time() - t_start

        # Save animation for Phase 1
        output_path1 = Path("./data/proof_single_move.mp4")
        output_path1.parent.mkdir(parents=True, exist_ok=True)
        env1.save_animation(filepath=output_path1)
    finally:
        if hasattr(env1, "_client"):
            env1._client.stop()

    time.sleep(2.0) # Grace period between phases

    # PHASE 2: Concurrent Two-Robot Move (Both to counter0)
    env2 = get_env(["robot1", "robot2"], record=True)
    try:
        all_actions = env2.get_actions()
        m1 = next(a for a in all_actions if a.name == "move robot1 robot1_loc counter0")
        m2 = next(a for a in all_actions if a.name == "move robot2 robot2_loc counter0")

        t_start = time.time()
        env2.act(m1, do_interrupt=False)
        env2.act(m2, do_interrupt=False)

        while any(F("free", r) not in env2.fluents for r in ["robot1", "robot2"]):
            _ = env2.state
            time.sleep(0.1)

        duration_concurrent = time.time() - t_start

        # Save animation for Phase 2
        output_path2 = Path("./data/proof_concurrent_moves.mp4")
        env2.save_animation(filepath=output_path2)

        # Assertion: Concurrent time should be similar to single move time (with some slack)
        assert duration_concurrent < duration_single * 1.3, f"Concurrency failed: Single={duration_single:.2f}s, Concurrent={duration_concurrent:.2f}s"

    finally:
        if hasattr(env2, "_client"):
            env2._client.stop()

if __name__ == "__main__":
    test_pyrobosim_concurrency()
