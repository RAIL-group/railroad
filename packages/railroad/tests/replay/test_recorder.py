"""Tests for building a RolloutLog from a live environment (recorder)."""

from __future__ import annotations

import numpy as np

from railroad.environment.types import Pose
from railroad.lsp.frontier_statistics import FixedPriorFrontierStatistics
from railroad.replay.recorder import build_rollout_log
from railroad.replay.replay_env import build_replay_env
from railroad.replay.serialization import load_rollout_log, save_rollout_log

from railroad.core import get_action_by_name
from railroad.replay.replay_env import goal_fluent

from .conftest import build_log_from_ascii, explore_first_select

MAP = """
##########
#S......G#
#...?....#
##########
"""


def _live_env():
    log = build_log_from_ascii(MAP)
    env = build_replay_env(log, FixedPriorFrontierStatistics(prob_feasible=0.8))
    return log, env


def test_build_rollout_log_snapshots_env() -> None:
    src_log, env = _live_env()
    recorded = build_rollout_log(
        env,
        goal_cell=src_log.goal_cell,
        robot_starts={"robot1": Pose(1.0, 1.0, 0.0)},
        env_name="unit",
        seed=3,
    )
    assert recorded.goal_cell == src_log.goal_cell
    assert recorded.env_name == "unit"
    assert recorded.seed == 3
    assert recorded.robot_starts == {"robot1": (1.0, 1.0, 0.0)}
    # The recorded grid is the env's observed grid; a frontier was seen.
    np.testing.assert_array_equal(recorded.recorded_grid, env.observed_grid)
    assert len(recorded.subgoals) == len(env.frontiers)
    for subgoal in recorded.subgoals:
        assert subgoal.signature
        assert subgoal.cells.shape[0] == 2


def test_records_makespan_as_actual_total_cost() -> None:
    """actual_total_cost is always the deployment makespan (env.state.time),
    never a silent zero — it is the cost replay bounds are compared against."""
    src_log, env = _live_env()
    goal = goal_fluent(list(src_log.robots))
    for _ in range(50):
        if goal.evaluate(env.state.fluents):
            break
        actions = env.get_actions()
        if not actions:
            break
        name = explore_first_select(env, actions, goal)
        if name in ("NONE", ""):
            break
        env.act(get_action_by_name(actions, name))

    assert env.state.time > 0  # the deployment actually moved
    recorded = build_rollout_log(
        env,
        goal_cell=src_log.goal_cell,
        robot_starts={"robot1": Pose(1.0, 1.0, 0.0)},
    )
    assert recorded.actual_total_cost == float(env.state.time)
    assert recorded.actual_total_cost > 0


def test_recorder_never_leaks_uninspected_container_contents() -> None:
    """Soundness: a revealed-but-UNsearched container's true contents must never
    enter the log. Replay may only see what the deployment observed; leaking
    ground-truth contents of a container the deployment never opened would give
    replay information no real deployment had (breaks the lower-bound guarantee).
    """
    from railroad.core import Fluent
    from railroad.environment.symbolic import LocationRegistry

    class _State:
        # Only c_searched was actually searched (searched fluent present).
        fluents = {Fluent("searched c_searched apple")}
        time = 42.0

    class _Env:
        observed_grid = np.zeros((5, 5))
        objects_by_type = {"container": {"c_searched", "c_revealed"}}
        # Full ground truth for BOTH (as the live env holds it) — the recorder
        # must not copy the unsearched one's contents into the log.
        _objects_at_locations = {"c_searched": {"apple"}, "c_revealed": {"ring"}}
        _location_registry = LocationRegistry(
            {"c_searched": np.array([1.0, 1.0]), "c_revealed": np.array([2.0, 3.0])}
        )
        pano_records: list = []
        config = None
        state = _State()

    log = build_rollout_log(
        _Env(),
        goal_cell=(0, 0),
        robot_starts={"robot1": Pose(0.0, 0.0, 0.0)},
        problem_class="object-search",
    )
    by_sig = {s.signature: s for s in log.subgoals}
    # Searched container: its observed contents are recorded.
    assert by_sig["c_searched"].contents == ("apple",)
    # Revealed-but-unsearched container: coords kept (cell was observed, so the
    # candidate may navigate there) but contents MUST be empty (no leak of "ring").
    assert by_sig["c_revealed"].centroid == (2, 3)
    assert by_sig["c_revealed"].contents == ()


def test_recorded_log_round_trips(tmp_path) -> None:
    src_log, env = _live_env()
    recorded = build_rollout_log(
        env,
        goal_cell=src_log.goal_cell,
        robot_starts={"robot1": Pose(1.0, 1.0, 0.0)},
    )
    save_rollout_log(recorded, tmp_path)
    loaded = load_rollout_log(tmp_path)
    np.testing.assert_array_equal(loaded.recorded_grid, recorded.recorded_grid)
    assert [s.signature for s in loaded.subgoals] == [
        s.signature for s in recorded.subgoals
    ]
