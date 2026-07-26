"""Served-vantage replay: a learned estimator consumes recorded panoramas.

GL-free — synthetic ``PanoRecord``s (no rendering), mirroring ``_FakeRecord`` in
``tests/lsp/test_frontier_statistics.py``. Proves that the replay environment
serves ``log.pano_records`` to ``LearnedFrontierStatistics`` (the real
best-vantage path) and that the *faked* model's output actually reaches the
planner — so a trained network would too, with no other change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from railroad.environment.types import Pose
from railroad.lsp.frontier_statistics import (
    DEFAULT_FRONTIER_STATISTICS,
    LearnedFrontierStatistics,
)
from railroad.replay import build_replay_env
from railroad.replay import ConstantFrontierStatisticsModel

from .conftest import build_log_from_ascii


def _arena(log, estimator):
    """A replay arena with *estimator* applied as the candidate policy."""
    env = build_replay_env(log)
    env.apply_policy(estimator)
    return env

MAP = """
##########
#S......G#
#...?....#
##########
"""


@dataclass
class _FakeRecord:
    """Duck-typed PanoRecord (no railsim/GL)."""

    robot: str
    time: float
    pose_cells: Pose
    pose_meters: tuple
    image: np.ndarray
    visibility_polygon: np.ndarray | None = None


def _covering_record(grid: np.ndarray) -> _FakeRecord:
    """A pano whose visibility polygon covers the whole grid (sees every frontier)."""
    h, w = grid.shape
    return _FakeRecord(
        robot="robot1",
        time=1.0,
        pose_cells=Pose(float(h // 2), float(w // 2), 0.0),
        pose_meters=(0.0, 0.0, 0.0),
        image=np.zeros((4, 16, 3), dtype=np.uint8),
        visibility_polygon=np.array(
            [
                [0.0, 0.0, float(h), float(h), 0.0],
                [0.0, float(w), float(w), 0.0, 0.0],
            ]
        ),
    )


def test_replay_serves_recorded_panos_to_learned_estimator() -> None:
    log = build_log_from_ascii(MAP)
    log.pano_records = [_covering_record(log.recorded_grid)]

    estimator = LearnedFrontierStatistics(ConstantFrontierStatisticsModel(prob_feasible=0.9, exploration_cost=8.0))
    env = _arena(log, estimator)

    assert env.pano_records, "replay env must expose the recorded pano buffer"
    assert env.frontiers, "the '?' pocket should yield a frontier"
    fid = next(iter(env.frontiers))
    # The faked model returns the 'optimistic' preset (0.9), not the fallback
    # default (0.5) — proving the panos were served and the model output used.
    assert env.frontier_statistics.get("robot1", fid).prob_feasible == pytest.approx(0.9)


def test_learned_estimator_falls_back_without_panos() -> None:
    log = build_log_from_ascii(MAP)  # no pano_records
    estimator = LearnedFrontierStatistics(ConstantFrontierStatisticsModel(prob_feasible=0.9, exploration_cost=8.0))
    env = _arena(log, estimator)

    assert env.pano_records == []
    fid = next(iter(env.frontiers))
    assert env.frontier_statistics.get("robot1", fid) == DEFAULT_FRONTIER_STATISTICS


CORRIDOR = """
##########
#S......G#
##########
"""


def _record_at(grid, pose, tag) -> _FakeRecord:
    h, w = grid.shape
    return _FakeRecord(
        robot="robot1",
        time=0.0,
        pose_cells=Pose(float(pose[0]), float(pose[1]), 0.0),
        pose_meters=(0.0, 0.0, 0.0),
        image=np.full((4, 16, 3), tag, dtype=np.uint8),
        visibility_polygon=np.array(
            [[0.0, 0.0, float(h), float(h), 0.0], [0.0, float(w), float(w), 0.0, 0.0]]
        ),
    )


def test_env_serves_recorded_pano_nearest_each_pose() -> None:
    """The replay env retrieves the recorded panorama nearest the robot's pose,
    growing the onboard buffer along the trajectory (not the deployment timeline)."""
    log = build_log_from_ascii(CORRIDOR)
    grid = log.recorded_grid
    log.pano_records = [
        _record_at(grid, (1, 1), tag=10),  # near the start
        _record_at(grid, (1, 8), tag=20),  # far end
    ]

    env = build_replay_env(log)
    # Initial sense at the start pose serves the nearest recorded pano (tag 10).
    assert env.pano_records, "an onboard observation should be served at the start"
    assert int(env.pano_records[-1].image[0, 0, 0]) == 10

    # Sensing from the far end serves the other recorded pano (tag 20).
    env.observe_from_pose("robot1", Pose(1.0, 8.0, 0.0), 5.0, allow_interrupt=False)
    assert int(env.pano_records[-1].image[0, 0, 0]) == 20

    # Both distinct recorded views were served (de-duplicated, not the static
    # deployment buffer replayed by timestamp).
    served_tags = {int(r.image[0, 0, 0]) for r in env.pano_records}
    assert served_tags == {10, 20}
    # Served observations carry the replay times, not the recorded (0.0) ones.
    assert any(r.time == 5.0 for r in env.pano_records)


def test_different_constant_models_yield_different_stats() -> None:
    log = build_log_from_ascii(MAP)
    log.pano_records = [_covering_record(log.recorded_grid)]

    opt = _arena(log, LearnedFrontierStatistics(ConstantFrontierStatisticsModel(prob_feasible=0.9, exploration_cost=8.0)))
    cau = _arena(log, LearnedFrontierStatistics(ConstantFrontierStatisticsModel(prob_feasible=0.3, exploration_cost=20.0)))
    fid_opt = next(iter(opt.frontiers))
    fid_cau = next(iter(cau.frontiers))

    assert opt.frontier_statistics.get("r", fid_opt).prob_feasible == pytest.approx(0.9)
    assert cau.frontier_statistics.get("r", fid_cau).prob_feasible == pytest.approx(0.3)
