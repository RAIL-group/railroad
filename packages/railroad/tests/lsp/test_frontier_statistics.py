"""Tests for the oracle / fixed-prior / learned frontier-statistics estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import pytest

from railroad.environment.types import Pose
from railroad.experimental.unknown_search.types import Frontier
from railroad.lsp import (
    DEFAULT_FRONTIER_STATISTICS,
    FixedPriorFrontierStatistics,
    FrontierObservation,
    FrontierStatistics,
    LearnedFrontierStatistics,
    OracleFrontierLabel,
    OracleFrontierStatistics,
)


@dataclass
class _FakeRecord:
    robot: str
    time: float
    pose_cells: Pose
    pose_meters: tuple
    image: np.ndarray
    visibility_polygon: np.ndarray | None = None


@dataclass
class _FakeEnv:
    """Just enough environment for estimator refresh.

    Exactly the ``FrontierStatisticsEnvironment`` protocol (plus pano_records):
    what the robot has *observed*, never ground truth. An estimator that needs
    to know more than the robot carries that knowledge itself.
    """

    frontiers: Dict[str, Frontier] = field(default_factory=dict)
    pano_records: List[Any] = field(default_factory=list)
    goal_cell: tuple = (50, 50)
    observed_grid: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 4), dtype=float)
    )


def _frontier(fid: str, cells: list[tuple[int, int]]) -> Frontier:
    arr = np.array(cells, dtype=int).T
    centroid = arr.mean(axis=1)
    return Frontier(
        id=fid,
        centroid_row=int(round(centroid[0])),
        centroid_col=int(round(centroid[1])),
        cells=arr,
    )


def test_fixed_prior_is_constant() -> None:
    estimator = FixedPriorFrontierStatistics(
        prob_feasible=0.7, delta_success_cost=1.0, exploration_cost=12.0
    )
    estimator.refresh(_FakeEnv())  # no-op
    assert estimator.get("robot1", "anything") == FrontierStatistics(0.7, 1.0, 12.0)


def test_oracle_label_maps_to_statistics() -> None:
    """The label -> planner-statistics conversion, the estimator's only arithmetic."""
    from_label = OracleFrontierStatistics.statistics_from_label

    feasible = from_label(OracleFrontierLabel("f_yes", 1.0, 30.0, 25.0, None, "h1"))
    assert feasible.prob_feasible == 1.0
    # Delta success cost = true cost - optimistic cost.
    assert feasible.delta_success_cost == pytest.approx(5.0)

    infeasible = from_label(OracleFrontierLabel("f_no", 0.0, None, None, 8.0, "h2"))
    assert infeasible.prob_feasible == 0.0
    assert infeasible.exploration_cost == 8.0

    # Missing or degenerate labels fall back to the default.
    assert from_label(None) == DEFAULT_FRONTIER_STATISTICS
    assert (
        from_label(OracleFrontierLabel("f_degenerate", 0.0, None, None, None, "h3"))
        == DEFAULT_FRONTIER_STATISTICS
    )


def test_oracle_estimator_requires_its_own_true_map() -> None:
    """Ground truth is carried, never read off the environment.

    An environment's "true grid" means whatever it simulates against, which is
    not always the real world (offline replay hands its arena a confinement
    grid). Requiring the map up front makes the mistake unwritable.
    """
    with pytest.raises(TypeError):
        OracleFrontierStatistics()  # ty: ignore[missing-argument]


def test_learned_estimator_predicts_from_observations() -> None:
    frontier = _frontier("f1", [(5, 5), (5, 6)])
    record = _FakeRecord(
        robot="robot1",
        time=1.0,
        pose_cells=Pose(10.0, 10.0, 0.0),
        pose_meters=(0.0, 0.0, 0.0),
        image=np.zeros((4, 16, 3), dtype=np.uint8),
        visibility_polygon=np.array([
            [0.0, 0.0, 20.0, 20.0, 0.0],
            [0.0, 20.0, 20.0, 0.0, 0.0],
        ]),
    )
    env = _FakeEnv(frontiers={"f1": frontier}, pano_records=[record])

    seen: List[FrontierObservation] = []

    def model(
        observations: Sequence[FrontierObservation],
    ) -> Sequence[FrontierStatistics]:
        seen.extend(observations)
        return [FrontierStatistics(0.9, 2.0, 5.0)] * len(observations)

    estimator = LearnedFrontierStatistics(model)
    estimator.refresh(env)

    # The model saw one observation, shaped like the training data.
    assert len(seen) == 1
    assert seen[0].image.shape == record.image.shape
    assert isinstance(seen[0].frontier_xy_ego, tuple)
    assert estimator.get("robot1", "f1") == FrontierStatistics(0.9, 2.0, 5.0)


def test_learned_estimator_defaults_without_vantage() -> None:
    # A frontier no panorama has seen falls back to the default; the
    # model is never called with an empty batch.
    frontier = _frontier("f1", [(5, 5)])
    env = _FakeEnv(frontiers={"f1": frontier}, pano_records=[])

    def model(
        observations: Sequence[FrontierObservation],
    ) -> Sequence[FrontierStatistics]:
        raise AssertionError("model must not be called without observations")

    estimator = LearnedFrontierStatistics(model)
    estimator.refresh(env)
    assert estimator.get("robot1", "f1") == DEFAULT_FRONTIER_STATISTICS


def test_learned_estimator_drops_stale_predictions() -> None:
    frontier = _frontier("f1", [(5, 5), (5, 6)])
    record = _FakeRecord(
        robot="robot1",
        time=1.0,
        pose_cells=Pose(10.0, 10.0, 0.0),
        pose_meters=(0.0, 0.0, 0.0),
        image=np.zeros((4, 16, 3), dtype=np.uint8),
        visibility_polygon=np.array([
            [0.0, 0.0, 20.0, 20.0, 0.0],
            [0.0, 20.0, 20.0, 0.0, 0.0],
        ]),
    )

    def model(
        observations: Sequence[FrontierObservation],
    ) -> Sequence[FrontierStatistics]:
        return [FrontierStatistics(0.9, 2.0, 5.0)] * len(observations)

    estimator = LearnedFrontierStatistics(model)
    estimator.refresh(_FakeEnv(frontiers={"f1": frontier}, pano_records=[record]))
    assert estimator.get("robot1", "f1") != DEFAULT_FRONTIER_STATISTICS

    # After the frontier disappears, its prediction is gone too.
    estimator.refresh(_FakeEnv(frontiers={}, pano_records=[record]))
    assert estimator.get("robot1", "f1") == DEFAULT_FRONTIER_STATISTICS
