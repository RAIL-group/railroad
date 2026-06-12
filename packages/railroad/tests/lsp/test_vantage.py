"""Tests for visibility-polygon vantage scoring and selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from railroad.environment.types import Pose
from railroad.experimental.unknown_search.types import Frontier
from railroad.lsp import count_cells_in_polygon, select_best_vantage


@dataclass
class _FakeRecord:
    robot: str
    time: float
    pose_cells: Pose
    pose_meters: tuple
    image: np.ndarray
    visibility_polygon: np.ndarray | None = None


def _square_polygon(r0: float, c0: float, r1: float, c1: float) -> np.ndarray:
    """Closed square loop covering [r0, r1] x [c0, c1]."""
    return np.array([
        [r0, r0, r1, r1, r0],
        [c0, c1, c1, c0, c0],
    ])


def _record(
    time: float,
    pose_rc: tuple[float, float],
    polygon: np.ndarray | None,
) -> _FakeRecord:
    return _FakeRecord(
        robot="robot1",
        time=time,
        pose_cells=Pose(pose_rc[0], pose_rc[1], 0.0),
        pose_meters=(0.0, 0.0, 0.0),
        image=np.zeros((2, 8, 3), dtype=np.uint8),
        visibility_polygon=polygon,
    )


def _frontier(cells: list[tuple[int, int]]) -> Frontier:
    arr = np.array(cells, dtype=int).T
    centroid = arr.mean(axis=1)
    return Frontier(
        id="f",
        centroid_row=int(round(centroid[0])),
        centroid_col=int(round(centroid[1])),
        cells=arr,
    )


def test_count_cells_in_polygon() -> None:
    polygon = _square_polygon(0, 0, 10, 10)
    inside = np.array([[2, 5], [2, 5]])
    outside = np.array([[20, 30], [20, 30]])
    assert count_cells_in_polygon(inside, polygon) == 2
    assert count_cells_in_polygon(outside, polygon) == 0
    mixed = np.array([[2, 20], [2, 20]])
    assert count_cells_in_polygon(mixed, polygon) == 1


def test_count_cells_inflation_pulls_in_boundary() -> None:
    polygon = _square_polygon(0, 0, 10, 10)
    # Cell (10, 5) has center (10.5, 5.5), just outside the square.
    boundary = np.array([[10], [5]])
    assert count_cells_in_polygon(boundary, polygon) == 0
    assert count_cells_in_polygon(boundary, polygon, inflation_radius=1.0) == 1


def test_count_cells_empty_inputs() -> None:
    polygon = _square_polygon(0, 0, 10, 10)
    assert count_cells_in_polygon(np.empty((2, 0)), polygon) == 0
    assert count_cells_in_polygon(np.array([[1], [1]]), np.empty((2, 0))) == 0


def test_select_best_vantage_prefers_more_visible_cells() -> None:
    frontier = _frontier([(5, 5), (5, 6), (5, 7)])
    sees_all = _record(2.0, (20.0, 20.0), _square_polygon(0, 0, 10, 10))
    sees_one = _record(1.0, (1.0, 1.0), _square_polygon(4, 4, 6, 6))
    best = select_best_vantage(frontier, [sees_one, sees_all])
    assert best is sees_all


def test_select_best_vantage_tie_breaks_by_distance() -> None:
    frontier = _frontier([(5, 5)])
    polygon = _square_polygon(0, 0, 10, 10)
    far = _record(1.0, (9.0, 9.0), polygon)
    near = _record(2.0, (5.0, 6.0), polygon)
    best = select_best_vantage(frontier, [far, near])
    assert best is near


def test_select_best_vantage_skips_missing_polygons_and_blind_views() -> None:
    frontier = _frontier([(5, 5)])
    no_polygon = _record(1.0, (5.0, 5.0), None)
    blind = _record(2.0, (50.0, 50.0), _square_polygon(40, 40, 60, 60))
    assert select_best_vantage(frontier, [no_polygon, blind]) is None
    assert select_best_vantage(frontier, []) is None
