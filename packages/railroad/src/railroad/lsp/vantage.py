"""Best-vantage selection: which panorama saw a frontier best.

Each :class:`PanoRecord` carries the visibility polygon of the laser scan
taken at its pose. A record's view of a frontier is scored by how many
frontier cells fall inside that polygon (inflated slightly to absorb
rasterization artifacts at the polygon boundary).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import numpy as np
from matplotlib.path import Path

from railroad.experimental.unknown_search.types import Frontier

if TYPE_CHECKING:
    from railroad.environment.railsim import PanoRecord


def _distances_to_boundary(
    points: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    """Min distance from each point (Nx2) to the polygon boundary (2xM loop)."""
    seg_starts = vertices[:, :-1].T  # Mx2
    seg_ends = vertices[:, 1:].T
    deltas = seg_ends - seg_starts
    lengths_sq = np.maximum(np.sum(deltas**2, axis=1), 1e-12)
    # Project every point onto every segment, clamped to the segment.
    rel = points[:, None, :] - seg_starts[None, :, :]  # NxMx2
    t = np.clip(np.sum(rel * deltas[None, :, :], axis=2) / lengths_sq, 0.0, 1.0)
    closest = seg_starts[None, :, :] + t[..., None] * deltas[None, :, :]
    return np.min(np.linalg.norm(points[:, None, :] - closest, axis=2), axis=1)


def count_cells_in_polygon(
    cells: np.ndarray,
    polygon_vertices: np.ndarray,
    inflation_radius: float = 0.0,
) -> int:
    """Count grid cells (2xN, row/col) seen by the (inflated) polygon.

    *polygon_vertices* is a 2xM closed vertex loop in cell coordinates.
    A cell counts when its center lies inside the polygon, or — to absorb
    rasterization artifacts at the boundary — within *inflation_radius*
    of the polygon boundary.
    """
    if cells.size == 0 or polygon_vertices.size == 0:
        return 0
    path = Path(polygon_vertices.T, closed=True)
    points = cells.T.astype(float) + 0.5
    visible = np.asarray(path.contains_points(points), dtype=bool)
    if inflation_radius > 0.0 and not visible.all():
        near_boundary = (
            _distances_to_boundary(points, polygon_vertices) <= inflation_radius
        )
        visible |= near_boundary
    return int(np.count_nonzero(visible))


def select_best_vantage(
    frontier: Frontier,
    records: Sequence["PanoRecord"],
    inflation_radius: float = 1.0,
) -> "PanoRecord | None":
    """Return the record with the best view of *frontier*, if any.

    Records are ranked by the number of frontier cells inside their
    (inflated) visibility polygon, breaking ties by distance from the
    vantage pose to the frontier centroid (closer is better), then by
    capture time (earlier is better, for determinism). Records without a
    visibility polygon are skipped; returns None when no record sees any
    frontier cell.
    """
    centroid = (float(frontier.centroid_row), float(frontier.centroid_col))

    best: "PanoRecord | None" = None
    best_key: tuple[float, float, float] | None = None
    for record in records:
        if record.visibility_polygon is None:
            continue
        num_visible = count_cells_in_polygon(
            frontier.cells, record.visibility_polygon, inflation_radius
        )
        if num_visible == 0:
            continue
        pose = record.pose_cells
        distance = math.hypot(
            centroid[0] - float(pose.x), centroid[1] - float(pose.y)
        )
        key = (-float(num_visible), distance, float(record.time))
        if best_key is None or key < best_key:
            best = record
            best_key = key
    return best
