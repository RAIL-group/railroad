"""Fakes shared by the LSP tests.

`_FakeRecord` was copy-pasted byte-identically into four test modules, and
`_frontier` into four more (three identical, one with the id hard-coded). A
module rather than `conftest.py` because these are imported by name, and
several `conftest.py` files exist in this repo for static tooling to confuse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from railroad.environment.types import Pose
from railroad.experimental.unknown_search.types import Frontier


@dataclass
class FakeRecord:
    """Stand-in for a `PanoRecord`, carrying only what the LSP code reads."""

    robot: str
    time: float
    pose_cells: Pose
    pose_meters: tuple
    image: np.ndarray
    visibility_polygon: np.ndarray | None = None


def square_polygon(r0: float, c0: float, r1: float, c1: float) -> np.ndarray:
    """Closed square loop covering [r0, r1] x [c0, c1]."""
    return np.array([
        [r0, r0, r1, r1, r0],
        [c0, c1, c1, c0, c0],
    ])


def make_frontier(fid: str, cells: list[tuple[int, int]]) -> Frontier:
    """A `Frontier` over `cells`, with the centroid computed from them."""
    arr = np.array(cells, dtype=int).T
    centroid = arr.mean(axis=1)
    return Frontier(
        id=fid,
        centroid_row=int(round(centroid[0])),
        centroid_col=int(round(centroid[1])),
        cells=arr,
    )
