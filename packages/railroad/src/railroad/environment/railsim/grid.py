"""Occupancy-grid utilities (vendored replacements for the lab `gridmap`)."""

from __future__ import annotations

import numpy as np
import scipy.ndimage


def _disk(radius_cells: float) -> np.ndarray:
    r = int(np.ceil(radius_cells))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx**2 + yy**2) <= radius_cells**2


def inflate_grid(occ_grid: np.ndarray, inflation_radius_cells: float) -> np.ndarray:
    """Dilate occupied cells (value >= 0.5) by a circular structuring element."""
    occupied = occ_grid >= 0.5
    if inflation_radius_cells > 0:
        occupied = scipy.ndimage.binary_dilation(occupied, structure=_disk(inflation_radius_cells))
    return occupied.astype(float)


def cells_connected(occ_grid: np.ndarray,
                    cell_a: tuple[int, int],
                    cell_b: tuple[int, int],
                    inflation_radius_cells: float = 0.0) -> bool:
    """Whether two free cells remain connected after inflating obstacles."""
    inflated = inflate_grid(occ_grid, inflation_radius_cells)
    free = inflated < 0.5
    labels, _ = scipy.ndimage.label(free)
    label_a = labels[cell_a]
    label_b = labels[cell_b]
    return bool(label_a != 0 and label_a == label_b)
