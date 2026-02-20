"""Plotting utilities for ProcTHOR environments."""

from typing import Any

import numpy as np

try:
    from skimage.morphology import erosion
    HAS_PLOTTING_DEPS = True
except ImportError:
    HAS_PLOTTING_DEPS = False

COLLISION_VAL = 1
FREE_VAL = 0
FOOT_PRINT = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])


def make_plotting_grid(grid_map: np.ndarray) -> np.ndarray:
    """Convert occupancy grid to RGB plotting grid."""
    if not HAS_PLOTTING_DEPS:
        raise ImportError("Plotting requires scikit-image: pip install scikit-image")

    grid = np.ones([grid_map.shape[0], grid_map.shape[1], 3]) * 0.75
    collision = grid_map >= 0.5
    thinned = erosion(collision, footprint=FOOT_PRINT)
    boundary = np.logical_xor(collision, thinned)
    free = np.logical_and(grid_map < 0.5, grid_map >= FREE_VAL)

    grid[:, :, :][free] = 1
    grid[:, :, 0][boundary] = 0
    grid[:, :, 1][boundary] = 0
    grid[:, :, 2][boundary] = 0

    return grid


def plot_grid(ax: Any, grid: np.ndarray) -> None:
    """Plot occupancy grid."""
    plotting_grid = make_plotting_grid(grid.T)
    ax.imshow(plotting_grid, origin="upper")
