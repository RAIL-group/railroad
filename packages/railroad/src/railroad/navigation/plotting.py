"""Plotting utilities for occupancy grids."""

from typing import Any
import numpy as np
from .constants import FREE_VAL, UNOBSERVED_VAL
from skimage.morphology import erosion


_BACKGROUND_GRAY = 0.75
"""Default fill for unobserved and outside-the-map cells."""


def make_plotting_grid(grid_map: np.ndarray) -> np.ndarray:
    """Convert occupancy grid to RGB plotting grid.

    Handles three cell types:
    - Free (value in [FREE_VAL, 0.5)): white
    - Obstacle (value >= 0.5): gray background with black erosion boundary
    - Unobserved / outside-map (value == UNOBSERVED_VAL or default): gray
    """
    grid = np.ones([grid_map.shape[0], grid_map.shape[1], 3]) * _BACKGROUND_GRAY
    collision = grid_map >= 0.5
    thinned = erosion(collision, footprint=np.ones((3, 3)))
    boundary = np.logical_xor(collision, thinned)
    free = np.logical_and(grid_map < 0.5, grid_map >= FREE_VAL)

    grid[:, :, :][free] = 1
    grid[:, :, 0][boundary] = 0
    grid[:, :, 1][boundary] = 0
    grid[:, :, 2][boundary] = 0

    return grid


def make_plotting_grid_rgba(grid_map: np.ndarray) -> np.ndarray:
    """Convert occupancy grid to RGBA plotting grid.

    Same rendering as ``make_plotting_grid`` for observed cells, but returns
    an (H, W, 4) array. Unobserved cells are fully transparent (alpha=0);
    observed cells are fully opaque (alpha=1).
    """
    rgb = make_plotting_grid(grid_map)
    rgba = np.ones([grid_map.shape[0], grid_map.shape[1], 4])
    rgba[:, :, :3] = rgb
    rgba[grid_map == UNOBSERVED_VAL, 3] = 0.0
    return rgba


def plot_grid_background(
    ax: Any,
    observed_grid: np.ndarray,
    true_grid: np.ndarray | None = None,
) -> None:
    """Render occupancy grid background with optional faded true-grid underlay.

    If *true_grid* is provided and *observed_grid* contains unobserved cells,
    the true grid is rendered at low alpha underneath a transparent-unobserved
    overlay of the observed grid.  Otherwise the observed grid is rendered
    directly.

    Both paths transpose the grid (``grid.T``) before rendering, matching the
    existing ``plot_grid`` convention.
    """
    has_unknown = bool(np.any(observed_grid == UNOBSERVED_VAL))
    if true_grid is not None and has_unknown:
        # Composite in numpy so the true-grid underlay blends against the
        # background gray rather than the white axes background.
        base = np.full(3, _BACKGROUND_GRAY)
        true_rgb = make_plotting_grid(true_grid.T)
        observed_rgba = make_plotting_grid_rgba(observed_grid.T)

        # Faint true grid on gray base
        underlay_alpha = 0.35
        composite = base * (1 - underlay_alpha) + true_rgb * underlay_alpha

        # Opaque observed overlay on top
        obs_alpha = observed_rgba[:, :, 3:4]
        composite = composite * (1 - obs_alpha) + observed_rgba[:, :, :3] * obs_alpha

        ax.imshow(composite, origin="upper", zorder=0)
    else:
        ax.imshow(make_plotting_grid(observed_grid.T), origin="upper", zorder=0)


def plot_grid(ax: Any, grid: np.ndarray) -> None:
    """Plot occupancy grid."""
    plotting_grid = make_plotting_grid(grid.T)
    ax.imshow(plotting_grid, origin="upper")
