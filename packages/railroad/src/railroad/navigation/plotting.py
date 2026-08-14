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


PHOTO_UNDERLAY_ALPHA = {
    "free": 0.15, "boundary": 1.0, "obstacle": 0.15, "unknown": 0.95,
}
"""Per-class opacity when an aligned scene image is drawn underneath.

Obstacle *outlines* stay opaque so the map still reads as a map, but obstacle
*interiors* go nearly transparent -- that is where the scene image earns its
place, and in ProcTHOR every cell that is not an agent-reachable position is
"occupied", so an opaque interior would blot out the entire house.

Unobserved cells stay nearly opaque. In an exploration run the observed /
unobserved boundary is the most important thing on the plot, and letting the
image bleed through the unobserved side blurs exactly that line -- it also
shows the robot something it has not seen. Free space takes only a light wash,
enough to lift the floor under a trail without hiding what is on it.
"""


def make_plotting_grid_alpha(
    grid_map: np.ndarray, *,
    free: float, boundary: float, obstacle: float, unknown: float,
) -> np.ndarray:
    """Per-cell opacity mask matching ``make_plotting_grid``'s classes.

    Classifies exactly as ``make_plotting_grid`` does, so the two cannot drift
    apart, but splits its single gray fill into *obstacle* interiors and
    *unknown* (unobserved or outside-the-map) cells -- they render alike, yet
    want opposite treatment over a scene image. Returns an (H, W) float array
    for ``imshow(alpha=...)``.
    """
    collision = grid_map >= 0.5
    thinned = erosion(collision, footprint=np.ones((3, 3)))
    is_boundary = np.logical_xor(collision, thinned)
    is_free = np.logical_and(grid_map < 0.5, grid_map >= FREE_VAL)

    alpha = np.full(grid_map.shape, unknown, dtype=float)
    alpha[collision] = obstacle
    alpha[is_free] = free
    alpha[is_boundary] = boundary
    return alpha


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


def make_frontier_overlay_rgba(
    grid_shape: tuple[int, ...],
    overlays: Any,
    *,
    alpha: float = 0.8,
) -> np.ndarray:
    """RGBA overlay coloring frontier cells by their predicted probability.

    Each overlay is a ``(cells, prob_feasible)`` pair, where *cells* is a
    2xN array of (row, col) grid indices. The probability is mapped onto
    the 0.1--0.9 span of the viridis colormap; everything else is fully
    transparent. *alpha* stays high so the colors read as viridis rather
    than washing out against the white free space.

    The returned image is transposed to match ``make_plotting_grid(grid.T)``:
    shape ``(grid_shape[1], grid_shape[0], 4)``, so grid cell (r, c) lands
    at pixel [c, r].
    """
    import matplotlib

    cmap = matplotlib.colormaps["viridis"]
    rgba = np.zeros((grid_shape[1], grid_shape[0], 4), dtype=np.float32)
    for cells, prob in overlays:
        cells = np.asarray(cells, dtype=int)
        if cells.size == 0:
            continue
        color = cmap(0.1 + 0.8 * float(np.clip(prob, 0.0, 1.0)))
        rgba[cells[1], cells[0], :3] = color[:3]
        rgba[cells[1], cells[0], 3] = alpha
    return rgba


def plot_grid_background(
    ax: Any,
    observed_grid: np.ndarray,
    true_grid: np.ndarray | None = None,
    *,
    translucent: bool = False,
) -> Any:
    """Render occupancy grid background with optional faded true-grid underlay.

    If *true_grid* is provided and *observed_grid* contains unobserved cells,
    the true grid is rendered at low alpha underneath a transparent-unobserved
    overlay of the observed grid.  Otherwise the observed grid is rendered
    directly.

    Both paths transpose the grid (``grid.T``) before rendering, matching the
    existing ``plot_grid`` convention.

    Set *translucent* when an aligned scene image sits at a lower zorder, to
    let it show through per cell class (see ``PHOTO_UNDERLAY_ALPHA``). The
    default renders fully opaque, exactly as before.

    Returns the ``AxesImage`` that was drawn.
    """
    alpha = None
    if translucent:
        alpha = make_plotting_grid_alpha(observed_grid.T, **PHOTO_UNDERLAY_ALPHA)

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

        return ax.imshow(composite, origin="upper", zorder=0, alpha=alpha)
    return ax.imshow(
        make_plotting_grid(observed_grid.T), origin="upper", zorder=0, alpha=alpha,
    )


def plot_grid(ax: Any, grid: np.ndarray) -> None:
    """Plot occupancy grid."""
    plotting_grid = make_plotting_grid(grid.T)
    ax.imshow(plotting_grid, origin="upper")
