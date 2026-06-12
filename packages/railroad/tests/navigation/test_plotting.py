"""Tests for navigation plotting helpers."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from railroad.navigation.plotting import make_frontier_overlay_rgba

VIRIDIS = matplotlib.colormaps["viridis"]


def test_frontier_overlay_colors_cells_and_leaves_rest_transparent() -> None:
    # Two cells of one frontier: (row=2, col=4) and (row=3, col=4)
    cells = np.array([[2, 3], [4, 4]])
    rgba = make_frontier_overlay_rgba((10, 6), [(cells, 1.0)], alpha=0.8)

    # Transposed to match make_plotting_grid(grid.T)
    assert rgba.shape == (6, 10, 4)

    # prob=1.0 maps to viridis(0.9); cell (r, c) lands at pixel [c, r]
    expected = VIRIDIS(0.9)
    for r, c in [(2, 4), (3, 4)]:
        assert np.allclose(rgba[c, r, :3], expected[:3], atol=1e-6)
        assert rgba[c, r, 3] == pytest.approx(0.8)

    mask = np.ones((6, 10), dtype=bool)
    mask[4, 2] = mask[4, 3] = False
    assert np.all(rgba[mask] == 0.0)


def test_frontier_overlay_probability_mapping_and_clipping() -> None:
    rgba = make_frontier_overlay_rgba(
        (5, 5),
        [
            (np.array([[0], [0]]), 0.0),
            (np.array([[4], [4]]), 2.0),  # out-of-range prob is clipped
        ],
    )
    assert np.allclose(rgba[0, 0, :3], VIRIDIS(0.1)[:3], atol=1e-6)
    assert np.allclose(rgba[4, 4, :3], VIRIDIS(0.9)[:3], atol=1e-6)


def test_frontier_overlay_handles_empty_input() -> None:
    rgba = make_frontier_overlay_rgba((4, 5), [])
    assert rgba.shape == (5, 4, 4)
    assert np.all(rgba == 0.0)

    rgba = make_frontier_overlay_rgba((4, 5), [(np.empty((2, 0)), 0.5)])
    assert np.all(rgba == 0.0)
