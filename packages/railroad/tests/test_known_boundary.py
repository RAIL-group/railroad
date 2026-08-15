"""Outlining the region a robot has observed.

Drawn over a scene image, which renders ground truth everywhere including
where the robot has not looked. The outline says how far it has seen without
shading over the map.
"""

import numpy as np
import pytest

from railroad.navigation.constants import UNOBSERVED_VAL
from railroad.navigation.plotting import (
    FRONTIER_COLOR,
    KNOWN_WALL_COLOR,
    UNTRAVERSABLE_SHADE,
    make_known_boundary_rgba,
    make_untraversable_shade_rgba,
)


def _in_grid_orientation(grid: np.ndarray) -> np.ndarray:
    """The overlay, transposed back so it indexes like the grid."""
    return np.transpose(make_known_boundary_rgba(grid), (1, 0, 2))


def _room(n: int = 12) -> np.ndarray:
    """A walled room, entirely unobserved."""
    grid = np.full((n, n), UNOBSERVED_VAL)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1.0
    return grid


def _seen_patch() -> np.ndarray:
    grid = _room()
    grid[3:8, 3:8] = 0.0
    return grid


def _seen_to_the_map_edge() -> np.ndarray:
    """Observed space running off three sides of the array."""
    grid = np.full((12, 12), UNOBSERVED_VAL)
    grid[:, :5] = 0.0
    return grid


@pytest.mark.parametrize("grid", [_seen_patch(), _seen_to_the_map_edge()])
def test_the_outline_is_closed(grid):
    """A gap reads as "the robot knows what is through here" -- the opposite.

    The second case is the one that bit: observed space running to the edge of
    the array has no unknown cell beyond it, so nothing was drawn along that
    whole stretch and the outline leaked.

    Checked against an explicit 8-neighbour sweep, with off-grid counted as
    unknown, so this does not simply restate the implementation.
    """
    unknown = grid == UNOBSERVED_VAL
    n_rows, n_cols = grid.shape
    expected = np.zeros_like(unknown)
    for row in range(n_rows):
        for col in range(n_cols):
            if unknown[row, col]:
                continue
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    r, c = row + d_row, col + d_col
                    if not (0 <= r < n_rows and 0 <= c < n_cols) or unknown[r, c]:
                        expected[row, col] = True

    painted = _in_grid_orientation(grid)[:, :, 3] > 0
    np.testing.assert_array_equal(painted, expected)
    # An outline, not a wash: the map it is drawn over stays visible.
    assert not (painted & unknown).any()


def test_the_outline_says_why_it_ends():
    """Black where the robot has seen what stops it, colour where it has not.

    Told apart because they mean opposite things: one is a wall, the other is
    somewhere exploration could still go.
    """
    grid = _room()
    grid[2:6, 2:6] = 0.0
    grid[2:6, 6] = 1.0  # an observed wall, unknown beyond it

    rgba = _in_grid_orientation(grid)
    np.testing.assert_allclose(rgba[3, 6, :3], KNOWN_WALL_COLOR)
    np.testing.assert_allclose(rgba[3, 2, :3], FRONTIER_COLOR)

    # Nothing to delimit once everything has been seen.
    assert not _in_grid_orientation(np.zeros((8, 8)))[:, :, 3].any()


def test_the_outline_matches_the_planner_s_own_frontier_rule():
    """Frontier cells must be exactly what ``extract_frontiers`` calls one.

    The rule is repeated in plotting rather than imported, because plotting
    sits below the exploration package -- so pin the two together, or they
    drift and the plot quietly disagrees with the planner.
    """
    extract = pytest.importorskip(
        "railroad.experimental.unknown_search.frontiers"
    ).extract_frontiers

    grid = _room(20)
    grid[4:16, 4:12] = 0.0
    grid[4:16, 12] = 1.0

    rgba = _in_grid_orientation(grid)
    painted_as_frontier = (
        np.all(np.isclose(rgba[:, :, :3], FRONTIER_COLOR), axis=2) & (rgba[:, :, 3] > 0)
    )

    from_planner = np.zeros(grid.shape, dtype=bool)
    for frontier in extract(grid):
        from_planner[frontier.cells[0], frontier.cells[1]] = True

    np.testing.assert_array_equal(painted_as_frontier, from_planner)


def test_only_what_cannot_be_stood_on_is_shaded():
    """An image shows the room, not the map; furniture reads as floor from
    above unless the space the robot can move through is picked out."""
    grid = _room()
    grid[3:8, 3:8] = 0.0

    shade = np.transpose(make_untraversable_shade_rgba(grid), (1, 0, 2))[:, :, 3]
    free = grid == 0.0
    np.testing.assert_allclose(shade[free], 0.0)
    np.testing.assert_allclose(shade[~free], UNTRAVERSABLE_SHADE)
