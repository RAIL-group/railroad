"""Outlining the region a robot has observed.

Drawn over a scene image, which renders ground truth everywhere including
where the robot has not looked. The outline says how far it has seen without
shading over the map.

The property that matters is that the outline is *closed*: a gap reads as
"the robot knows what is through here", which is the opposite of the truth.
"""

import numpy as np
import pytest

from railroad.navigation.constants import UNOBSERVED_VAL
from railroad.navigation.plotting import (
    FRONTIER_COLOR,
    KNOWN_WALL_COLOR,
    make_known_boundary_rgba,
)


def _painted(grid: np.ndarray) -> np.ndarray:
    """Mask of outlined cells, back in grid orientation."""
    rgba = make_known_boundary_rgba(grid)
    return np.transpose(rgba, (1, 0, 2))[:, :, 3] > 0


def _should_be_outlined(grid: np.ndarray) -> np.ndarray:
    """Every observed cell touching the unknown, or the edge of the map.

    Computed the long way round -- an explicit 8-neighbour sweep with
    off-grid treated as unknown -- so it does not simply restate the
    implementation.
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
                    off_grid = not (0 <= r < n_rows and 0 <= c < n_cols)
                    if off_grid or unknown[r, c]:
                        expected[row, col] = True
    return expected


def _room(n: int = 12) -> np.ndarray:
    """A walled room, entirely unobserved."""
    grid = np.full((n, n), UNOBSERVED_VAL)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1.0
    return grid


class TestTheOutlineIsClosed:
    def test_it_covers_every_cell_that_borders_the_unknown(self):
        grid = _room()
        grid[3:8, 3:8] = 0.0  # a patch the robot has seen
        np.testing.assert_array_equal(_painted(grid), _should_be_outlined(grid))

    def test_observed_space_running_off_the_map_still_closes(self):
        """The bug: no unknown cell beyond the array, so nothing was drawn.

        A robot that has seen right up to the edge of its grid left the
        outline open along that whole stretch.
        """
        grid = np.full((12, 12), UNOBSERVED_VAL)
        grid[:, :5] = 0.0  # observed all the way to three map edges
        painted = _painted(grid)
        np.testing.assert_array_equal(painted, _should_be_outlined(grid))
        assert painted[0, :5].all(), "top edge is open"
        assert painted[-1, :5].all(), "bottom edge is open"
        assert painted[:, 0].all(), "left edge is open"

    def test_a_fully_observed_map_needs_no_outline(self):
        grid = np.zeros((8, 8))
        assert not _painted(grid).any()


class TestTheOutlineSaysWhyItEnds:
    def test_walls_are_black_and_frontiers_are_grey(self):
        grid = _room()
        # Observed free space, walled on one side and open on the other.
        grid[2:6, 2:6] = 0.0
        grid[2:6, 6] = 1.0  # an observed wall, unknown beyond it

        rgba = np.transpose(make_known_boundary_rgba(grid), (1, 0, 2))
        np.testing.assert_allclose(rgba[3, 6, :3], KNOWN_WALL_COLOR)
        np.testing.assert_allclose(rgba[3, 2, :3], FRONTIER_COLOR)

    def test_it_stays_an_outline_rather_than_a_wash(self):
        """The whole reason for drawing this instead of shading the unknown.

        Stated against the unknown region it delimits rather than as a fraction
        of the grid, which would just measure how much perimeter the fixture
        happens to have.
        """
        grid = _room(40)
        grid[5:35, 5:20] = 0.0
        painted = _painted(grid)
        unknown = grid == UNOBSERVED_VAL
        assert not (painted & unknown).any(), "painting over the unknown region"
        assert painted.sum() < 0.35 * unknown.sum()

    def test_the_frontier_grey_reads_against_a_dark_room(self):
        """Mid-grey, not near-black: it sits on a photo of unknown brightness."""
        assert 0.5 < FRONTIER_COLOR[0] < 0.85
        assert FRONTIER_COLOR > KNOWN_WALL_COLOR


def test_the_outline_matches_the_planner_s_own_frontier_rule():
    """Grey cells must be exactly what ``extract_frontiers`` calls a frontier.

    The rule is repeated in plotting rather than imported, because plotting
    sits below the exploration package -- so pin the two together.
    """
    extract = pytest.importorskip(
        "railroad.experimental.unknown_search.frontiers"
    ).extract_frontiers

    grid = _room(20)
    grid[4:16, 4:12] = 0.0
    grid[4:16, 12] = 1.0

    rgba = np.transpose(make_known_boundary_rgba(grid), (1, 0, 2))
    grey = np.all(np.isclose(rgba[:, :, :3], FRONTIER_COLOR), axis=2) & (rgba[:, :, 3] > 0)

    from_planner = np.zeros(grid.shape, dtype=bool)
    for frontier in extract(grid):
        from_planner[frontier.cells[0], frontier.cells[1]] = True

    np.testing.assert_array_equal(grey, from_planner)
