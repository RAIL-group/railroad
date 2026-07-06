"""Tests for the pure offline-replay cost bounds (``railroad.replay.cost``).

These are the L0 anchor: no environment, no GL, no torch — just grids and
hand-verified numbers. See replay_design.md §14.5.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from railroad.navigation.constants import UNOBSERVED_VAL
from railroad.navigation.pathing import compute_cost_grid_from_position
from railroad.replay.cost import (
    Bounds,
    Commit,
    accumulate_bounds,
    optimistic_cost_grid_from_goal,
    optimistic_cost_to_goal,
)


# --------------------------------------------------------------------------
# optimistic_cost_to_goal / optimistic_cost_grid_from_goal
# --------------------------------------------------------------------------


def test_straight_line_cost_is_distance(parse_grid) -> None:
    """On an open corridor the cost is the geometric path length in cells."""
    grid, markers = parse_grid("G....P")
    goal = markers["G"][0]
    point = markers["P"][0]
    assert optimistic_cost_to_goal(grid, point, goal) == pytest.approx(5.0)


def test_goal_cell_has_zero_cost(parse_grid) -> None:
    grid, markers = parse_grid(
        """
        ....
        .G..
        ....
        """
    )
    goal = markers["G"][0]
    cost_grid = optimistic_cost_grid_from_goal(grid, goal)
    assert cost_grid[goal[0], goal[1]] == pytest.approx(0.0)


def test_routes_through_unobserved_space(parse_grid) -> None:
    """Unseen-as-free opens a path that observed-only space blocks.

    The collision wall in column 1 separates G from P; the only gap is an
    *unobserved* cell. Treating unobserved as free (the optimistic bound)
    must find a finite path, whereas treating it as an obstacle must not.
    """
    grid, markers = parse_grid(
        """
        G#P
        .#.
        .?.
        """
    )
    goal = markers["G"][0]
    point = markers["P"][0]

    # Observed-only: the '?' gap is impassable -> no path -> inf at P.
    observed_only = compute_cost_grid_from_position(
        grid,
        start=[goal[0], goal[1]],
        unknown_as_obstacle=True,
        only_return_cost_grid=True,
    )
    assert isinstance(observed_only, np.ndarray)
    assert not math.isfinite(float(observed_only[point[0], point[1]]))

    # Optimistic (unseen-as-free): the gap opens -> finite path.
    assert math.isfinite(optimistic_cost_to_goal(grid, point, goal))


def test_unreachable_point_returns_inf(parse_grid) -> None:
    """A point fully boxed in by collisions is unreachable even optimistically."""
    grid, markers = parse_grid(
        """
        G......
        .......
        ..###..
        ..#P#..
        ..###..
        """
    )
    goal = markers["G"][0]
    point = markers["P"][0]
    assert optimistic_cost_to_goal(grid, point, goal) == math.inf


def test_out_of_bounds_point_returns_inf(parse_grid) -> None:
    grid, markers = parse_grid("G....")
    goal = markers["G"][0]
    assert optimistic_cost_to_goal(grid, (10, 10), goal) == math.inf


def test_cost_grid_is_deterministic(parse_grid) -> None:
    grid, markers = parse_grid(
        """
        G..#..
        ...#..
        ......
        """
    )
    goal = markers["G"][0]
    first = optimistic_cost_grid_from_goal(grid, goal)
    second = optimistic_cost_grid_from_goal(grid, goal)
    np.testing.assert_array_equal(first, second)


def test_unobserved_cells_are_not_mutated_in_input(parse_grid) -> None:
    """The recorded grid passed in is not modified by the cost computation."""
    grid, markers = parse_grid(
        """
        G.?
        ..?
        """
    )
    goal = markers["G"][0]
    before = grid.copy()
    optimistic_cost_grid_from_goal(grid, goal)
    np.testing.assert_array_equal(grid, before)
    assert np.any(grid == UNOBSERVED_VAL)


# --------------------------------------------------------------------------
# accumulate_bounds
# --------------------------------------------------------------------------


def test_accumulate_bounds_picks_minimum_commit() -> None:
    commits = [
        Commit(cost_accrued=10.0, optimistic_to_goal=5.0),
        Commit(cost_accrued=20.0, optimistic_to_goal=2.0),  # 22
        Commit(cost_accrued=3.0, optimistic_to_goal=4.0),  # 7 -> min
    ]
    bounds = accumulate_bounds(commits, total_cost=30.0)
    assert bounds == Bounds(optimistic_lb=7.0, simply_connected_lb=30.0)


def test_accumulate_bounds_empty_is_exact_cost() -> None:
    """No commits ⇒ exact replay ⇒ optimistic bound is the realized cost, not
    inf (inf would be a non-lower-bound, reading as 'always worse' in selection)."""
    bounds = accumulate_bounds([], total_cost=42.0)
    assert bounds.optimistic_lb == 42.0
    assert bounds.simply_connected_lb == 42.0


def test_accumulate_bounds_tolerates_inf_commit() -> None:
    """An unreachable commit must not poison a finite minimum."""
    commits = [
        Commit(cost_accrued=5.0, optimistic_to_goal=math.inf),
        Commit(cost_accrued=8.0, optimistic_to_goal=3.0),  # 11 -> min
    ]
    bounds = accumulate_bounds(commits, total_cost=15.0)
    assert bounds.optimistic_lb == pytest.approx(11.0)


def test_accumulate_bounds_all_inf() -> None:
    commits = [Commit(cost_accrued=5.0, optimistic_to_goal=math.inf)]
    bounds = accumulate_bounds(commits, total_cost=15.0)
    assert bounds.optimistic_lb == math.inf
    assert bounds.simply_connected_lb == 15.0


def test_optimistic_never_exceeds_simply_connected_when_at_goal(parse_grid) -> None:
    """Sanity: a commit whose cost-to-goal is 0 bounds at its accrued cost."""
    commits = [Commit(cost_accrued=12.0, optimistic_to_goal=0.0)]
    bounds = accumulate_bounds(commits, total_cost=12.0)
    assert bounds.optimistic_lb <= bounds.simply_connected_lb
