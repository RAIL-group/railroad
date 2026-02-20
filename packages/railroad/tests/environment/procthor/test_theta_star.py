"""Tests for Theta* any-angle path planning utilities."""

import numpy as np
import pytest

from railroad.environment.navigation.constants import OBSTACLE_THRESHOLD
from railroad.environment.navigation.pathing import (
    _line_of_sight,
    _supercover_line,
    _theta_star,
    build_traversal_costs,
    get_cost_and_path,
    get_cost_and_path_theta,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_grid() -> np.ndarray:
    """10x10 grid with no obstacles."""
    return np.zeros((10, 10))


@pytest.fixture
def wall_grid() -> np.ndarray:
    """10x10 grid with a vertical wall at column 5, gap at row 9."""
    grid = np.zeros((10, 10))
    grid[:9, 5] = 1.0  # wall with gap at (9, 5)
    return grid


@pytest.fixture
def blocked_grid() -> np.ndarray:
    """10x10 grid with a complete vertical wall — no passage."""
    grid = np.zeros((10, 10))
    grid[:, 5] = 1.0
    return grid


# ---------------------------------------------------------------------------
# build_traversal_costs
# ---------------------------------------------------------------------------


class TestBuildTraversalCosts:
    def test_obstacles_are_inf(self, empty_grid: np.ndarray) -> None:
        empty_grid[3, 3] = 1.0
        costs = build_traversal_costs(empty_grid)
        assert np.isinf(costs[3, 3])

    def test_free_cells_finite(self, empty_grid: np.ndarray) -> None:
        costs = build_traversal_costs(empty_grid)
        assert np.all(np.isfinite(costs))

    def test_no_soft_cost_gives_one(self, empty_grid: np.ndarray) -> None:
        costs = build_traversal_costs(empty_grid, use_soft_cost=False)
        np.testing.assert_array_equal(costs, np.ones((10, 10)))

    def test_soft_cost_increases_near_obstacles(self) -> None:
        grid = np.zeros((20, 20))
        grid[10, 10] = 1.0
        costs = build_traversal_costs(grid, use_soft_cost=True)
        # Cell adjacent to obstacle should cost more than a distant free cell
        assert costs[10, 9] > costs[0, 0]


# ---------------------------------------------------------------------------
# _supercover_line
# ---------------------------------------------------------------------------


class TestSupercoverLine:
    def test_horizontal(self) -> None:
        cells = _supercover_line(0, 0, 0, 4)
        assert cells[0] == (0, 0)
        assert cells[-1] == (0, 4)
        assert len(cells) == 5

    def test_vertical(self) -> None:
        cells = _supercover_line(0, 0, 4, 0)
        assert cells[0] == (0, 0)
        assert cells[-1] == (4, 0)
        assert len(cells) == 5

    def test_single_point(self) -> None:
        cells = _supercover_line(3, 7, 3, 7)
        assert cells == [(3, 7)]

    def test_diagonal_includes_neighbors(self) -> None:
        cells = _supercover_line(0, 0, 3, 3)
        # Supercover must include corner neighbors on diagonal steps
        assert len(cells) > 4  # more than just the 4 diagonal cells

    def test_contains_endpoints(self) -> None:
        for args in [(0, 0, 5, 3), (2, 7, 9, 1), (0, 0, 0, 0)]:
            cells = _supercover_line(*args)
            assert cells[0] == (args[0], args[1])
            assert cells[-1] == (args[2], args[3])


# ---------------------------------------------------------------------------
# _line_of_sight
# ---------------------------------------------------------------------------


class TestLineOfSight:
    def test_clear_path(self, empty_grid: np.ndarray) -> None:
        costs = build_traversal_costs(empty_grid, use_soft_cost=False)
        assert _line_of_sight(costs, (0, 0), (9, 9))

    def test_blocked_path(self, wall_grid: np.ndarray) -> None:
        costs = build_traversal_costs(wall_grid, use_soft_cost=False)
        assert not _line_of_sight(costs, (0, 0), (0, 9))

    def test_same_point(self, empty_grid: np.ndarray) -> None:
        costs = build_traversal_costs(empty_grid, use_soft_cost=False)
        assert _line_of_sight(costs, (5, 5), (5, 5))


# ---------------------------------------------------------------------------
# _theta_star
# ---------------------------------------------------------------------------


class TestThetaStar:
    def test_empty_grid_cost_approx_euclidean(
        self, empty_grid: np.ndarray
    ) -> None:
        costs = build_traversal_costs(empty_grid, use_soft_cost=False)
        cost, path = _theta_star(costs, (0, 0), (9, 9))
        euclidean = np.hypot(9, 9)
        # With unit cost, the path cost should be close to Euclidean
        assert cost == pytest.approx(euclidean, rel=0.1)

    def test_empty_grid_few_waypoints(self, empty_grid: np.ndarray) -> None:
        costs = build_traversal_costs(empty_grid, use_soft_cost=False)
        _, path = _theta_star(costs, (0, 0), (9, 9))
        # Any-angle should produce a near-straight-line (few waypoints)
        assert path.shape[0] == 2
        assert path.shape[1] <= 5  # much fewer than 10+ for grid-aligned

    def test_wall_with_gap(self, wall_grid: np.ndarray) -> None:
        costs = build_traversal_costs(wall_grid, use_soft_cost=False)
        cost, path = _theta_star(costs, (0, 0), (0, 9))
        assert np.isfinite(cost)
        assert path.shape[0] == 2
        assert path.shape[1] >= 2

    def test_start_equals_goal(self, empty_grid: np.ndarray) -> None:
        costs = build_traversal_costs(empty_grid, use_soft_cost=False)
        cost, path = _theta_star(costs, (4, 4), (4, 4))
        assert cost == 0.0
        np.testing.assert_array_equal(path, np.array([[4], [4]]))

    def test_unreachable_returns_inf(self, blocked_grid: np.ndarray) -> None:
        costs = build_traversal_costs(blocked_grid, use_soft_cost=False)
        cost, path = _theta_star(costs, (0, 0), (0, 9))
        assert np.isinf(cost)

    def test_path_avoids_obstacles(self, wall_grid: np.ndarray) -> None:
        costs = build_traversal_costs(wall_grid, use_soft_cost=False)
        _, path = _theta_star(costs, (0, 0), (0, 9))
        # No waypoint should be in an obstacle cell
        for i in range(path.shape[1]):
            r, c = path[0, i], path[1, i]
            assert wall_grid[r, c] < OBSTACLE_THRESHOLD

    @pytest.mark.parametrize(
        "start, goal",
        [
            ((0, 0), (9, 9)),
            ((9, 0), (0, 9)),
            ((0, 0), (0, 9)),
            ((0, 0), (9, 0)),
            ((5, 5), (5, 8)),
        ],
    )
    def test_various_endpoints(
        self,
        empty_grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> None:
        costs = build_traversal_costs(empty_grid, use_soft_cost=False)
        cost, path = _theta_star(costs, start, goal)
        assert np.isfinite(cost)
        assert path.shape[0] == 2
        # First and last waypoints match start/goal
        assert (path[0, 0], path[1, 0]) == start
        assert (path[0, -1], path[1, -1]) == goal


# ---------------------------------------------------------------------------
# get_cost_and_path_theta
# ---------------------------------------------------------------------------


class TestGetCostAndPathTheta:
    def test_return_type_and_shape(self, empty_grid: np.ndarray) -> None:
        cost, path = get_cost_and_path_theta(empty_grid, (0, 0), (9, 9))
        assert isinstance(cost, float)
        assert isinstance(path, np.ndarray)
        assert path.shape[0] == 2

    def test_start_in_obstacle_forced_free(self) -> None:
        grid = np.zeros((10, 10))
        grid[0, 0] = 1.0  # start is an obstacle
        cost, path = get_cost_and_path_theta(grid, (0, 0), (9, 9))
        assert np.isfinite(cost)
        assert path.shape[0] == 2


# ---------------------------------------------------------------------------
# get_cost_and_path (delegation)
# ---------------------------------------------------------------------------


class TestGetCostAndPathDelegation:
    def test_returns_finite_cost_and_path(self, empty_grid: np.ndarray) -> None:
        cost, path = get_cost_and_path(empty_grid, (0, 0), (9, 9))
        assert np.isfinite(cost)
        assert path.shape[0] == 2
        assert path.shape[1] >= 2
