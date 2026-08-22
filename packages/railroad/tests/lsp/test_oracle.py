"""Tests for oracle frontier labeling against the true map."""

from __future__ import annotations

import numpy as np
import pytest

from lsp.helpers import make_frontier

from railroad.experimental.unknown_search.types import Frontier
from railroad.lsp import (
    build_lookahead_grid,
    compute_oracle_frontier_labels,
    frontier_cells_hash,
    is_goal_observed,
    mask_grid_with_frontiers,
)
from railroad.navigation.constants import (
    COLLISION_VAL,
    FREE_VAL,
    UNOBSERVED_VAL,
)


def _t_junction() -> tuple[np.ndarray, np.ndarray, Frontier, Frontier, tuple[int, int]]:
    """Corridor with a dead-end branch.

    True map: horizontal corridor row 6 (cols 1..19) with a vertical
    dead-end branch up col 10 (rows 1..6). The robot has observed the
    middle of the corridor (cols 3..12) and the bottom of the branch
    (rows 4..5). Goal at (6, 19), beyond the east frontier.
    """
    true_grid = COLLISION_VAL * np.ones((13, 21))
    true_grid[6, 1:20] = FREE_VAL
    true_grid[1:7, 10] = FREE_VAL

    observed = UNOBSERVED_VAL * np.ones_like(true_grid)
    observed[6, 3:13] = FREE_VAL
    observed[4:6, 10] = FREE_VAL

    f_goal = make_frontier("f_goal", [(6, 12)])
    f_dead = make_frontier("f_dead", [(4, 10)])
    return true_grid, observed, f_goal, f_dead, (6, 19)


def test_mask_grid_with_frontiers() -> None:
    grid = FREE_VAL * np.ones((10, 10))
    f1 = make_frontier("f1", [(2, 2), (2, 3)])
    f2 = make_frontier("f2", [(7, 7)])
    original = grid.copy()

    masked = mask_grid_with_frontiers(grid, [f1, f2], keep=f1)
    assert masked[7, 7] == COLLISION_VAL
    assert masked[2, 2] == FREE_VAL and masked[2, 3] == FREE_VAL
    np.testing.assert_array_equal(grid, original)  # input not mutated

    # keep by id behaves identically
    masked_by_id = mask_grid_with_frontiers(grid, [f1, f2], keep="f1")
    np.testing.assert_array_equal(masked, masked_by_id)

    # no keep: everything masked
    masked_all = mask_grid_with_frontiers(grid, [f1, f2])
    assert masked_all[2, 2] == COLLISION_VAL
    assert masked_all[7, 7] == COLLISION_VAL


def test_build_lookahead_grid() -> None:
    true_grid = np.array([[FREE_VAL, COLLISION_VAL], [FREE_VAL, FREE_VAL]])
    observed = np.array([[FREE_VAL, UNOBSERVED_VAL], [UNOBSERVED_VAL, FREE_VAL]])
    lookahead = build_lookahead_grid(true_grid, observed)
    assert lookahead[0, 1] == COLLISION_VAL  # filled from true
    assert lookahead[1, 0] == FREE_VAL  # filled from true
    assert not np.any(lookahead == UNOBSERVED_VAL)


def test_is_goal_observed() -> None:
    observed = UNOBSERVED_VAL * np.ones((5, 5))
    observed[2, 2] = FREE_VAL
    assert is_goal_observed(observed, (2, 2))
    assert not is_goal_observed(observed, (0, 0))
    assert not is_goal_observed(observed, (99, 99))  # out of bounds


def test_frontier_cells_hash_order_invariant() -> None:
    f_a = make_frontier("a", [(1, 2), (3, 4), (5, 6)])
    f_b = make_frontier("b", [(5, 6), (1, 2), (3, 4)])
    f_c = make_frontier("c", [(1, 2), (3, 4)])
    assert frontier_cells_hash(f_a) == frontier_cells_hash(f_b)
    assert frontier_cells_hash(f_a) != frontier_cells_hash(f_c)


def test_t_junction_success_and_failure_labels() -> None:
    true_grid, observed, f_goal, f_dead, goal = _t_junction()

    labels = compute_oracle_frontier_labels(
        true_grid, observed, [f_goal, f_dead], goal
    )

    success = labels["f_goal"]
    assert success.prob_feasible == 1.0
    # Straight-line corridor distance (6,19) -> (6,12)
    assert success.success_cost is not None
    assert success.success_cost == pytest.approx(7.0, abs=0.5)
    assert success.optimistic_cost is not None
    assert success.optimistic_cost <= success.success_cost + 1e-6
    assert success.exploration_cost is None

    failure = labels["f_dead"]
    assert failure.prob_feasible == 0.0
    assert failure.success_cost is None
    # Farthest unseen cell of the branch is (1,10), 3 cells up from (4,10)
    assert failure.exploration_cost == pytest.approx(3.0, abs=0.5)


def test_exploration_cost_factor_scales() -> None:
    true_grid, observed, f_goal, f_dead, goal = _t_junction()
    labels_1 = compute_oracle_frontier_labels(
        true_grid, observed, [f_goal, f_dead], goal, exploration_cost_factor=1.0
    )
    labels_2 = compute_oracle_frontier_labels(
        true_grid, observed, [f_goal, f_dead], goal, exploration_cost_factor=2.0
    )
    assert labels_1["f_dead"].exploration_cost is not None
    assert labels_2["f_dead"].exploration_cost == pytest.approx(
        2.0 * labels_1["f_dead"].exploration_cost
    )


def test_exploration_cost_uses_only_own_component() -> None:
    """A dead-end frontier's cost covers its own unknown region, not others."""
    true_grid, observed, f_goal, f_dead, goal = _t_junction()
    labels = compute_oracle_frontier_labels(
        true_grid, observed, [f_goal, f_dead], goal
    )
    # The east unknown region (cols 13..19) is far larger than the branch;
    # if components leaked, the cost would be ~7+. The branch tops out at 3.
    exploration_cost = labels["f_dead"].exploration_cost
    assert exploration_cost is not None
    assert exploration_cost < 5.0


def test_optimistic_cost_below_true_cost_with_detour() -> None:
    """A wall hidden in unseen space makes the true path longer."""
    true_grid = COLLISION_VAL * np.ones((13, 21))
    true_grid[1:12, 1:20] = FREE_VAL
    true_grid[1:10, 15] = COLLISION_VAL  # wall forces a detour via the bottom

    observed = UNOBSERVED_VAL * np.ones_like(true_grid)
    observed[5:8, 1:5] = FREE_VAL

    frontier = make_frontier("f", [(5, 4), (6, 4), (7, 4)])
    goal = (6, 18)

    labels = compute_oracle_frontier_labels(true_grid, observed, [frontier], goal)
    label = labels["f"]
    assert label.prob_feasible == 1.0
    assert label.success_cost is not None and label.optimistic_cost is not None
    assert label.optimistic_cost < label.success_cost


def test_unreachable_goal_and_no_unknown_region() -> None:
    """Failure with no reachable unseen space yields a degenerate label."""
    true_grid = COLLISION_VAL * np.ones((10, 12))
    true_grid[6, 1:11] = FREE_VAL  # fully-observed corridor
    true_grid[2, 2:4] = FREE_VAL  # disconnected pocket holding the goal

    observed = UNOBSERVED_VAL * np.ones_like(true_grid)
    observed[6, 1:11] = FREE_VAL

    frontier = make_frontier("f", [(6, 10)])
    goal = (2, 2)

    labels = compute_oracle_frontier_labels(true_grid, observed, [frontier], goal)
    label = labels["f"]
    assert label.prob_feasible == 0.0
    assert label.success_cost is None
    assert label.exploration_cost is None


def test_accepts_mapping_input() -> None:
    true_grid, observed, f_goal, f_dead, goal = _t_junction()
    labels = compute_oracle_frontier_labels(
        true_grid, observed, {"f_goal": f_goal, "f_dead": f_dead}, goal
    )
    assert set(labels) == {"f_goal", "f_dead"}
    assert labels["f_goal"].prob_feasible == 1.0
    assert labels["f_dead"].prob_feasible == 0.0
