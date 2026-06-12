"""Oracle frontier labeling against the true map for point-goal navigation.

For each frontier, every *other* frontier is masked (its cells set to
collision), sealing all other exits from observed space. A grid search on
the resulting "lookahead" grid (unobserved cells filled with true values)
then decides whether a path through that frontier reaches the goal, and
with what cost.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Mapping, Sequence, cast

import numpy as np
import scipy.ndimage

from railroad.experimental.unknown_search.types import Frontier
from railroad.navigation.constants import (
    COLLISION_VAL,
    FREE_VAL,
    OBSTACLE_THRESHOLD,
    UNOBSERVED_VAL,
)
from railroad.navigation.pathing import compute_cost_grid_from_position

from .types import OracleFrontierLabel

_UNREACHABLE = 1e8


def frontier_cells_hash(frontier: Frontier) -> str:
    """Deterministic hash of a frontier's cell set."""
    cells = frontier.cells
    order = np.lexsort((cells[1, :], cells[0, :]))
    return hashlib.sha1(
        np.ascontiguousarray(cells[:, order]).tobytes()
    ).hexdigest()


def _resolve_keep_id(keep: Frontier | str | None) -> str | None:
    if keep is None:
        return None
    return keep if isinstance(keep, str) else keep.id


def mask_grid_with_frontiers(
    grid: np.ndarray,
    frontiers: Iterable[Frontier],
    keep: Frontier | str | None = None,
) -> np.ndarray:
    """Return a copy of *grid* with all frontiers' cells set to collision.

    The frontier matching *keep* (a Frontier or frontier id) is left
    untouched, so it remains the only passage between observed space and
    the rest of the map.
    """
    keep_id = _resolve_keep_id(keep)
    masked = grid.copy()
    for frontier in frontiers:
        if frontier.id == keep_id:
            continue
        masked[frontier.cells[0, :], frontier.cells[1, :]] = COLLISION_VAL
    return masked


def build_lookahead_grid(
    true_grid: np.ndarray,
    observed_grid: np.ndarray,
) -> np.ndarray:
    """Observed cells keep their observed values; unobserved take true values."""
    lookahead = observed_grid.copy()
    unobserved = observed_grid == UNOBSERVED_VAL
    lookahead[unobserved] = true_grid[unobserved]
    return lookahead


def is_goal_observed(
    observed_grid: np.ndarray,
    goal_cell: tuple[int, int],
) -> bool:
    """Whether the goal cell has been observed."""
    row, col = int(goal_cell[0]), int(goal_cell[1])
    if not (0 <= row < observed_grid.shape[0] and 0 <= col < observed_grid.shape[1]):
        return False
    return float(observed_grid[row, col]) != UNOBSERVED_VAL


def _finite_costs(costs: "np.typing.ArrayLike") -> np.ndarray:
    flat = np.asarray(costs, dtype=float).ravel()
    return flat[np.isfinite(flat) & (flat < _UNREACHABLE)]


def _min_finite_cost_at_cells(
    cost_grid: np.ndarray, cells: np.ndarray
) -> float | None:
    finite = _finite_costs(cost_grid[cells[0, :], cells[1, :]])
    if finite.size == 0:
        return None
    return float(finite.min())


def compute_oracle_frontier_labels(
    true_grid: np.ndarray,
    observed_grid: np.ndarray,
    frontiers: Mapping[str, Frontier] | Sequence[Frontier],
    goal_cell: tuple[int, int],
    *,
    exploration_cost_factor: float = 1.0,
) -> dict[str, OracleFrontierLabel]:
    """Label every frontier by searching the true map through it alone.

    Per frontier: mask all other frontiers in the lookahead grid and run
    Dijkstra from the goal. A finite cost at the frontier means the
    frontier leads to the goal (``success_cost`` is that path cost, and
    ``optimistic_cost`` repeats the search with unobserved space assumed
    free). Otherwise the frontier fails, and ``exploration_cost`` is the
    distance to the farthest reachable cell of the unknown region behind
    it (its connected component), scaled by *exploration_cost_factor*.
    """
    if isinstance(frontiers, Mapping):
        frontier_list = list(cast(Mapping[str, Frontier], frontiers).values())
    else:
        frontier_list = list(frontiers)
    goal = (int(goal_cell[0]), int(goal_cell[1]))

    # Unknown free space, partitioned into connected components so each
    # failure frontier's exploration cost only covers its own region.
    unknown_free = (
        (observed_grid == UNOBSERVED_VAL) & (true_grid < OBSTACLE_THRESHOLD)
        & (true_grid >= FREE_VAL)
    )
    unk_labels, _ = scipy.ndimage.label(unknown_free, structure=np.ones((3, 3)))

    labels: dict[str, OracleFrontierLabel] = {}
    for frontier in frontier_list:
        masked = mask_grid_with_frontiers(
            build_lookahead_grid(true_grid, observed_grid),
            frontier_list,
            keep=frontier,
        )

        goal_cost_grid = compute_cost_grid_from_position(
            masked,
            start=[goal[0], goal[1]],
            unknown_as_obstacle=True,
            only_return_cost_grid=True,
        )
        assert isinstance(goal_cost_grid, np.ndarray)
        success_cost = _min_finite_cost_at_cells(goal_cost_grid, frontier.cells)

        if success_cost is not None:
            optimistic_grid = mask_grid_with_frontiers(
                observed_grid, frontier_list, keep=frontier
            )
            optimistic_grid[optimistic_grid == UNOBSERVED_VAL] = FREE_VAL
            optimistic_cost_grid = compute_cost_grid_from_position(
                optimistic_grid,
                start=[goal[0], goal[1]],
                unknown_as_obstacle=True,
                only_return_cost_grid=True,
            )
            assert isinstance(optimistic_cost_grid, np.ndarray)
            optimistic_cost = _min_finite_cost_at_cells(
                optimistic_cost_grid, frontier.cells
            )
            labels[frontier.id] = OracleFrontierLabel(
                frontier_id=frontier.id,
                prob_feasible=1.0,
                success_cost=success_cost,
                optimistic_cost=optimistic_cost,
                exploration_cost=None,
                cells_hash=frontier_cells_hash(frontier),
            )
            continue

        # Failure: cost to the farthest reachable cell of the unknown
        # component(s) adjacent to this frontier.
        frontier_mask = np.zeros(true_grid.shape, dtype=bool)
        frontier_mask[frontier.cells[0, :], frontier.cells[1, :]] = True
        adjacent = scipy.ndimage.binary_dilation(
            frontier_mask, structure=np.ones((3, 3))
        )
        component_ids = np.unique(unk_labels[adjacent & (unk_labels > 0)])

        exploration_cost: float | None = None
        if component_ids.size > 0:
            frontier_cost_grid = compute_cost_grid_from_position(
                masked,
                start=frontier.cells,
                unknown_as_obstacle=True,
                only_return_cost_grid=True,
            )
            assert isinstance(frontier_cost_grid, np.ndarray)
            in_region = np.asarray(
                np.isin(unk_labels, component_ids), dtype=bool
            )
            finite = _finite_costs(
                cast(np.ndarray, frontier_cost_grid[in_region])
            )
            if finite.size > 0:
                exploration_cost = exploration_cost_factor * float(finite.max())

        labels[frontier.id] = OracleFrontierLabel(
            frontier_id=frontier.id,
            prob_feasible=0.0,
            success_cost=None,
            optimistic_cost=None,
            exploration_cost=exploration_cost,
            cells_hash=frontier_cells_hash(frontier),
        )

    return labels
