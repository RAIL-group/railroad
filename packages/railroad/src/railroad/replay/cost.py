"""Pure cost-bound computation for offline replay.

Two lower bounds on what an alternative policy would have cost, computed
from recorded deployment data (see ``replay_design.md`` §6, §14.5):

- **optimistic** — ``min`` over the alternative's subgoal commitments of
  ``cost_accrued + optimistic_cost_to_goal(committed subgoal)``, where the
  cost-to-goal treats unobserved space as free (admissible: it can only
  *under*estimate the true cost, so the result is a valid lower bound).
- **simply-connected** ("pessimistic") — the total cost the alternative
  accrued, i.e. the explore-everything cost when no subgoal is a shortcut.

Everything here is a pure function of grids / costs: no environment, no
GL, no torch. The optimistic cost grid mirrors
:meth:`railroad.lsp.LSPEnvironmentMixin._optimistic_goal_cost_grid` but is
decoupled so it can be unit-tested in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from railroad.navigation.constants import FREE_VAL, UNOBSERVED_VAL
from railroad.navigation.pathing import compute_cost_grid_from_position

Cell = Tuple[int, int]


def optimistic_cost_grid_from_goal(
    recorded_grid: np.ndarray, goal_cell: Cell
) -> np.ndarray:
    """Dijkstra cost grid from *goal_cell* on the unseen-as-free map.

    All ``UNOBSERVED`` cells are treated as free (observed obstacles are
    respected), so the path length to any cell is an admissible lower
    bound on the true travel cost given the goal's known coordinates.
    Unreachable cells hold a non-finite cost.
    """
    grid = np.asarray(recorded_grid, dtype=float).copy()
    grid[grid == UNOBSERVED_VAL] = FREE_VAL
    cost_grid = compute_cost_grid_from_position(
        grid,
        start=[int(goal_cell[0]), int(goal_cell[1])],
        unknown_as_obstacle=True,
        only_return_cost_grid=True,
    )
    assert isinstance(cost_grid, np.ndarray)
    return cost_grid


def optimistic_cost_to_goal(
    recorded_grid: np.ndarray, point: Cell, goal_cell: Cell
) -> float:
    """Admissible lower-bound travel cost from *point* to *goal_cell*.

    Unseen-as-free path length (see :func:`optimistic_cost_grid_from_goal`).
    Returns ``inf`` when *point* is out of bounds or unreachable.
    """
    return _lookup(optimistic_cost_grid_from_goal(recorded_grid, goal_cell), point)


def _lookup(cost_grid: np.ndarray, point: Cell) -> float:
    row = int(round(float(point[0])))
    col = int(round(float(point[1])))
    if not (0 <= row < cost_grid.shape[0] and 0 <= col < cost_grid.shape[1]):
        return math.inf
    cost = float(cost_grid[row, col])
    return cost if math.isfinite(cost) else math.inf


@dataclass(frozen=True)
class Commit:
    """One alternative-policy commitment to an unrecorded subgoal.

    ``cost_accrued`` is the real cost paid to reach the commit point and
    ``optimistic_to_goal`` the admissible cost-to-goal from the committed
    subgoal — both in the deployment's cost unit (seconds / makespan), so the
    resulting bounds compare directly with the recorded ``actual_total_cost``.
    ``robot`` / ``frontier_signature`` are provenance (the signature is the
    replay-stable subgoal key, §6.1).
    """

    cost_accrued: float
    optimistic_to_goal: float
    robot: str = ""
    frontier_signature: str = ""


@dataclass(frozen=True)
class Bounds:
    """The two lower bounds an offline replay yields for one policy."""

    optimistic_lb: float
    simply_connected_lb: float


def accumulate_bounds(commits: Sequence[Commit], total_cost: float) -> Bounds:
    """Reduce a replay's commits + final cost to its two bounds.

    ``optimistic_lb`` is the cheapest "if this subgoal had led straight to the
    goal" total across all commits; ``simply_connected_lb`` is the total cost
    accrued. With **no commits** the candidate reached the goal touching only
    subgoals the deployment had already resolved — an *exact* replay — so the
    tightest lower bound is the realized cost itself, ``total_cost``. (Returning
    ``inf`` there would be a non-lower-bound that reads as "always worse" in
    selection.)
    """
    if commits:
        optimistic_lb = min(c.cost_accrued + c.optimistic_to_goal for c in commits)
    else:
        optimistic_lb = total_cost
    return Bounds(
        optimistic_lb=float(optimistic_lb),
        simply_connected_lb=float(total_cost),
    )
