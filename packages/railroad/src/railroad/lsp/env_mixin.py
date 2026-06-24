"""Environment mixin for LSP point-goal navigation.

Mix in *before* :class:`UnknownSpaceEnvironment`::

    class MyEnv(LSPEnvironmentMixin, UnknownSpaceEnvironment): ...

The subclass must set ``self._lsp_goal_cell`` and
``self._lsp_frontier_statistics`` (a
:class:`~railroad.lsp.FrontierStatisticsEstimator`) before calling
``super().__init__`` — the base constructor resolves operators via
:meth:`define_operators` and triggers the initial ``refresh_frontiers``,
both of which this mixin hooks. Operators (navigable moves, the gated
move-to-goal, lsp-explore parameterized by the estimator, and a no-op)
are owned by the environment; don't pass ``operators=``.

The goal is treated like an object whose spatial location is known in
advance: its name lives in ``objects_by_type["goal"]`` and its
coordinates are registered in the location registry from the start. Two
fluents track it (see :mod:`railroad.lsp.operators`): an lsp-explore
success marks it ``reachable`` (a route exists; the gated ``move-to-goal``
heads there), and directly observing its cell marks it ``revealed`` and
promotes it to a real ``location`` so the ordinary ``move`` takes over.
This mixin owns the ``revealed`` half: it watches the observed map and
fires :meth:`_reveal_goal_if_observed`. Exploring a frontier never
relocates the robot; the robot reaches the goal only by a real move.

The true-map oracle is needed only to resolve explore outcomes during
*simulated* execution and to label training data; planning works with
any frontier-statistics estimator. Where no oracle exists (deployment),
``oracle_labels`` is empty and explore outcomes fall back to the base
environment's probabilistic sampling — subclasses there should resolve
outcomes from real sensing instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import numpy as np

from railroad._bindings import Fluent, GroundedEffect
from railroad.core import Operator
from railroad.experimental.unknown_search.operators import (
    construct_move_navigable_operator,
)
from railroad.navigation.constants import FREE_VAL, UNOBSERVED_VAL
from railroad.navigation.pathing import compute_cost_grid_from_position
from railroad.operators import construct_no_op_operator

from .frontier_statistics import FrontierStatisticsEstimator
from .operators import (
    construct_lsp_explore_operator,
    construct_move_to_goal_operator,
)
from .oracle import compute_oracle_frontier_labels, is_goal_observed
from .types import OracleFrontierLabel

_UNREACHABLE = 1e8

if TYPE_CHECKING:
    from railroad.experimental.unknown_search import UnknownSpaceEnvironment

    _Base = UnknownSpaceEnvironment
else:
    _Base = object


class LSPEnvironmentMixin(_Base):
    """Operators, oracle labels, goal revelation, and explore execution."""

    # NOTE: the symbolic goal name must not end in "_loc" — the base
    # environment reserves that suffix for transient robot anchors and
    # filters out move actions targeting such names.
    _lsp_goal_cell: Tuple[int, int]
    _lsp_frontier_statistics: FrontierStatisticsEstimator
    _lsp_goal_name: str = "goal"
    _lsp_exploration_cost_factor: float = 1.0

    @property
    def frontier_statistics(self) -> FrontierStatisticsEstimator:
        """The estimator parameterizing lsp-explore actions."""
        return self._lsp_frontier_statistics

    def define_operators(self) -> List[Operator]:
        """LSP point-goal operators; override to customize the set."""
        return [
            construct_move_navigable_operator(self.estimate_move_time_safe),
            construct_move_to_goal_operator(self.estimate_goal_move_time),
            construct_lsp_explore_operator(
                self._lsp_frontier_statistics,
                self.optimistic_goal_cost,
                speed_cells_per_sec=self._config.speed_cells_per_sec,
                goal_name=self._lsp_goal_name,
            ),
            construct_no_op_operator(no_op_time=300.0, extra_cost=100.0),
        ]

    @property
    def frontier_probability_overlays(self) -> List[Tuple[np.ndarray, float]]:
        """(cells, prob_feasible) per current frontier, for visualization.

        Reports what the *planner* believes — whatever the active
        estimator predicts (learned, oracle, or fixed prior). The
        dashboard duck-types this to color frontier cells in plots and
        videos. Cells are copies, safe to keep across refreshes.
        """
        robot = next(iter(sorted(self._objects_by_type.get("robot", set()))), "")
        return [
            (
                np.array(frontier.cells, dtype=int, copy=True),
                float(
                    self._lsp_frontier_statistics.get(robot, frontier_id)
                    .prob_feasible
                ),
            )
            for frontier_id, frontier in self.frontiers.items()
        ]

    @property
    def goal_cell(self) -> Tuple[int, int]:
        return self._lsp_goal_cell

    @property
    def goal_name(self) -> str:
        return self._lsp_goal_name

    @property
    def oracle_available(self) -> bool:
        """Whether a true map exists to compute oracle labels from."""
        return getattr(self, "_true_grid", None) is not None

    @property
    def oracle_labels(self) -> Dict[str, OracleFrontierLabel]:
        """Current true-map frontier labels (cached per grid generation).

        Empty when no true map is available (deployment).
        """
        if not self.oracle_available:
            return {}
        labeled_generation = getattr(self, "_lsp_labeled_generation", None)
        if labeled_generation != (self._grid_generation, len(self._frontiers)):
            self._lsp_labels = compute_oracle_frontier_labels(
                self._true_grid,
                self._observed_grid,
                self._frontiers,
                self._lsp_goal_cell,
                exploration_cost_factor=self._lsp_exploration_cost_factor,
            )
            self._lsp_labeled_generation = (
                self._grid_generation, len(self._frontiers)
            )
        return self._lsp_labels

    @property
    def goal_observed(self) -> bool:
        return is_goal_observed(self._observed_grid, self._lsp_goal_cell)

    # ------------------------------------------------------------------
    # Goal move-time estimation
    # ------------------------------------------------------------------

    def _optimistic_goal_cost_grid(self) -> np.ndarray:
        """Cost grid from the goal on the unseen-as-free observed map.

        The goal's location is known in advance, so even before a real
        path is observed we can lower-bound the travel cost to it by the
        path length on the map where all unseen space is unoccupied
        (observed obstacles respected). Cached per grid generation.
        """
        if getattr(self, "_lsp_opt_cost_generation", None) != self._grid_generation:
            grid = self._observed_grid.copy()
            grid[grid == UNOBSERVED_VAL] = FREE_VAL
            cost_grid = compute_cost_grid_from_position(
                grid,
                start=[self._lsp_goal_cell[0], self._lsp_goal_cell[1]],
                unknown_as_obstacle=True,
                only_return_cost_grid=True,
            )
            assert isinstance(cost_grid, np.ndarray)
            self._lsp_opt_cost_grid = cost_grid
            self._lsp_opt_cost_generation = self._grid_generation
        return self._lsp_opt_cost_grid

    def _optimistic_goal_cost_cells(self, loc_from: str) -> float | None:
        """Optimistic (unseen-as-free) path cost in cells to the goal.

        Lower bound on the travel cost from *loc_from* to the goal given
        the goal's known coordinates: the path length on the map where all
        unseen space is treated as free (observed obstacles respected).
        Returns ``None`` when no such path exists or *loc_from* has no
        registered coordinates.
        """
        registry = self._location_registry
        if registry is None:
            return None
        coords_from = registry.get(loc_from)
        if coords_from is None:
            return None
        cost_grid = self._optimistic_goal_cost_grid()
        row = int(round(float(coords_from[0])))
        col = int(round(float(coords_from[1])))
        if not (0 <= row < cost_grid.shape[0] and 0 <= col < cost_grid.shape[1]):
            return None
        cost = float(cost_grid[row, col])
        if np.isfinite(cost) and cost < _UNREACHABLE:
            return cost
        return None

    def optimistic_goal_cost(self, robot: str, frontier_id: str) -> float:
        """Optimistic lower-bound travel cost (in cells) frontier → goal.

        Supplied to ``lsp-explore`` so a frontier's ``delta_success_cost``
        (the *extra* cost beyond this bound) can be turned back into a full
        success cost. Falls back to the generic safe estimate (converted to
        a cost) when no optimistic path is known.
        """
        cost = self._optimistic_goal_cost_cells(frontier_id)
        if cost is not None:
            return cost
        speed = max(self._config.speed_cells_per_sec, 1e-6)
        return self.estimate_move_time_safe(robot, frontier_id, self._lsp_goal_name) * speed

    def estimate_goal_move_time(
        self, robot: str, loc_from: str, loc_to: str
    ) -> float:
        """Move-time estimate for moves to the (known-location) goal.

        Uses the real observed-map path time when one exists; otherwise
        the optimistic (unseen-as-free) path time — the lower bound on
        reaching the goal given its known location. Falls back to the
        generic safe estimate for non-goal targets or degenerate maps.
        """
        if loc_to != self._lsp_goal_name:
            return self.estimate_move_time_safe(robot, loc_from, loc_to)

        t = self.estimate_move_time(robot, loc_from, loc_to)
        if np.isfinite(t):
            return t

        cost = self._optimistic_goal_cost_cells(loc_from)
        if cost is not None:
            speed = self._config.speed_cells_per_sec
            return cost / max(speed, 1e-6)
        return self.estimate_move_time_safe(robot, loc_from, loc_to)

    # ------------------------------------------------------------------
    # Frontier refresh: goal setup, revelation, and data hook
    # ------------------------------------------------------------------

    def refresh_frontiers(self) -> None:
        super().refresh_frontiers()
        self._ensure_goal_setup()
        self._reveal_goal_if_observed()
        self._lsp_frontier_statistics.refresh(self)
        self._after_lsp_refresh()

    def _ensure_goal_setup(self) -> None:
        """Register the goal object and its (known) coordinates."""
        self._objects_by_type.setdefault("goal", set()).add(self._lsp_goal_name)
        registry = self._location_registry
        if registry is not None and registry.get(self._lsp_goal_name) is None:
            registry.register(
                self._lsp_goal_name,
                np.array(self._lsp_goal_cell, dtype=float),
            )

    def _reveal_goal_if_observed(self) -> None:
        """Reveal the goal and stabilize it as a destination once seen.

        Adding the goal to the base locations keeps the robot's
        ``at <robot> goal`` fluent from being remapped away when it
        arrives (it also makes the goal a regular move target, which is
        fine post-revelation).
        """
        if self._lsp_goal_name in self._base_locations:
            return
        if not self.goal_observed:
            return
        self.register_discovered_location(self._lsp_goal_name, self._lsp_goal_cell)
        self._fluents.add(Fluent("revealed", self._lsp_goal_name))

    def _after_lsp_refresh(self) -> None:
        """Hook called after frontiers, labels, and goal state update."""

    # ------------------------------------------------------------------
    # Execution: resolve lsp-explore outcomes from the oracle
    # ------------------------------------------------------------------

    def _match_lsp_explore_branches(
        self, effect: GroundedEffect
    ) -> tuple[str, list[GroundedEffect], list[GroundedEffect]] | None:
        """Return (frontier_id, success_effects, failure_effects) if *effect*
        is an lsp-explore branch point, else None."""
        branches = effect.prob_effects
        if len(branches) != 2:
            return None

        success_effects = None
        failure_effects = None
        frontier_id = None
        for _, branch_effects in branches:
            is_success = any(
                f.name == "reachable" and not f.negated
                and f.args and f.args[0] == self._lsp_goal_name
                for eff in branch_effects
                for f in eff.resulting_fluents
            )
            if is_success:
                success_effects = list(branch_effects)
                for eff in branch_effects:
                    for f in eff.resulting_fluents:
                        if f.name == "explored" and not f.negated and f.args:
                            frontier_id = f.args[0]
            else:
                failure_effects = list(branch_effects)

        if success_effects is None or failure_effects is None or frontier_id is None:
            return None
        return frontier_id, success_effects, failure_effects

    def resolve_probabilistic_effect(
        self,
        effect: GroundedEffect,
        current_fluents: Set[Fluent],
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        """Resolve lsp-explore outcomes from the oracle label; defer otherwise.

        The planner may have used any property provider, but execution is
        always grounded in the true map: the success branch (revealing
        the goal) fires iff a path through the frontier reaches the goal.
        """
        if effect.is_probabilistic:
            match = self._match_lsp_explore_branches(effect)
            if match is not None:
                frontier_id, success_effects, failure_effects = match
                label = self.oracle_labels.get(frontier_id)
                if label is not None:
                    if label.prob_feasible >= 0.5:
                        return success_effects, current_fluents
                    return failure_effects, current_fluents
        return super().resolve_probabilistic_effect(effect, current_fluents)
