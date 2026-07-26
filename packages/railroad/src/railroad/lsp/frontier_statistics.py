"""Frontier-statistics estimators: where explore-action parameters come from.

Every frontier-exploration action is parameterized by
:class:`FrontierStatistics` — the probability the frontier leads to the
goal and the success/exploration costs. Three approaches produce them:

- **Oracle** (:class:`OracleFrontierStatistics`): exact statistics from
  the environment's true-map labels. Simulation only — used for an
  idealized planning baseline and to label training data.
- **Fixed prior** (:class:`FixedPriorFrontierStatistics`): constants for
  every frontier. Needs nothing but the constructor arguments, so it
  always works, including in deployment.
- **Learned** (:class:`LearnedFrontierStatistics`): a model predicts the
  statistics from what the robot has *seen* of each frontier — the same
  panorama-based :class:`FrontierObservation` the training data stores.

Estimators are refreshed by the environment whenever frontiers change
(:meth:`FrontierStatisticsEstimator.refresh`), and queried per frontier
when actions are grounded (:meth:`FrontierStatisticsEstimator.get`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from railroad.experimental.unknown_search.types import Frontier

from .oracle import compute_oracle_frontier_labels
from .types import FrontierObservation, FrontierStatistics, OracleFrontierLabel
from .views import compute_frontier_views

DEFAULT_FRONTIER_STATISTICS = FrontierStatistics(
    prob_feasible=0.5, delta_success_cost=0.0, exploration_cost=10.0
)


class FrontierStatisticsEnvironment(Protocol):
    """What an estimator may read from the environment on refresh.

    Deliberately only three things, and none of them is ground truth: an
    estimator sees what the robot has *observed*, never what is actually there.
    A policy that knows more than the robot (the oracle) must carry that
    knowledge itself, which is what keeps "belief" and "truth" separable — and
    what lets one estimator serve both a live deployment and an offline replay
    arena, whose notions of a "true grid" differ.

    :class:`~railroad.lsp.LSPEnvironmentMixin` satisfies this, as do the replay
    arenas.
    """

    @property
    def frontiers(self) -> Mapping[str, Frontier]: ...

    @property
    def goal_cell(self) -> Tuple[int, int]: ...

    @property
    def observed_grid(self) -> "np.ndarray": ...


class FrontierStatisticsEstimator(ABC):
    """Maps each frontier to its planner-facing statistics."""

    def refresh(self, environment: FrontierStatisticsEnvironment) -> None:
        """Update internal state from the environment.

        Called by the environment whenever frontiers change, before any
        action grounding queries :meth:`get`. The default does nothing.
        """

    @abstractmethod
    def get(self, robot: str, frontier_id: str) -> FrontierStatistics:
        """Statistics of exploring *frontier_id* with *robot*."""


class OracleFrontierStatistics(FrontierStatisticsEstimator):
    """Exact statistics from true-map oracle labels.

    **The estimator carries its own ground truth.** It never asks the
    environment for it — the environment supplies only what it has currently
    *observed*, plus (unless overridden) the goal cell.

    That separation is deliberate. An environment's own "true grid" means
    *whatever it simulates against*, which is not always the real world: offline
    replay deliberately hands its environment a **confinement grid** (unobserved
    cells turned into walls) so the replayed robot cannot wander into space the
    deployment never saw. An oracle labelling against that would call every
    frontier dead while still looking like an oracle. Requiring the true map up
    front makes that failure impossible to write.

    It also matches what an oracle *is*: a black box that knows the answer. How
    it knows is its own business — in offline replay it may consult ground truth
    the arena cannot, because the replayed cost accounting reads only recorded
    data regardless of what drove the robot.

    Frontiers without a label (e.g. degenerate ones) fall back to *default*.
    """

    def __init__(
        self,
        true_grid: "np.ndarray",
        *,
        goal_cell: Optional[Tuple[int, int]] = None,
        default: FrontierStatistics = DEFAULT_FRONTIER_STATISTICS,
        exploration_cost_factor: float = 1.0,
    ) -> None:
        self._default = default
        self._labels: Dict[str, OracleFrontierLabel] = {}
        self._true_grid = np.asarray(true_grid, dtype=float)
        self._goal_cell = (
            None if goal_cell is None else (int(goal_cell[0]), int(goal_cell[1]))
        )
        self._exploration_cost_factor = float(exploration_cost_factor)

    def refresh(self, environment: FrontierStatisticsEnvironment) -> None:
        goal_cell = (
            self._goal_cell if self._goal_cell is not None else environment.goal_cell
        )
        self._labels = compute_oracle_frontier_labels(
            self._true_grid,
            environment.observed_grid,
            environment.frontiers,
            goal_cell,
            exploration_cost_factor=self._exploration_cost_factor,
        )

    def get(self, robot: str, frontier_id: str) -> FrontierStatistics:
        return self.statistics_from_label(
            self._labels.get(frontier_id), self._default
        )

    @staticmethod
    def statistics_from_label(
        label: Optional[OracleFrontierLabel],
        default: FrontierStatistics = DEFAULT_FRONTIER_STATISTICS,
    ) -> FrontierStatistics:
        """Convert one oracle label into planner-facing statistics.

        The two branches carry different information, so each fills in the field
        the other leaves undefined from *default*:

        - **feasible** (the frontier reaches the goal): ``delta_success_cost`` is
          the *extra* cost over the optimistic estimate, which is what
          ``lsp-explore`` adds back to its own optimistic bound. Exploration cost
          is undefined.
        - **infeasible**: only ``exploration_cost`` is known.

        A missing label, or one whose branch-specific cost is ``None``
        (degenerate frontiers), falls back to *default* entirely.

        Static so the mapping stays directly testable without standing up an
        environment — it is pure, and it is the only part of this estimator with
        non-obvious arithmetic.
        """
        if label is None:
            return default
        if label.prob_feasible >= 0.5:
            if label.success_cost is None:
                return default
            optimistic = (
                label.optimistic_cost
                if label.optimistic_cost is not None
                else label.success_cost
            )
            return FrontierStatistics(
                prob_feasible=label.prob_feasible,
                delta_success_cost=max(0.0, label.success_cost - optimistic),
                exploration_cost=default.exploration_cost,
            )
        if label.exploration_cost is None:
            return default
        return FrontierStatistics(
            prob_feasible=label.prob_feasible,
            delta_success_cost=default.delta_success_cost,
            exploration_cost=label.exploration_cost,
        )


class FixedPriorFrontierStatistics(FrontierStatisticsEstimator):
    """The same fixed statistics for every frontier.

    The default prior is optimistic — every frontier looks promising —
    which makes planning greedily goal-directed.
    """

    def __init__(
        self,
        prob_feasible: float = 0.8,
        delta_success_cost: float = 0.0,
        exploration_cost: float = 10.0,
    ) -> None:
        self._statistics = FrontierStatistics(
            prob_feasible=prob_feasible,
            delta_success_cost=delta_success_cost,
            exploration_cost=exploration_cost,
        )

    def get(self, robot: str, frontier_id: str) -> FrontierStatistics:
        return self._statistics


class FrontierStatisticsModel(Protocol):
    """A model predicting frontier statistics from observations.

    Called with one observation per visible frontier (a batch, so neural
    networks evaluate all frontiers in a single forward pass) and must
    return one :class:`FrontierStatistics` per observation, in order.
    """

    def __call__(
        self, observations: Sequence[FrontierObservation]
    ) -> Sequence[FrontierStatistics]: ...


class LearnedFrontierStatistics(FrontierStatisticsEstimator):
    """Statistics predicted by a model from panorama observations.

    On :meth:`refresh`, each frontier is paired with its best panoramic
    vantage and turned into a :class:`FrontierObservation` (exactly the
    inputs stored in the training data); the model predicts statistics
    for the whole batch. Frontiers no panorama has seen yet fall back to
    *default*. Requires no oracle, so it works in deployment.
    """

    def __init__(
        self,
        model: FrontierStatisticsModel,
        *,
        vantage_inflation_radius: float = 1.0,
        default: FrontierStatistics = DEFAULT_FRONTIER_STATISTICS,
    ) -> None:
        self._model = model
        self._vantage_inflation_radius = vantage_inflation_radius
        self._default = default
        self._statistics: Dict[str, FrontierStatistics] = {}

    def refresh(self, environment: FrontierStatisticsEnvironment) -> None:
        views = compute_frontier_views(
            frontiers=environment.frontiers,
            pano_records=getattr(environment, "pano_records", []),
            goal_cell=environment.goal_cell,
            vantage_inflation_radius=self._vantage_inflation_radius,
        )
        self._statistics = {}
        if not views:
            return
        frontier_ids = list(views)
        predictions = self._model(
            [views[fid].observation for fid in frontier_ids]
        )
        self._statistics = dict(zip(frontier_ids, predictions))

    def get(self, robot: str, frontier_id: str) -> FrontierStatistics:
        return self._statistics.get(frontier_id, self._default)
