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
from typing import Dict, Mapping, Protocol, Sequence, Tuple

from railroad.experimental.unknown_search.types import Frontier

from .types import FrontierObservation, FrontierStatistics, OracleFrontierLabel
from .views import compute_frontier_views

DEFAULT_FRONTIER_STATISTICS = FrontierStatistics(
    prob_feasible=0.5, delta_success_cost=0.0, exploration_cost=10.0
)


class FrontierStatisticsEnvironment(Protocol):
    """What an estimator may read from the environment on refresh.

    :class:`~railroad.lsp.LSPEnvironmentMixin` satisfies this. Only the
    oracle estimator touches ``oracle_*``; deployment environments can
    report ``oracle_available = False`` and still serve the fixed-prior
    and learned estimators.
    """

    @property
    def oracle_available(self) -> bool: ...

    @property
    def oracle_labels(self) -> Mapping[str, OracleFrontierLabel]: ...

    @property
    def frontiers(self) -> Mapping[str, Frontier]: ...

    @property
    def goal_cell(self) -> Tuple[int, int]: ...


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
    """Exact statistics from the environment's true-map oracle labels.

    Only usable where an oracle exists (simulation): :meth:`refresh`
    snapshots ``environment.oracle_labels``. Frontiers without a label
    (e.g. degenerate ones) fall back to *default*.
    """

    def __init__(
        self, default: FrontierStatistics = DEFAULT_FRONTIER_STATISTICS
    ) -> None:
        self._default = default
        self._labels: Dict[str, OracleFrontierLabel] = {}

    def refresh(self, environment: FrontierStatisticsEnvironment) -> None:
        if not environment.oracle_available:
            raise RuntimeError(
                "OracleFrontierStatistics requires an environment with a "
                "true map. Without one (deployment), use "
                "FixedPriorFrontierStatistics or LearnedFrontierStatistics."
            )
        self._labels = dict(environment.oracle_labels)

    def get(self, robot: str, frontier_id: str) -> FrontierStatistics:
        label = self._labels.get(frontier_id)
        if label is None:
            return self._default
        if label.prob_feasible >= 0.5:
            if label.success_cost is None:
                return self._default
            optimistic = (
                label.optimistic_cost
                if label.optimistic_cost is not None
                else label.success_cost
            )
            return FrontierStatistics(
                prob_feasible=label.prob_feasible,
                delta_success_cost=max(0.0, label.success_cost - optimistic),
                exploration_cost=self._default.exploration_cost,
            )
        if label.exploration_cost is None:
            return self._default
        return FrontierStatistics(
            prob_feasible=label.prob_feasible,
            delta_success_cost=self._default.delta_success_cost,
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
