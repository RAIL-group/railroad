"""Frontier-property providers: where explore-action parameters come from.

A provider maps a (robot, frontier) pair to the planner-facing
:class:`FrontierProperties`. The oracle provider reads true-map labels
maintained by the environment; the optimistic provider returns fixed
constants. A learned model implementing the same protocol slots in later.
"""

from __future__ import annotations

from typing import Callable, Mapping, Protocol

from .types import FrontierProperties, OracleFrontierLabel

DEFAULT_FRONTIER_PROPERTIES = FrontierProperties(
    prob_feasible=0.5, delta_success_cost=0.0, exploration_cost=10.0
)


class FrontierPropertyProvider(Protocol):
    """Source of planner-facing frontier-exploration parameters."""

    def get(self, robot: str, frontier_id: str) -> FrontierProperties: ...


class OracleFrontierPropertyProvider:
    """Reads true-map oracle labels via a deferred lookup callable.

    ``labels_fn`` is called on every query so the provider always sees
    the environment's current labels (e.g. ``lambda: env.oracle_labels``).
    Frontiers without a label (or with degenerate cost fields) fall back
    to *default*.
    """

    def __init__(
        self,
        labels_fn: Callable[[], Mapping[str, OracleFrontierLabel]],
        default: FrontierProperties = DEFAULT_FRONTIER_PROPERTIES,
    ) -> None:
        self._labels_fn = labels_fn
        self._default = default

    def get(self, robot: str, frontier_id: str) -> FrontierProperties:
        label = self._labels_fn().get(frontier_id)
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
            return FrontierProperties(
                prob_feasible=label.prob_feasible,
                delta_success_cost=max(0.0, label.success_cost - optimistic),
                exploration_cost=self._default.exploration_cost,
            )
        if label.exploration_cost is None:
            return self._default
        return FrontierProperties(
            prob_feasible=label.prob_feasible,
            delta_success_cost=self._default.delta_success_cost,
            exploration_cost=label.exploration_cost,
        )


class OptimisticFrontierPropertyProvider:
    """Fixed-parameter baseline: every frontier looks promising."""

    def __init__(
        self,
        prob_feasible: float = 0.8,
        delta_success_cost: float = 0.0,
        exploration_cost: float = 10.0,
    ) -> None:
        self._properties = FrontierProperties(
            prob_feasible=prob_feasible,
            delta_success_cost=delta_success_cost,
            exploration_cost=exploration_cost,
        )

    def get(self, robot: str, frontier_id: str) -> FrontierProperties:
        return self._properties
