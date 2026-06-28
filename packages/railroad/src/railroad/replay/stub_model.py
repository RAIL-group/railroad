"""Stub frontier-statistics models — fake the *output*, keep the real pipeline.

A learned policy in this stack is ``LearnedFrontierStatistics(model)``, where
``model`` is a :class:`~railroad.lsp.frontier_statistics.FrontierStatisticsModel`
mapping a batch of :class:`~railroad.lsp.types.FrontierObservation` (a frontier's
best-vantage panorama + egocentric geometry) to per-frontier
:class:`~railroad.lsp.types.FrontierStatistics`.

During development we fake **only** the numeric output with preset values, while
exercising the real served-vantage pipeline (record panoramas → serve in replay →
build observations → call the model). A trained network drops in at the *same call
site* with no other change::

    from railroad.lsp.model import load_frontier_statistics_model
    estimator = LearnedFrontierStatistics(load_frontier_statistics_model(path))
    #                                      ^ replaces preset_model(...)

This module is torch-free and GL-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from railroad.lsp.types import FrontierObservation, FrontierStatistics


@dataclass
class PresetFrontierStatisticsModel:
    """A ``FrontierStatisticsModel`` returning preset stats per observation.

    Same protocol as the trained ``LSPFrontierNet`` wrapper, so it is
    interchangeable with ``load_frontier_statistics_model(...)``. It still
    *receives* the real observations (proving the pipeline); it just ignores
    them and returns the configured constants.
    """

    prob_feasible: float = 0.8
    delta_success_cost: float = 0.0
    exploration_cost: float = 10.0

    def __call__(
        self, observations: Sequence[FrontierObservation]
    ) -> List[FrontierStatistics]:
        stats = FrontierStatistics(
            prob_feasible=self.prob_feasible,
            delta_success_cost=self.delta_success_cost,
            exploration_cost=self.exploration_cost,
        )
        return [stats for _ in observations]


# A few example "models" of differing quality so candidate policies behave
# differently under replay (the input a selection layer compares).
_PROFILES: Dict[str, Dict[str, float]] = {
    "optimistic": {"prob_feasible": 0.9, "exploration_cost": 8.0},
    "cautious": {"prob_feasible": 0.3, "exploration_cost": 20.0},
    "uniform": {"prob_feasible": 0.5, "exploration_cost": 10.0},
}


def preset_model(
    profile: str = "uniform", **overrides: float
) -> PresetFrontierStatisticsModel:
    """Build a preset model from a named profile, with optional overrides."""
    params: Dict[str, float] = dict(_PROFILES.get(profile, {}))
    params.update(overrides)
    return PresetFrontierStatisticsModel(**params)


@dataclass
class PresetSearchModel:
    """Preset object-find probabilities for the search domain.

    The search operators take an ``object_find_prob(robot, loc, obj)`` callable.
    This stub returns a constant (optionally per-object) probability; a learned
    search estimator replaces it later via the same ``prob`` call site.
    """

    find_prob: float = 0.5
    per_object: Dict[str, float] = field(default_factory=dict)

    def prob(self, robot: str, loc: str, obj: str) -> float:
        del robot, loc
        return self.per_object.get(obj, self.find_prob)
