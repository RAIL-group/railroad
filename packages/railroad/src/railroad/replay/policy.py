"""A candidate policy to replay over a recorded deployment.

The three replay flavors consume differently-shaped policy knobs (a
frontier-statistics estimator for navigation; frontier/container find-probability
callables for object search). :class:`CandidatePolicy` is the single container the
:func:`~railroad.replay.driver.run_replay` entry accepts, so the driver and the
deferred selection layer (:mod:`railroad.replay.selection`) can hold and iterate a
list of candidates uniformly — each replay environment reads the fields it needs
(via ``apply_policy``) and ignores the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from railroad.lsp.frontier_statistics import (
    FixedPriorFrontierStatistics,
    FrontierStatisticsEstimator,
)

# (robot, subgoal, object) -> probability. Shared by both search operators.
ProbFn = Callable[[str, str, str], float]

# The policy-agnostic find-probability: a bare CandidatePolicy() replays with
# this everywhere a find-prob is read, so the arena is valid without any policy.
NEUTRAL_FIND_PROB = 0.5


def _neutral_find_prob(robot: str, subgoal: str, obj: str) -> float:
    return NEUTRAL_FIND_PROB


@dataclass
class CandidatePolicy:
    """A replayable candidate, carrying whatever the target flavor consumes.

    - **navigation**: ``frontier_statistics`` (a ``FrontierStatisticsEstimator``).
    - **unknown-map search**: ``frontier_find_prob`` + ``container_find_prob``
      (and ``refresh_estimators`` for served-vantage perception — normally a
      single ``LearnedFrontierStatistics`` fed the recorded panoramas).
    - **known-map search**: ``container_find_prob``.

    Every field is optional; the ``resolve_*`` helpers fall back to a neutral
    prior when a field is ``None`` (so a bare ``CandidatePolicy()`` is a valid,
    policy-agnostic replay). ``name`` labels the candidate for selection output.
    """

    name: str = ""
    frontier_statistics: Optional[FrontierStatisticsEstimator] = None
    frontier_find_prob: Optional[ProbFn] = None
    container_find_prob: Optional[ProbFn] = None
    refresh_estimators: Sequence = ()

    def resolve_frontier_find_prob(self) -> ProbFn:
        """The frontier find-probability, or the neutral prior if unset."""
        return self.frontier_find_prob or _neutral_find_prob

    def resolve_container_find_prob(self) -> ProbFn:
        """The container find-probability, or the neutral prior if unset."""
        return self.container_find_prob or _neutral_find_prob

    def resolve_frontier_statistics(self) -> FrontierStatisticsEstimator:
        """The frontier-statistics estimator, or a neutral fixed prior if unset."""
        if self.frontier_statistics is not None:
            return self.frontier_statistics
        return FixedPriorFrontierStatistics()
