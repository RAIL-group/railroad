"""A candidate policy to replay over a recorded deployment.

The three replay domains consume differently-shaped policy knobs (a
frontier-statistics estimator for navigation; frontier/container find-probability
callables for object search). :class:`CandidatePolicy` is the single container the
unified :func:`~railroad.replay.domains.replay` entry accepts, so the driver and
the deferred selection layer (:mod:`railroad.replay.selection`) can hold and
iterate a list of candidates uniformly — each domain reads the fields it needs
and ignores the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Sequence

if TYPE_CHECKING:
    from railroad.lsp.frontier_statistics import FrontierStatisticsEstimator

# (robot, subgoal, object) -> probability. Shared by both search operators.
ProbFn = Callable[[str, str, str], float]


@dataclass
class CandidatePolicy:
    """A replayable candidate, carrying whatever the target domain consumes.

    - **navigation**: ``frontier_statistics`` (a ``FrontierStatisticsEstimator``).
    - **unknown-map search**: ``frontier_find_prob`` + ``container_find_prob``
      (and ``refresh_estimators`` for served-vantage perception — normally a
      single ``LearnedFrontierStatistics`` fed the recorded panoramas).
    - **known-map search**: ``container_find_prob``.

    Every field is optional; a domain that needs one and finds it ``None``
    falls back to a neutral prior (so a bare ``CandidatePolicy()`` is a valid,
    policy-agnostic replay). ``name`` labels the candidate for selection output.
    """

    name: str = ""
    frontier_statistics: "FrontierStatisticsEstimator | None" = None
    frontier_find_prob: Optional[ProbFn] = None
    container_find_prob: Optional[ProbFn] = None
    refresh_estimators: Sequence = ()
