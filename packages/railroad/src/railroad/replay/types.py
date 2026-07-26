"""Serializable records for offline replay.

A :class:`RolloutLog` is everything one real deployment hands to offline replay.
For *navigation* replay the load-bearing fields are ``recorded_grid`` (the final
observed map — the replay arena), ``goal_cell``, and ``robot_starts`` /
``config``; the rest is provenance. :class:`ReplayResult` is what a replay
produces: the two bounds plus the commits and termination reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from railroad._bindings import Fluent, Goal

from .cost import Bounds, Commit

Pose3 = Tuple[float, float, float]


@dataclass(frozen=True)
class SubgoalRecord:
    """A recorded subgoal (a frontier of the final map, or a container).

    ``signature`` is the replay-stable key (a cell-set hash for frontiers).
    ``contents`` carries per-container observations for the search extension;
    empty for navigation.

    ``searched`` records whether the deployment actually *searched* this
    container (vs. merely revealing its cell). It is load-bearing for the bound:
    a searched container's outcome is known (replay it exactly, no optimism),
    while a revealed-but-unsearched container's outcome is unknown — replay must
    force not-found and log an optimistic commit there. We do **not**
    infer an unsearched container's emptiness from the object being found
    elsewhere: that would assume an object occupies exactly one container, which
    the bound must not rely on. ``contents`` is empty unless ``searched``.
    """

    signature: str
    centroid: Tuple[int, int]
    cells: np.ndarray  # 2xN (row, col)
    contents: Tuple[str, ...] = ()
    searched: bool = False


@dataclass
class StepRecord:
    """Provenance for one planning step of the recorded deployment."""

    time: float
    robot_poses: Dict[str, Pose3]
    chosen_action: str
    net_motion: Dict[str, float]


@dataclass
class RolloutLog:
    """A single recorded deployment, the input to offline replay."""

    recorded_grid: np.ndarray
    goal_cell: Tuple[int, int]
    robot_starts: Dict[str, Pose3]
    problem_class: str = "navigation"
    env_name: str = ""
    seed: int | None = None
    # The deployment's planning goal — a full ``Goal`` (or ``Fluent``), so it may
    # be compound (e.g. ``found book & found plate``), not just one object.
    # Recorded so the log is self-describing: build_replay_env reads the goal off
    # the log and replays toward it (search flavors require it; navigation falls
    # back to the point-goal derived from ``robot_starts`` when absent).
    goal: Goal | Fluent | None = None
    subgoals: List[SubgoalRecord] = field(default_factory=list)
    steps: List[StepRecord] = field(default_factory=list)
    actual_total_cost: float = 0.0
    # The deployment's NavigationConfig (dataclasses.asdict): replay rebuilds its
    # config from this so it senses/maps exactly as the deployment did. Values
    # are float/int/bool (preserved through JSON); empty for logs built without a
    # live env (replay then falls back to a default config).
    config: Dict[str, Any] = field(default_factory=dict)
    # Accumulated panorama buffer from the deployment. Served to a learned
    # frontier-statistics estimator during replay (best-vantage perception); empty
    # for non-visual logs. Held as ``Any`` because it is genuinely heterogeneous —
    # a railsim ``PanoRecord`` from a real deploy, a ``ServedPano`` during replay,
    # or a ``LoadedPanoRecord`` after load — and importing railsim's ``PanoRecord``
    # would pull the GL extra the replay core stays free of.
    pano_records: List[Any] = field(default_factory=list)

    @property
    def robots(self) -> List[str]:
        return sorted(self.robot_starts)


@dataclass
class ReplayResult:
    """The outcome of replaying one alternative policy over a log."""

    bounds: Bounds
    commits: List[Commit]
    termination: str
    total_cost: float
    sim_time: float
    goal_reached: bool
    # Object-search provenance: (location, cost_accrued, found) per search; empty
    # for navigation.
    search_log: List[Tuple[str, float, bool]] = field(default_factory=list)
