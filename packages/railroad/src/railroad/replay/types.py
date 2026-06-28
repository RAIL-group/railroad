"""Serializable records for offline replay.

A :class:`RolloutLog` is everything one real deployment hands to offline
replay (see ``replay_design.md`` §4). For *navigation* replay the load-
bearing fields are ``recorded_grid`` (the final observed map — the replay
arena), ``goal_cell``, and ``robot_starts`` / ``config``; the rest is
provenance. :class:`ReplayResult` is what a replay produces: the two
bounds plus the commits and termination reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

from .cost import Bounds, Commit

if TYPE_CHECKING:
    from railroad.environment.railsim import PanoRecord

Pose3 = Tuple[float, float, float]


@dataclass(frozen=True)
class SubgoalRecord:
    """A recorded subgoal (a frontier of the final map, or a container).

    ``signature`` is the replay-stable key (a cell-set hash for frontiers,
    §6.1). ``contents`` carries per-container observations for the search
    extension (§4); empty for navigation.
    """

    signature: str
    centroid: Tuple[int, int]
    cells: np.ndarray  # 2xN (row, col)
    contents: Tuple[str, ...] = ()


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
    subgoals: List[SubgoalRecord] = field(default_factory=list)
    steps: List[StepRecord] = field(default_factory=list)
    actual_total_cost: float = 0.0
    # The deployment's NavigationConfig (dataclasses.asdict): replay rebuilds its
    # config from this so it senses/maps exactly as the deployment did. Values
    # are float/int/bool (preserved through JSON); empty for logs built without a
    # live env (replay then falls back to a default config).
    config: Dict[str, Any] = field(default_factory=dict)
    # Accumulated panorama buffer from the deployment. Served to a learned
    # frontier-statistics estimator during replay (best-vantage perception,
    # design §2.1); empty for non-visual logs. Typed loosely so the replay
    # core stays importable without the railsim (GL) extra.
    pano_records: List["PanoRecord"] = field(default_factory=list)

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
