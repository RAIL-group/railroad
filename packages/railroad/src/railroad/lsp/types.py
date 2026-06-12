"""Types for learning over subgoals planning (LSP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Tuple

import numpy as np


class FrontierStatistics(NamedTuple):
    """Planner-facing statistics of a frontier-exploration action.

    Costs are in grid cells; operator constructors convert them to time
    via a robot speed. ``delta_success_cost`` is the *extra* cost of
    reaching the goal through the frontier beyond the direct travel a
    subsequent move-to-goal accounts for (the LSP decomposition: true
    path cost minus the optimistic estimate).
    """

    prob_feasible: float
    delta_success_cost: float
    exploration_cost: float


class FrontierObservation(NamedTuple):
    """What a robot can *see* of a frontier: the input to a learned model.

    The image is the panorama from the best vantage, rolled so the
    frontier is image-centered; the frontier and goal locations are in
    that frame (x forward, y left). These fields match a
    :class:`TrainingDatum` minus the oracle labels, so a model trained
    on the generated data consumes observations directly.
    """

    image: np.ndarray  # HxWx3 uint8
    frontier_xy_ego: Tuple[float, float]
    goal_xy_ego: Tuple[float, float]


class OracleFrontierLabel(NamedTuple):
    """Ground-truth label of a frontier for point-goal navigation.

    ``prob_feasible`` is 1.0 when a path through this frontier (with all
    other frontiers masked) reaches the goal in the true map, else 0.0.
    ``success_cost``/``optimistic_cost`` are set only on success;
    ``exploration_cost`` only on failure (None when the unknown region
    behind the frontier is unreachable).
    """

    frontier_id: str
    prob_feasible: float
    success_cost: float | None
    optimistic_cost: float | None
    exploration_cost: float | None
    cells_hash: str


@dataclass
class TrainingDatum:
    """One LSP training example: a frontier seen from its best vantage."""

    image: np.ndarray  # HxWx3 uint8, rolled so the frontier is centered
    frontier_xy_ego: Tuple[float, float]  # (forward, left) in image frame
    goal_xy_ego: Tuple[float, float]
    label: bool  # frontier leads to the goal
    success_cost: float | None
    optimistic_cost: float | None
    exploration_cost: float | None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LSPDataConfig:
    """Knobs for oracle labeling and training-data emission."""

    vantage_inflation_radius: float = 1.0
    # 1.0 stores the farthest-point distance; 2.0 gives the reference
    # implementation's out-and-back exploration cost.
    exploration_cost_factor: float = 1.0
    # Costs are rounded before hashing into the change-detection
    # signature so sub-cell jitter does not re-emit data.
    cost_round_decimals: int = 1
