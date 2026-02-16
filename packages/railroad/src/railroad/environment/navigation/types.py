"""Types for unknown-space navigation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ..types import Pose as _Pose
from ..types import PoseLike as _PoseLike

# Backward-compatible re-exports.
Pose = _Pose
PoseLike = _PoseLike


class Frontier(NamedTuple):
    """A connected frontier region on the observed grid."""

    id: str
    centroid_row: int
    centroid_col: int
    cells: np.ndarray  # 2xN array of (row, col) indices


@dataclass
class NavigationConfig:
    """Runtime configuration for unknown-space navigation."""

    sensor_range: float = 60.0
    sensor_fov_rad: float = 2 * math.pi
    sensor_num_rays: int = 181
    sensor_dt: float = 0.08
    speed_cells_per_sec: float = 2.0
    trajectory_use_soft_cost: bool = True
    trajectory_soft_cost_scale: float = 6.0
    # Keep action-selection cost estimates cheap, but allow selected moves
    # to execute on any-angle Theta* paths.
    move_execution_use_theta_star: bool = False
    # If False, move skills will ignore interrupt requests and run to completion.
    move_execution_interruptible: bool = True
    occupied_prob: float = 0.9
    unoccupied_prob: float = 0.1
    connect_neighbor_distance: int = 2
    interrupt_min_new_cells: int = 20
    interrupt_min_dt: float = 1.0
    max_move_action_time: float = float("inf")
    scan_inflation_radius: float = 5.0
    correct_with_known_map: bool = True
    record_frames: bool = True
