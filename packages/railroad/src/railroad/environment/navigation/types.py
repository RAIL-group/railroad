"""Types for unknown-space navigation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PoseLike(Protocol):
    """Protocol for objects with x, y, yaw properties (grid coordinates)."""

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...

    @property
    def yaw(self) -> float: ...


@dataclass
class Pose:
    """Concrete robot pose in grid coordinates.

    Attributes:
        x: Row position (float).
        y: Column position (float).
        yaw: Heading in radians.
    """

    x: float
    y: float
    yaw: float = 0.0


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
    occupied_prob: float = 0.9
    unoccupied_prob: float = 0.1
    connect_neighbor_distance: int = 2
    interrupt_min_new_cells: int = 20
    interrupt_min_dt: float = 1.0
    max_move_action_time: float = float("inf")
    scan_inflation_radius: float = 5.0
    correct_with_known_map: bool = True
    record_frames: bool = True
