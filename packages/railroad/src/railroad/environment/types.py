"""Common environment types shared across submodules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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
