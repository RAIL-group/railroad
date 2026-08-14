"""Common environment types shared across submodules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
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


@dataclass(frozen=True)
class TopDownView:
    """A top-down scene image positioned in occupancy-grid cell coordinates.

    Scenes render their overhead view in their own native frame -- metres for
    ProcTHOR, cells for railsim -- so each provider converts to grid cells here.
    That lets the dashboard draw the image underneath a trajectory without
    knowing anything about the scene's units.

    Attributes:
        image: Row-major RGB array, row 0 at ``min_y`` and column 0 at ``min_x``.
        min_x: Left edge along the plot's data-x axis (the grid's first index).
        max_x: Right edge along data-x.
        min_y: Top edge along the plot's data-y axis (the grid's second index).
        max_y: Bottom edge along data-y.

    The bounds are the image's *outer edges*, in the same continuous cell units
    the occupancy grid is drawn in -- where cell ``k`` is centred on ``k`` and
    spans ``k - 0.5`` to ``k + 0.5``.
    """

    image: np.ndarray
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """The ``imshow(extent=...)`` 4-tuple, for ``origin="upper"``.

        Matplotlib always orders this ``(left, right, bottom, top)`` whatever
        *origin* is; with ``origin="upper"`` row 0 is drawn at *top*. Since row
        0 is ``min_y``, ``bottom`` comes out greater than ``top`` -- which is
        what keeps the y axis increasing downwards, matching how the occupancy
        grid is drawn.
        """
        return (self.min_x, self.max_x, self.max_y, self.min_y)
