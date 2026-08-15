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

    Scenes render in their own units -- metres for ProcTHOR, cells for
    railsim -- and convert here, so the dashboard can draw the image under a
    trajectory without knowing anything about the scene.

    The bounds are the image's *outer edges*, in the continuous cell units the
    grid is drawn in, where cell ``k`` is centred on ``k`` and spans ``k-0.5``
    to ``k+0.5``. ``image`` is row-major with row 0 at ``min_y``.
    """

    image: np.ndarray
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """The ``imshow(extent=...)`` 4-tuple, for ``origin="upper"``.

        Always ordered ``(left, right, bottom, top)`` whatever *origin* is, and
        ``origin="upper"`` draws row 0 at *top*. Row 0 being ``min_y``,
        ``bottom`` comes out greater than ``top`` -- which keeps the y axis
        increasing downwards, as the grid is drawn.
        """
        return (self.min_x, self.max_x, self.max_y, self.min_y)
