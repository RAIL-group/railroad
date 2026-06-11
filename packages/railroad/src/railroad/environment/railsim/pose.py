"""Robot/camera pose representation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pose:
    """A planar pose in world coordinates.

    Units are meters and radians. ``yaw = 0`` faces +x; positive yaw
    rotates +x toward +y (right-handed, z-up).
    """

    x: float
    y: float
    yaw: float = 0.0
