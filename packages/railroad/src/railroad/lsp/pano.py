"""Panorama orientation and egocentric coordinates for LSP training data.

Conventions (see ``railroad.environment.railsim.render.pano``): panoramas
are robot-aligned equirectangular images whose center column looks along
the robot heading and whose right half turns clockwise (the robot's
right). Grid frame: ``grid[i, j] <-> (x=i, y=j)``, yaw 0 faces +row and
positive yaw rotates +row toward +col, so the bearing from a pose to a
cell is ``atan2(d_col, d_row)``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

import numpy as np

from railroad.environment.types import PoseLike

if TYPE_CHECKING:
    from railroad.environment.railsim import PanoRecord


def wrap_angle(theta: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float(math.atan2(math.sin(theta), math.cos(theta)))


def bearing_to_target(
    pose: PoseLike, target_rc: tuple[float, float]
) -> float:
    """Bearing of *target_rc* (row, col) relative to the pose heading."""
    world = math.atan2(
        float(target_rc[1]) - float(pose.y),
        float(target_rc[0]) - float(pose.x),
    )
    return wrap_angle(world - float(pose.yaw))


def roll_pano_to_bearing(
    image: np.ndarray, relative_bearing: float
) -> np.ndarray:
    """Roll a robot-aligned pano so *relative_bearing* is image-centered.

    A pano rendered at yaw ``theta`` equals
    ``np.roll(pano_at_yaw0, +round(W * theta / 2pi), axis=1)``; rolling a
    pose-aligned pano by the relative bearing therefore re-centers the
    image on that bearing, as if the robot had turned to face it.
    """
    width = image.shape[1]
    shift = int(round(width * relative_bearing / (2.0 * math.pi)))
    return np.roll(image, shift, axis=1)


def egocentric_xy(
    pose: PoseLike,
    target_rc: tuple[float, float],
    frame_yaw: float,
) -> Tuple[float, float]:
    """Target location in a frame at the pose position with yaw *frame_yaw*.

    Returns ``(x, y)`` with x forward along the frame heading and y to
    the left (image-right is -y).
    """
    d_row = float(target_rc[0]) - float(pose.x)
    d_col = float(target_rc[1]) - float(pose.y)
    cos_y = math.cos(frame_yaw)
    sin_y = math.sin(frame_yaw)
    return (
        cos_y * d_row + sin_y * d_col,
        -sin_y * d_row + cos_y * d_col,
    )


def make_training_view(
    record: "PanoRecord",
    frontier_rc: tuple[float, float],
    goal_rc: tuple[float, float],
) -> tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Orient a pano toward a frontier and compute egocentric locations.

    Returns the rolled image (frontier bearing image-centered) and the
    (x, y) locations of the frontier and the goal in the oriented frame.
    """
    pose = record.pose_cells
    relative_bearing = bearing_to_target(pose, frontier_rc)
    image = roll_pano_to_bearing(record.image, relative_bearing)
    frame_yaw = float(pose.yaw) + relative_bearing
    return (
        image,
        egocentric_xy(pose, frontier_rc, frame_yaw),
        egocentric_xy(pose, goal_rc, frame_yaw),
    )
