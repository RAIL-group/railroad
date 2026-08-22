"""Tests for panorama orientation and egocentric coordinate math."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lsp.helpers import FakeRecord

from railroad.environment.types import Pose
from railroad.lsp import (
    bearing_to_target,
    egocentric_xy,
    make_training_view,
    roll_pano_to_bearing,
    wrap_angle,
)


def _column_image(width: int = 8, height: int = 2) -> np.ndarray:
    """Image whose every pixel value equals its column index."""
    return np.tile(np.arange(width, dtype=np.uint8), (height, 1))[..., None].repeat(
        3, axis=-1
    )


def test_wrap_angle() -> None:
    assert wrap_angle(0.0) == 0.0
    assert wrap_angle(2 * math.pi) == pytest.approx(0.0, abs=1e-9)
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert wrap_angle(-3 * math.pi / 2) == pytest.approx(math.pi / 2)


def test_bearing_to_target() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    # +row is straight ahead at yaw 0
    assert bearing_to_target(pose, (5.0, 0.0)) == pytest.approx(0.0)
    # +col is the robot's left (positive bearing)
    assert bearing_to_target(pose, (0.0, 5.0)) == pytest.approx(math.pi / 2)
    # Yaw rotates the frame: facing +col, +col is straight ahead
    pose_left = Pose(0.0, 0.0, math.pi / 2)
    assert bearing_to_target(pose_left, (0.0, 5.0)) == pytest.approx(0.0)


def test_roll_identity_at_zero_bearing() -> None:
    image = _column_image(8)
    np.testing.assert_array_equal(roll_pano_to_bearing(image, 0.0), image)


def test_roll_centers_target_column() -> None:
    """Content at the bearing's source column moves to image center.

    With width 8 and the pano convention (column c covers longitude
    pi - 2*pi*(c+0.5)/W), a target at bearing +pi/2 (robot's left) sits
    between columns 1 and 2; rolling by the bearing shifts it to the
    center (between columns 3 and 4).
    """
    image = _column_image(8)
    rolled = roll_pano_to_bearing(image, math.pi / 2)
    assert rolled[0, 3, 0] == image[0, 1, 0]
    assert rolled[0, 4, 0] == image[0, 2, 0]

    # Negative bearing (robot's right) rolls the other way.
    rolled_right = roll_pano_to_bearing(image, -math.pi / 2)
    assert rolled_right[0, 3, 0] == image[0, 5, 0]


def test_roll_full_turn_is_identity() -> None:
    image = _column_image(11)
    np.testing.assert_array_equal(
        roll_pano_to_bearing(image, 2 * math.pi), image
    )


def test_egocentric_xy_conventions() -> None:
    pose = Pose(0.0, 0.0, 0.0)
    # Straight ahead in a yaw-0 frame: x forward
    assert egocentric_xy(pose, (3.0, 0.0), 0.0) == pytest.approx((3.0, 0.0))
    # +col in a yaw-0 frame: y left
    assert egocentric_xy(pose, (0.0, 2.0), 0.0) == pytest.approx((0.0, 2.0))
    # Frame rotated to face the target: target is straight ahead
    frame_yaw = math.atan2(4.0, 3.0)
    x, y = egocentric_xy(pose, (3.0, 4.0), frame_yaw)
    assert x == pytest.approx(5.0)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_make_training_view() -> None:
    pose = Pose(10.0, 10.0, 0.3)
    record = FakeRecord(
        robot="robot1",
        time=1.0,
        pose_cells=pose,
        pose_meters=(0.0, 0.0, 0.3),
        image=_column_image(360, 4),
    )
    frontier_rc = (16.0, 18.0)
    goal_rc = (30.0, 10.0)

    image, frontier_xy, goal_xy = make_training_view(record, frontier_rc, goal_rc)
    assert image.shape == record.image.shape

    # The frontier lies straight ahead in the oriented frame.
    distance = math.hypot(6.0, 8.0)
    assert frontier_xy[0] == pytest.approx(distance)
    assert frontier_xy[1] == pytest.approx(0.0, abs=1e-9)

    # The goal keeps its range, expressed in the same frame.
    assert math.hypot(*goal_xy) == pytest.approx(20.0)
