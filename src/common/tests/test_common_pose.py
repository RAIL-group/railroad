"""Some simple tests for the shared Pose class"""

import math
import pytest

from common import Pose, compute_path_length


def test_pose_core():
    pose_a = Pose(x=1, y=2)
    assert pose_a.x == pytest.approx(1)
    assert pose_a.y == pytest.approx(2)
    assert pose_a.yaw == pytest.approx(0)

    pose_b = Pose(3, 5.5)
    assert pose_b.x == pytest.approx(3)
    assert pose_b.y == pytest.approx(5.5)
    assert pose_b.yaw == pytest.approx(0)

    pose_c = Pose(8.75, 5, yaw=2.2)
    assert pose_c.x == pytest.approx(8.75)
    assert pose_c.y == pytest.approx(5)
    assert pose_c.yaw == pytest.approx(2.2)

    assert Pose.cartesian_distance(pose_a, Pose(1, 5)) == pytest.approx(3)
    assert Pose.cartesian_distance(pose_a, Pose(1, 5, 0.1)) == pytest.approx(3)
    assert Pose.cartesian_distance(pose_a, Pose(5, 2, 0.1)) == pytest.approx(4)
    assert Pose.cartesian_distance(pose_a,
                                   Pose(2, 1,
                                        0.1)) == pytest.approx(math.sqrt(2))


def test_pose_composition_no_yaw():
    """Test that we can combine pose objects (no yaw)."""

    pose_a = Pose(x=1, y=2)
    pose_b = Pose(x=3, y=5.5)

    pose_aa = pose_a * pose_a
    pose_ab = pose_a * pose_b
    pose_ba = pose_b * pose_a

    assert pose_aa.x == pytest.approx(2)
    assert pose_aa.y == pytest.approx(4)
    assert pose_aa.yaw == pytest.approx(0)
    assert Pose.cartesian_distance(pose_aa, Pose(2, 4)) == pytest.approx(0)

    assert pose_ab.x == pytest.approx(4)
    assert pose_ab.y == pytest.approx(7.5)
    assert pose_ab.yaw == pytest.approx(0)
    assert Pose.cartesian_distance(pose_ab, Pose(4, 7.5)) == pytest.approx(0)

    assert pose_ba.x == pytest.approx(4)
    assert pose_ba.y == pytest.approx(7.5)
    assert pose_ba.yaw == pytest.approx(0)
    assert Pose.cartesian_distance(pose_ba, Pose(4, 7.5)) == pytest.approx(0)


def test_pose_composition_with_yaw():
    """Test that we can combine pose objects (no yaw)."""

    pose_a = Pose(x=1, y=2, yaw=math.pi)
    pose_b = Pose(x=3, y=5.5, yaw=math.pi / 2)

    # Applying a twice gets us back to the origin
    pose_aa = pose_a * pose_a
    assert pose_aa.x == pytest.approx(0)
    assert pose_aa.y == pytest.approx(0)
    assert (pose_aa.yaw == pytest.approx(0) or pose_aa.yaw == pytest.approx(2 * math.pi))

    # First b, then a (we left-multiply transforms)
    pose_ab = pose_a * pose_b
    assert pose_ab.x == pytest.approx(pose_b.x - pose_a.y)
    assert pose_ab.y == pytest.approx(pose_b.y + pose_a.x)
    assert (pose_ab.yaw == pytest.approx(3 * math.pi / 2)
            or pose_ab.yaw == pytest.approx(- math.pi / 2))

    # First a, then b (we left-multiply transforms)
    pose_ba = pose_b * pose_a
    assert pose_ba.x == pytest.approx(pose_a.x - pose_b.x)
    assert pose_ba.y == pytest.approx(pose_a.y - pose_b.y)
    assert (pose_ba.yaw == pytest.approx(3 * math.pi / 2)
            or pose_ba.yaw == pytest.approx(- math.pi / 2))


def test_compute_pose_path_length():
    pose_a = Pose(0, 0)
    pose_b = Pose(3, 4)
    pose_c = Pose(7, 7)

    path_len_a = compute_path_length([pose_a])
    assert path_len_a == pytest.approx(0)

    path_len_aa = compute_path_length([pose_a, pose_a])
    assert path_len_aa == pytest.approx(0)

    path_len_bb = compute_path_length([pose_b, pose_b])
    assert path_len_bb == pytest.approx(0)

    path_len_ab = compute_path_length([pose_a, pose_b])
    assert path_len_ab == pytest.approx(5)

    path_len_aba = compute_path_length([pose_a, pose_b, pose_a])
    assert path_len_aba == pytest.approx(10)

    path_len_abc = compute_path_length([pose_a, pose_b, pose_c])
    assert path_len_abc == pytest.approx(10)
