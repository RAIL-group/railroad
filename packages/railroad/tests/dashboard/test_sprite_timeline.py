"""Where an object's sprite sits over the course of a plan.

The interesting cases are the two windows where the object is in *neither*
``at`` nor ``holding``: ``pick`` deletes the location at dispatch and only grants
``holding`` on completion, and ``place`` mirrors it. Those are the spans the
sprite should be travelling, not vanishing.
"""

import numpy as np
import pytest

from railroad.core import Fluent as F
from railroad.dashboard._sprites import build_timelines, sample

COORDS = {"start": (0.0, 0.0), "shelf": (10.0, 0.0), "counter": (20.0, 0.0)}


def _entry(time, fluents):
    return {"time": time, "fluents": set(fluents)}


@pytest.fixture
def carried_history():
    """Found at the shelf at t=10, picked over 10-20, placed over 40-50."""
    return [
        _entry(0.0, [F("at r1 start"), F("free r1")]),
        _entry(10.0, [F("at r1 shelf"), F("found mug"), F("at mug shelf")]),
        _entry(20.0, [F("at r1 shelf"), F("found mug"), F("holding r1 mug")]),
        _entry(40.0, [F("at r1 counter"), F("found mug"), F("holding r1 mug")]),
        _entry(50.0, [F("at r1 counter"), F("found mug"), F("at mug counter")]),
    ]


@pytest.fixture
def robot_track():
    """r1 waits at the shelf through the pick, drives over, waits through the place.

    A robot stands *beside* a container rather than on it, so its y differs from
    the location's -- which is what makes a pick tween visible at all.
    """

    def track(times):
        times = np.asarray(times, dtype=float)
        keys = [0.0, 10.0, 20.0, 40.0, 60.0]
        xs = np.interp(times, keys, [0.0, 10.0, 10.0, 20.0, 20.0])
        return {"r1": np.column_stack([xs, np.full_like(xs, 2.0)])}

    return track


def _timeline(history, positions=None):
    return build_timelines(
        history=history,
        entity_positions={"mug": positions or []},
        selected=["mug"],
        env_coords=COORDS,
    )["mug"]


def test_anchors_capture_the_rest_ride_rest_story(carried_history):
    timeline = _timeline(carried_history)
    assert timeline.found_time == 10.0
    assert [(a.time, a.kind) for a in timeline.anchors] == [
        (10.0, "rest"), (20.0, "ride"), (40.0, "ride"), (50.0, "rest")
    ]


def test_invisible_until_found_then_fades_in(carried_history, robot_track):
    timeline = _timeline(carried_history)
    times = np.array([0.0, 5.0, 10.0, 12.0, 14.0, 20.0])
    _pos, alpha = sample(timeline, times, robot_track(times), fade=4.0)
    assert list(alpha[:3]) == [0.0, 0.0, 0.0]
    assert alpha[3] == pytest.approx(0.5)
    assert alpha[4] == pytest.approx(1.0)
    assert alpha[5] == pytest.approx(1.0)


def test_rests_at_its_location_before_being_picked(carried_history, robot_track):
    timeline = _timeline(carried_history)
    times = np.array([10.0])
    pos, _alpha = sample(timeline, times, robot_track(times), fade=4.0)
    assert pos[0] == pytest.approx(COORDS["shelf"])


def test_pick_tween_runs_from_the_shelf_to_the_robot(carried_history, robot_track):
    """Strictly between the two at the midpoint, exactly at the robot on completion."""
    timeline = _timeline(carried_history)
    times = np.array([10.0, 15.0, 20.0])
    pos, _alpha = sample(timeline, times, robot_track(times), fade=4.0)
    assert pos[0] == pytest.approx(COORDS["shelf"])
    assert 0.0 < pos[1][1] < 2.0
    assert pos[2] == pytest.approx((10.0, 2.0))


def test_carried_sprite_tracks_the_moving_robot(carried_history, robot_track):
    timeline = _timeline(carried_history)
    times = np.array([20.0, 30.0, 40.0])
    pos, _alpha = sample(timeline, times, robot_track(times), fade=4.0)
    assert pos == pytest.approx(robot_track(times)["r1"])


def test_place_tween_ends_at_the_destination(carried_history, robot_track):
    timeline = _timeline(carried_history)
    times = np.array([45.0, 50.0, 60.0])
    pos, _alpha = sample(timeline, times, robot_track(times), fade=4.0)
    assert 0.0 < pos[0][1] < 2.0
    assert pos[1] == pytest.approx(COORDS["counter"])
    assert pos[2] == pytest.approx(COORDS["counter"])


def test_still_held_at_goal_time_keeps_following_the_robot(robot_track):
    history = [
        _entry(10.0, [F("found mug"), F("at mug shelf")]),
        _entry(20.0, [F("found mug"), F("holding r1 mug")]),
    ]
    timeline = _timeline(history)
    times = np.array([20.0, 40.0])
    pos, _alpha = sample(timeline, times, robot_track(times), fade=1.0)
    assert pos[1] == pytest.approx((20.0, 2.0))  # the robot's position at t=40


def test_in_flight_place_uses_the_finalized_position(robot_track):
    """finalize_trajectories reads pending `at` out of the upcoming effects."""
    history = [
        _entry(10.0, [F("found mug"), F("at mug shelf")]),
        _entry(20.0, [F("found mug"), F("holding r1 mug")]),
    ]
    timeline = _timeline(history, positions=[(10.0, "shelf", None), (60.0, "counter", None)])
    assert timeline.anchors[-1].loc == "counter"
    times = np.array([60.0])
    pos, _alpha = sample(timeline, times, robot_track(times), fade=1.0)
    assert pos[0] == pytest.approx(COORDS["counter"])


def test_a_limbo_snapshot_tweens_rather_than_teleports(robot_track):
    """Multi-robot: act() can return mid-pick, with the object in neither fluent."""
    history = [
        _entry(10.0, [F("found mug"), F("at mug shelf")]),
        _entry(15.0, [F("found mug")]),  # pick dispatched, not yet complete
        _entry(20.0, [F("found mug"), F("holding r1 mug")]),
    ]
    timeline = _timeline(history)
    times = np.array([10.0, 15.0, 20.0])
    pos, _alpha = sample(timeline, times, robot_track(times), fade=1.0)
    assert [a.kind for a in timeline.anchors] == ["rest", "ride"]
    assert pos[0] == pytest.approx(COORDS["shelf"])
    assert 0.0 < pos[1][1] < 2.0
    assert np.all(np.isfinite(pos))


def test_a_carrier_without_an_interpolated_track_falls_back_to_a_rest(carried_history):
    """Robots with too few waypoints are dropped from marker_positions."""
    timeline = _timeline(carried_history)
    times = np.array([30.0])
    pos, _alpha = sample(timeline, times, {}, fade=1.0)
    assert pos[0] == pytest.approx(COORDS["shelf"])


def test_sampling_is_a_pure_function_of_time(carried_history, robot_track):
    """What lets the video's poster frame jump to the end and rewind."""
    timeline = _timeline(carried_history)
    times = np.linspace(0.0, 60.0, 25)
    batch_pos, batch_alpha = sample(timeline, times, robot_track(times), fade=4.0)
    for index, time in enumerate(times):
        one_pos, one_alpha = sample(timeline, [time], robot_track([time]), fade=4.0)
        assert one_pos[0] == pytest.approx(batch_pos[index])
        assert one_alpha[0] == pytest.approx(batch_alpha[index])


def test_offsets_shift_rest_and_ride_alike(carried_history, robot_track):
    timeline = _timeline(carried_history)
    times = np.array([10.0, 30.0])  # one resting, one riding
    plain, _ = sample(timeline, times, robot_track(times), fade=1.0)
    shifted, _ = sample(
        timeline, times, robot_track(times), fade=1.0, offset_for=lambda _a: (0.0, -2.0)
    )
    assert shifted - plain == pytest.approx(np.array([[0.0, -2.0], [0.0, -2.0]]))


def test_objects_without_resolvable_coordinates_are_dropped():
    history = [_entry(10.0, [F("found mug"), F("at mug nowhere")])]
    assert build_timelines(
        history=history, entity_positions={}, selected=["mug"], env_coords=COORDS
    ) == {}
