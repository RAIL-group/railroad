import numpy as np
import pytest

from railroad.core import Fluent as F
from railroad.plotting.sprites import build_timelines, sample

COORDS = {"start": (0.0, 0.0), "shelf": (10.0, 0.0), "counter": (20.0, 0.0)}


def entry(time, *fluents):
    return {"time": time, "fluents": {F(value) for value in fluents}}


@pytest.fixture
def history():
    return [
        entry(0, "at r1 start", "free r1"),
        entry(10, "at r1 shelf", "found mug", "at mug shelf"),
        entry(20, "at r1 shelf", "found mug", "holding r1 mug"),
        entry(40, "at r1 counter", "found mug", "holding r1 mug"),
        entry(50, "at r1 counter", "found mug", "at mug counter"),
    ]


@pytest.fixture
def robot_track():
    def track(times):
        times = np.asarray(times, dtype=float)
        xs = np.interp(times, [0, 10, 20, 40, 60], [0, 10, 10, 20, 20])
        return {"r1": np.column_stack([xs, np.full_like(xs, 2.0)])}

    return track


def timeline(history, positions: list | None = None):
    return build_timelines(
        history=history,
        entity_positions={"mug": positions or []},
        selected=["mug"],
        env_coords=COORDS,
    )["mug"]


def test_rest_ride_rest_journey_and_fade(history, robot_track):
    line = timeline(history)
    assert line.found_time == 10
    assert [(a.time, a.kind) for a in line.anchors] == [
        (10, "rest"),
        (20, "ride"),
        (40, "ride"),
        (50, "rest"),
    ]

    times = np.array([0, 10, 12, 15, 20, 30, 40, 45, 50, 60], dtype=float)
    positions, alpha = sample(line, times, robot_track(times), fade=4)
    assert alpha[[0, 1, 2, 4]] == pytest.approx([0, 0, 0.5, 1])
    assert positions[1] == pytest.approx(COORDS["shelf"])
    assert 0 < positions[3, 1] < 2
    assert positions[4:7] == pytest.approx(robot_track(times)["r1"][4:7])
    assert 0 < positions[7, 1] < 2
    assert positions[8:] == pytest.approx(np.array([COORDS["counter"]] * 2))


def test_held_object_keeps_following_the_robot(robot_track):
    line = timeline(
        [
            entry(10, "found mug", "at mug shelf"),
            entry(20, "found mug", "holding r1 mug"),
        ]
    )
    times = np.array([20, 40], dtype=float)
    positions, _ = sample(line, times, robot_track(times), fade=1)
    assert positions[1] == pytest.approx((20, 2))


def test_pending_place_uses_finalized_position(robot_track):
    line = timeline(
        [entry(10, "found mug", "at mug shelf"), entry(20, "holding r1 mug")],
        [(10, "shelf", None), (60, "counter", None)],
    )
    positions, _ = sample(line, [60], robot_track([60]), fade=1)
    assert positions[0] == pytest.approx(COORDS["counter"])


def test_limbo_snapshot_still_tweens(robot_track):
    line = timeline(
        [
            entry(10, "found mug", "at mug shelf"),
            entry(15, "found mug"),
            entry(20, "found mug", "holding r1 mug"),
        ]
    )
    positions, _ = sample(line, [10, 15, 20], robot_track([10, 15, 20]), fade=1)
    assert [a.kind for a in line.anchors] == ["rest", "ride"]
    assert 0 < positions[1, 1] < 2


def test_missing_robot_track_falls_back_to_rest(history):
    positions, _ = sample(timeline(history), [30], {}, fade=1)
    assert positions[0] == pytest.approx(COORDS["shelf"])


def test_offsets_apply_to_rest_and_ride(history, robot_track):
    times = np.array([10, 30], dtype=float)
    line = timeline(history)
    plain, _ = sample(line, times, robot_track(times), fade=1)
    shifted, _ = sample(
        line, times, robot_track(times), fade=1, offset_for=lambda _anchor: (0, -2)
    )
    assert shifted - plain == pytest.approx(np.array([[0, -2], [0, -2]]))


def test_unresolvable_objects_are_dropped():
    assert not build_timelines(
        history=[entry(10, "found mug", "at mug nowhere")],
        entity_positions={},
        selected=["mug"],
        env_coords=COORDS,
    )
