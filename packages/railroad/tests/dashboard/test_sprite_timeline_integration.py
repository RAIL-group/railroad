"""The anchor model, checked against fluents a real environment actually emits.

Everything else about sprite placement rests on one claim: that the snapshots the
dashboard takes after each ``env.act`` straddle the pick and place windows, so the
transitions never need finer sampling. Hand-written histories cannot check that
claim -- only the operators can.
"""

import numpy as np
import pytest

from railroad import operators
from railroad._bindings import State
from railroad.core import Fluent as F, get_action_by_name
from railroad.dashboard._sprites import build_timelines, sample
from railroad.environment import ObjectSearchEnvironment

COORDS = {"shelf": (10.0, 0.0), "counter": (20.0, 0.0)}


@pytest.fixture
def plan_history():
    """Search the shelf, pick the mug, carry it to the counter, put it down."""
    class FetchEnvironment(ObjectSearchEnvironment):
        def define_operators(self):
            return [
                operators.construct_search_operator(1.0, 10.0),
                operators.construct_pick_operator_blocking(4.0),
                operators.construct_move_operator_blocking(lambda r, a, b: 8.0),
                operators.construct_place_operator_blocking(6.0),
            ]

    env = FetchEnvironment(
        state=State(0.0, {F("at r1 shelf"), F("free r1")}, []),
        objects_by_type={
            "robot": {"r1"},
            "location": {"shelf", "counter"},
            "object": {"mug"},
        },
        true_object_locations={"shelf": {"mug"}, "counter": set()},
    )

    history = [{"time": env.state.time, "fluents": set(env.state.fluents)}]
    for name in (
        "search r1 shelf mug",
        "pick r1 shelf mug",
        "move r1 shelf counter",
        "place r1 counter mug",
    ):
        env.act(get_action_by_name(env.get_actions(), name))
        history.append({"time": env.state.time, "fluents": set(env.state.fluents)})
    return history


def test_the_plan_produces_the_expected_fluent_transitions(plan_history):
    """The premise: `at` and `holding` hand off across consecutive snapshots."""
    def held(entry):
        return any(f.name == "holding" for f in entry["fluents"])

    def located(entry):
        return any(f.name == "at" and f.args[0] == "mug" for f in entry["fluents"])

    assert [round(e["time"], 1) for e in plan_history] == [0.0, 10.0, 14.0, 22.0, 28.0]
    assert [located(e) for e in plan_history] == [False, True, False, False, True]
    assert [held(e) for e in plan_history] == [False, False, True, True, False]


def test_anchors_track_the_object_through_the_whole_plan(plan_history):
    timeline = build_timelines(
        history=plan_history, entity_positions={}, selected=["mug"], env_coords=COORDS,
    )["mug"]

    assert timeline.found_time == 10.0
    assert [(a.time, a.kind, a.loc or a.robot) for a in timeline.anchors] == [
        (10.0, "rest", "shelf"),
        (14.0, "ride", "r1"),
        (22.0, "ride", "r1"),
        (28.0, "rest", "counter"),
    ]


def test_the_sprite_is_hidden_before_the_search_succeeds(plan_history):
    timeline = build_timelines(
        history=plan_history, entity_positions={}, selected=["mug"], env_coords=COORDS,
    )["mug"]
    times = np.array([0.0, 5.0, 9.9])
    _pos, alpha = sample(timeline, times, _robot(times), fade=2.0)
    assert not alpha.any()


def test_the_sprite_rides_the_robot_between_pick_and_place(plan_history):
    """The pick window is 10-14 and the place window 22-28; the ride is 14-22."""
    timeline = build_timelines(
        history=plan_history, entity_positions={}, selected=["mug"], env_coords=COORDS,
    )["mug"]
    times = np.array([10.0, 12.0, 14.0, 18.0, 22.0, 25.0, 28.0])
    pos, alpha = sample(timeline, times, _robot(times), fade=2.0)

    assert pos[0] == pytest.approx(COORDS["shelf"])
    assert 0.0 < pos[1][1] < 2.0                       # lifting off the shelf
    assert pos[2] == pytest.approx((10.0, 2.0))        # in hand, robot still at shelf
    assert pos[3] == pytest.approx(_robot(times)["r1"][3])   # carried across
    assert pos[4] == pytest.approx((20.0, 2.0))        # arrived, place dispatched
    assert 0.0 < pos[5][1] < 2.0                       # setting down
    assert pos[6] == pytest.approx(COORDS["counter"])
    assert alpha[1:] == pytest.approx(np.ones(6))


def _robot(times):
    """r1 holds at the shelf, drives 14->22, then holds at the counter."""
    times = np.asarray(times, dtype=float)
    xs = np.interp(times, [0.0, 14.0, 22.0, 40.0], [10.0, 10.0, 20.0, 20.0])
    return {"r1": np.column_stack([xs, np.full_like(xs, 2.0)])}
