import pytest

from interruption.planning_framework import ap_heuristic_fn

from railroad.core import Fluent as F, Goal, LiteralGoal, State
from railroad.operators.core import construct_move_operator, construct_pick_operator


def _build_move_and_pick_fixture() -> tuple[State, Goal, list]:
    """
    Minimal move-then-pick problem: robot r1 starts at "start", box is at
    "target". Reaching "holding r1 box" costs move (5.0) + pick (2.0) = 7.0,
    the same fixture shape used by railroad's own test_ff_heuristic_move_and_pick,
    rebuilt here with the real railroad.operators.core operators (matching this
    package's existing test_astar_planner.py convention).
    """
    move_op = construct_move_operator(5.0)
    pick_op = construct_pick_operator(2.0)
    objects_by_type = {
        "robot": ["r1"], "location": ["start", "target"], "object": ["box"],
    }
    actions = [*move_op.instantiate(objects_by_type), *pick_op.instantiate(objects_by_type)]
    state = State(time=0, fluents={F("at r1 start"), F("free r1"), F("at box target")})
    goal = LiteralGoal(F("holding r1 box"))
    return state, goal, actions


def test_ap_heuristic_fn_without_v_ap_matches_ff_heuristic():
    """include_v_ap=False must ignore v_ap entirely, not just default it to 0."""
    state, goal, actions = _build_move_and_pick_fixture()

    assert ap_heuristic_fn(False, state, goal, actions, v_ap=1000.0) == 7.0


def test_ap_heuristic_fn_with_v_ap_adds_expected_value():
    state, goal, actions = _build_move_and_pick_fixture()

    assert ap_heuristic_fn(True, state, goal, actions, v_ap=3.5) == pytest.approx(10.5)


@pytest.mark.parametrize("v_ap", [0.0, -2.0, 100.0])
def test_ap_heuristic_fn_with_v_ap_is_additive(v_ap):
    """
    Guards against a future change accidentally clamping or scaling v_ap
    instead of a plain addition -- nothing in ap_heuristic_fn currently
    prevents the result from going negative when v_ap is negative enough.
    """
    state, goal, actions = _build_move_and_pick_fixture()
    base = ap_heuristic_fn(False, state, goal, actions, v_ap=0.0)

    assert ap_heuristic_fn(True, state, goal, actions, v_ap=v_ap) == pytest.approx(base + v_ap)


def test_ap_heuristic_fn_with_v_ap_true_and_zero_matches_without_v_ap():
    state, goal, actions = _build_move_and_pick_fixture()

    without = ap_heuristic_fn(False, state, goal, actions, v_ap=0.0)
    with_zero = ap_heuristic_fn(True, state, goal, actions, v_ap=0.0)

    assert with_zero == without
