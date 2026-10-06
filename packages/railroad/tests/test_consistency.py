"""Tests for the heuristic consistency checker (railroad.consistency)."""

import math

import pytest

from railroad import operators
from railroad.consistency import check_consistency, lookahead, record_consistency, residual
from railroad.core import Fluent as F, State, get_action_by_name, transition
from railroad.planner import MCTSPlanner

POSITION = {"start": 0.0, "a": 10.0, "b": 30.0, "goal": 20.0}


def _problem(objects, find_prob=None):
    """One robot at `start`; known objects wait at `a`, `b`, ... in order."""
    objects_by_type = {"robot": {"r1"}, "location": set(POSITION), "object": set(objects)}
    ops = [operators.construct_move_operator(lambda r, a, b: abs(POSITION[a] - POSITION[b])),
           operators.construct_pick_operator(2.0),
           operators.construct_place_operator(2.0)]
    fluents = {F("revealed start"), F("revealed goal"), F("at r1 start"), F("free r1")}
    if find_prob is None:
        for obj, loc in zip(objects, ("a", "b")):
            fluents |= {F(f"at {obj} {loc}"), F(f"found {obj}")}
    else:
        ops.append(operators.construct_search_operator(find_prob, 5.0))
    actions = sorted((a for op in ops for a in op.instantiate(objects_by_type)), key=lambda a: a.name)
    goal = F(f"at {objects[0]} goal")
    for obj in objects[1:]:
        goal = goal & F(f"at {obj} goal")
    return actions, State(0.0, fluents, []), goal


def test_a_fetch_is_consistent_at_every_step():
    """One robot, one known object: the estimate is exact, so one step of
    lookahead never moves it."""
    actions, state, goal = _problem(["box"])
    report = check_consistency(MCTSPlanner(actions), state, goal, actions, rollouts=3, epsilon=0.0)
    assert report.residuals and all(abs(r.gap) < 1e-6 for r in report.residuals)
    assert report.summary()["consistent"] == 1.0
    assert "advance" in str(report)


def test_the_sum_term_makes_the_value_fall_faster_than_time():
    """The value's sum of completion times is a shaping term: with two goals
    open it falls at 1.5x the rate of time, so lookahead finds the first 10 s
    move 5 s cheaper than the value implied; with one goal left it is exact."""
    actions, state, goal = _problem(["box", "cup"])
    report = check_consistency(MCTSPlanner(actions), state, goal, actions, rollouts=1, epsilon=0.0)
    assert report.residuals[0].gap == pytest.approx(-0.5 * 10.0)
    assert report.residuals[-1].gap == pytest.approx(0.0)


def test_a_dead_end_outcome_is_valued_as_mcts_values_it():
    """Searching the last place may fail for good; MCTS values that leaf at
    h = 0 (no dead_end_penalty), so the search's lookahead value is finite."""
    actions, state, goal = _problem(["box"], find_prob=lambda r, l, o: 0.5 if l == "a" else 0.0)
    planner = MCTSPlanner(actions)
    at_a = transition(state, get_action_by_name(actions, "move r1 start a"))[0][0]
    q = {a.name: q for a, q, _ in lookahead(planner, at_a, goal, actions)}
    assert math.isfinite(q["search r1 a box"])
    r = residual(planner, at_a, goal, actions)
    assert r is not None and r.action == "search r1 a box"


def test_record_consistency_checks_every_planner_call_and_restores_the_planner():
    actions, state, goal = _problem(["box"])
    original = MCTSPlanner.__call__
    with record_consistency() as report:
        MCTSPlanner(actions)(state, goal, max_iterations=50)
        MCTSPlanner(actions)(state, goal, max_iterations=50)
    assert len(report.residuals) == 2
    assert MCTSPlanner.__call__ is original
