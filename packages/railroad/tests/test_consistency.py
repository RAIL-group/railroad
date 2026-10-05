"""Tests for the heuristic consistency checker (railroad.consistency)."""

import math

import pytest

from railroad import operators
from railroad.consistency import (
    ConsistencyReport,
    check_consistency,
    lookahead,
    record_consistency,
    residual,
)
from railroad.core import Fluent as F, State, get_action_by_name, transition
from railroad.planner import MCTSPlanner

POSITION = {"start": 0.0, "a": 10.0, "b": 30.0, "goal": 20.0}


def _problem(robots, find_prob=None):
    objects_by_type = {"robot": set(robots), "location": set(POSITION), "object": {"box"}}
    ops = [
        operators.construct_move_operator(lambda r, a, b: abs(POSITION[a] - POSITION[b])),
        operators.construct_pick_operator(2.0),
        operators.construct_place_operator(2.0),
    ]
    if find_prob is not None:
        ops.append(operators.construct_search_operator(find_prob, 5.0))
    actions = sorted((a for op in ops for a in op.instantiate(objects_by_type)), key=lambda a: a.name)
    fluents = {F("revealed start"), F("revealed goal")}
    for r in robots:
        fluents |= {F(f"at {r} start"), F(f"free {r}")}
    if find_prob is None:
        fluents |= {F("at box a"), F("found box")}
    return actions, State(0.0, fluents, []), F("at box goal")


def test_lookahead_values_each_action_by_elapsed_time_plus_successor_value():
    actions, state, goal = _problem(["r1"])
    planner = MCTSPlanner(actions)
    for action, q, commit in lookahead(planner, state, goal, actions):
        (succ, prob), = transition(state, action)
        assert prob == 1.0 and not commit
        assert q == pytest.approx(succ.time - state.time + planner.heuristic(succ, goal))


def test_a_fetch_on_a_line_is_consistent_at_every_step():
    """Deterministic, one robot, one object: the estimate is exact, so the
    one-step lookahead never moves it."""
    actions, state, goal = _problem(["r1"])
    report = check_consistency(MCTSPlanner(actions), state, goal, actions, rollouts=3, epsilon=0.0)
    assert report.residuals
    assert all(abs(r.gap) < 1e-6 for r in report.residuals)
    assert report.summary()["consistent"] == 1.0


def test_commit_steps_are_reported_separately():
    actions, state, goal = _problem(["r1", "r2"], find_prob=lambda r, l, o: {"a": 0.6, "b": 0.5}.get(l, 0.0))
    report = check_consistency(MCTSPlanner(actions), state, goal, actions, rollouts=4, depth=12)
    commit, advance = report.summary(commit=True), report.summary(commit=False)
    assert commit["n"] > 0 and advance["n"] > 0
    assert commit["n"] + advance["n"] == report.summary()["n"]
    assert "commit" in str(report)


def test_a_dead_end_outcome_is_valued_as_mcts_values_it():
    """Searching the last place may fail for good; MCTS values that leaf at
    h = 0 (no dead_end_penalty), so the search's lookahead value is finite."""
    actions, state, goal = _problem(["r1"], find_prob=lambda r, l, o: 0.5 if l == "a" else 0.0)
    planner = MCTSPlanner(actions)
    at_a = transition(state, get_action_by_name(actions, "move r1 start a"))[0][0]
    q = {a.name: q for a, q, _ in lookahead(planner, at_a, goal, actions)}
    assert math.isfinite(q["search r1 a box"])
    r = residual(planner, at_a, goal, actions)
    assert r is not None and r.action == "search r1 a box"


def test_record_consistency_checks_every_planner_call_and_restores_the_planner():
    actions, state, goal = _problem(["r1"])
    original = MCTSPlanner.__call__
    with record_consistency() as report:
        MCTSPlanner(actions)(state, goal, max_iterations=50)
        MCTSPlanner(actions)(state, goal, max_iterations=50)
    assert isinstance(report, ConsistencyReport)
    assert len(report.residuals) == 2
    assert MCTSPlanner.__call__ is original


def test_charging_flowtime_makes_two_deliveries_consistent():
    """With two goal tasks open the value falls at 1.5x the rate of time (its
    sum term counts each open task); charging MCTS that same rate
    (flowtime_objective) makes the one-step lookahead agree with it exactly."""
    objects_by_type = {"robot": {"r1"}, "location": set(POSITION), "object": {"box", "cup"}}
    ops = [
        operators.construct_move_operator(lambda r, a, b: abs(POSITION[a] - POSITION[b])),
        operators.construct_pick_operator(2.0),
        operators.construct_place_operator(2.0),
    ]
    actions = sorted((a for op in ops for a in op.instantiate(objects_by_type)), key=lambda a: a.name)
    state = State(0.0, {F("at r1 start"), F("free r1"), F("at box a"), F("found box"),
                        F("at cup b"), F("found cup")}, [])
    goal = F("at box goal") & F("at cup goal")

    time_only = check_consistency(MCTSPlanner(actions), state, goal, actions, rollouts=1, epsilon=0.0)
    first = time_only.residuals[0]
    assert first.gap == pytest.approx(-0.5 * 10.0)  # the first move takes 10 s

    flowtime = MCTSPlanner(actions, heuristic_options={"flowtime_objective": True})
    report = check_consistency(flowtime, state, goal, actions, rollouts=1, epsilon=0.0)
    assert report.residuals
    assert all(abs(r.gap) < 1e-6 for r in report.residuals)
