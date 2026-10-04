"""Tests for the concurrency-aware heuristic (heuristic_concurrent.hpp).

The scenarios are small multi-robot fetch problems on a line of locations, so
every expected value can be worked out by hand from the move times.
"""

import math

import pytest

from railroad import operators
from railroad._bindings import concurrent_heuristic
from railroad.core import (
    Fluent,
    State,
    convert_goal_to_positive_preconditions,
    convert_state_to_positive_preconditions,
    get_action_by_name,
    project_state,
    transition,
)
from railroad.planner import MCTSPlanner, _normalize_goal

F = Fluent

# Locations on a line; a move costs the distance between them.
POSITION = {"start": 0.0, "a": 10.0, "b": 30.0, "c": -40.0, "goal": 20.0}


def _move_time(robot, frm, to):
    del robot
    return abs(POSITION[frm] - POSITION[to])


def _actions(robots, objects, find_prob=None, search_time=5.0, pick_time=2.0):
    """Ground move / pick / place (and optionally search) for the line world."""
    objects_by_type = {
        "robot": set(robots),
        "location": set(POSITION),
        "object": set(objects),
    }
    ops = [
        operators.construct_move_operator(_move_time),
        operators.construct_pick_operator(pick_time),
        operators.construct_place_operator(pick_time),
    ]
    if find_prob is not None:
        ops.append(operators.construct_search_operator(find_prob, search_time))
    actions = []
    for op in ops:
        actions.extend(op.instantiate(objects_by_type))
    # Grounding iterates sets, whose order follows Python's per-process string
    # hashing; fix it so MCTS tie-breaking is reproducible.
    return sorted(actions, key=lambda a: a.name)


def _state(robots, extra=(), time=0.0):
    fluents = {F("revealed start")} | set(extra)
    for r, loc in robots.items():
        fluents |= {F(f"at {r} {loc}"), F(f"free {r}")}
    return State(time, fluents, [])


def _breakdown(planner, state, goal, **options):
    """Components of the concurrent heuristic, under the planner's conversion."""
    goal = _normalize_goal(goal)
    planner.heuristic(state, goal)  # builds the mapping and projection
    converted = project_state(
        convert_state_to_positive_preconditions(state, planner._current_mapping),
        planner._relevant,
    )
    g = convert_goal_to_positive_preconditions(goal, planner._current_mapping)
    return concurrent_heuristic(converted, g, planner._search_actions, **options)


def _apply(actions, state, name):
    return transition(state, get_action_by_name(actions, name))[0][0]


def test_single_fetch_is_costed_as_a_route():
    """One robot fetching a known object: the estimate is the exact duration.

    The delete relaxation lets the robot reach `goal` from `start` directly
    (20) instead of from `a` (10), which would undercount the trip by 10.
    """
    actions = _actions(["r1"], ["box"])
    state = _state({"r1": "start"}, extra={F("at box a"), F("found box")})
    goal = F("at box goal")
    planner = MCTSPlanner(actions, heuristic="concurrent")
    d = _breakdown(planner, state, goal)
    # start -> a (10), pick (2), a -> goal (10), place (2)
    assert d["makespan"] == pytest.approx(24.0)
    assert d["value"] == pytest.approx(24.0)  # one goal: sum == max

    # Without chaining, the return leg is costed from `start`: 10+2+20+2.
    planner = MCTSPlanner(actions, heuristic="concurrent")
    teleport = _breakdown(planner, state, goal, route_chaining=False)
    assert teleport["makespan"] == pytest.approx(34.0)


def test_robot_already_at_target_must_walk_back():
    """An agent standing at the target still has to return after fetching."""
    actions = _actions(["r1"], ["box"])
    state = _state({"r1": "goal"}, extra={F("at box a"), F("found box")})
    d = _breakdown(MCTSPlanner(actions, heuristic="concurrent"), state, F("at box goal"))
    # goal -> a (10), pick (2), a -> goal (10), place (2)
    assert d["makespan"] == pytest.approx(24.0)


def test_two_robots_split_two_fetches():
    """Two far-apart fetches go to different robots: makespan, not the sum."""
    actions = _actions(["r1", "r2"], ["box", "cup"])
    state = _state(
        {"r1": "start", "r2": "start"},
        extra={F("at box b"), F("at cup c"), F("found box"), F("found cup")},
    )
    goal = F("at box goal") & F("at cup goal")
    d = _breakdown(MCTSPlanner(actions, heuristic="concurrent"), state, goal)
    box = 30 + 2 + 10 + 2   # start -> b -> goal
    cup = 40 + 2 + 60 + 2   # start -> c -> goal
    assert {agent for _, agent in d["assignment"]} == {"r1", "r2"}
    assert d["makespan"] == pytest.approx(max(box, cup))
    assert d["completion_sum"] == pytest.approx(box + cup)


def test_one_robot_does_both_fetches_in_sequence():
    actions = _actions(["r1"], ["box", "cup"])
    state = _state(
        {"r1": "start"},
        extra={F("at box b"), F("at cup c"), F("found box"), F("found cup")},
    )
    goal = F("at box goal") & F("at cup goal")
    d = _breakdown(MCTSPlanner(actions, heuristic="concurrent"), state, goal)
    # The longer fetch first (start -> c -> goal = 104), then goal -> b -> goal.
    assert d["makespan"] == pytest.approx(104 + (10 + 2 + 10 + 2))


def test_committing_to_a_move_is_not_free():
    """Starting the right move must not make the state look cheaper.

    With two robots the state after r1 commits is still at t=0 (r2 is free).
    The FF heuristic's relaxed transition treats r1 as already arrived, so its
    estimate drops by the whole move; the timed relaxation keeps the remaining
    travel time.
    """
    actions = _actions(["r1", "r2"], ["box"])
    state = _state({"r1": "start", "r2": "c"}, extra={F("at box b"), F("found box")})
    goal = F("at box goal")
    conc = MCTSPlanner(actions, heuristic="concurrent")
    ff = MCTSPlanner(actions)

    after = _apply(actions, state, "move r1 start b")
    assert after.time == 0.0
    assert conc.heuristic(after, goal) == pytest.approx(conc.heuristic(state, goal))
    assert ff.heuristic(after, goal) < ff.heuristic(state, goal) - 20


def test_pending_search_is_weighted_not_an_arbitrary_branch():
    """An in-flight search contributes its outcome *probabilistically*.

    The value lies between the two resolved outcomes (found here / not found
    here), rather than equalling whichever branch a hash map lists first.
    """
    actions = _actions(["r1", "r2"], ["box"], find_prob=0.5)
    state = _state({"r1": "a", "r2": "start"}, extra={F("revealed goal")})
    goal = F("at box goal")
    planner = MCTSPlanner(actions, heuristic="concurrent")
    searching = _apply(actions, state, "search r1 a box")
    assert searching.time == 0.0  # r2 is free, so the search is still in flight
    h = planner.heuristic(searching, goal)

    outcomes = transition(searching, get_action_by_name(actions, "move r2 start goal"))
    resolved = sorted(planner.heuristic(s, goal) + s.time for s, _ in outcomes)
    assert len(resolved) == 2
    assert resolved[0] - 1e-6 <= h <= resolved[1] + 1e-6


def test_searches_of_one_location_are_one_attempt():
    """Robots searching the same place are not independent trials.

    Both searches delete `not-searched a box`, so whichever runs first uses up
    location `a`. A second robot standing next to `a` must not shrink the
    expected retry overhead as if it were an independent 50% chance.
    """
    actions_1 = _actions(["r1"], ["box"], find_prob=0.5)
    actions_2 = _actions(["r1", "r2"], ["box"], find_prob=0.5)
    goal = F("found box")
    one = _breakdown(
        MCTSPlanner(actions_1, heuristic="concurrent", agent_aware=False),
        _state({"r1": "a"}), goal)
    two = _breakdown(
        MCTSPlanner(actions_2, heuristic="concurrent", agent_aware=False),
        _state({"r1": "a", "r2": "a"}), goal)
    assert dict(two["deltas"])["found box"] == pytest.approx(dict(one["deltas"])["found box"])

    # Counted as independent trials, the second robot "halves" the failure risk.
    phantom = _breakdown(
        MCTSPlanner(actions_2, heuristic="concurrent", agent_aware=False),
        _state({"r1": "a", "r2": "a"}), goal, group_attempts=False)
    assert dict(phantom["deltas"])["found box"] < dict(one["deltas"])["found box"]


def test_goal_already_true_is_zero_and_unreachable_is_inf():
    actions = _actions(["r1"], ["box"])
    planner = MCTSPlanner(actions, heuristic="concurrent")
    done = _state({"r1": "start"}, extra={F("at box goal"), F("found box")})
    assert planner.heuristic(done, F("at box goal")) == 0.0
    # Nothing can ever reveal where `box` is: no search operator.
    lost = _state({"r1": "start"})
    assert math.isinf(planner.heuristic(lost, F("at box goal")))


@pytest.mark.parametrize(("preferred_first", "backup"),
                         [(False, "mean"), (True, "mean"), (False, "max")])
def test_mcts_with_concurrent_heuristic_completes_two_fetches(preferred_first, backup):
    actions = _actions(["r1", "r2"], ["box", "cup"])
    state = _state(
        {"r1": "start", "r2": "start"},
        extra={F("at box b"), F("at cup c"), F("found box"), F("found cup")},
    )
    goal = F("at box goal") & F("at cup goal")
    planner = MCTSPlanner(actions, heuristic="concurrent",
                          preferred_first=preferred_first, backup=backup)
    for _ in range(20):
        if goal.evaluate(state.fluents):
            break
        name = planner(state, goal, max_iterations=1000, c=50, max_depth=20,
                       heuristic_multiplier=1)
        assert name != "NONE"
        state = _apply(actions, state, name)
    assert goal.evaluate(state.fluents)
    # The work is split: faster than one robot doing both (104 + 24).
    assert state.time < 128


def test_unknown_heuristic_or_backup_is_rejected():
    with pytest.raises(ValueError):
        MCTSPlanner(_actions(["r1"], ["box"]), heuristic="nope")
    with pytest.raises(ValueError):
        MCTSPlanner(_actions(["r1"], ["box"]), backup="median")
