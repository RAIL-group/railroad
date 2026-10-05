"""Tests for the concurrency-aware heuristic (heuristic_concurrent.hpp).

The scenarios are small multi-robot fetch problems on a line of locations, so
every expected value can be worked out by hand from the move times.
"""

import math
from typing import Any, cast

import pytest

from railroad import operators
from railroad.core import (
    Fluent,
    State,
    get_action_by_name,
    transition,
)
from railroad.planner import MCTSPlanner

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
    """Components of the concurrent heuristic, optionally with switches changed."""
    if options:
        planner = MCTSPlanner(planner._original_actions,
                              heuristic_options={**planner._heuristic_options, **options})
    return planner.heuristic_breakdown(state, goal)


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
    ff = MCTSPlanner(actions, heuristic="ff")

    after = _apply(actions, state, "move r1 start b")
    assert after.time == 0.0
    assert conc.heuristic(after, goal) == pytest.approx(conc.heuristic(state, goal))
    assert ff.heuristic(after, goal) < ff.heuristic(state, goal) - 20


@pytest.mark.parametrize("probs", [{"a": 0.5, "b": 0.3},
                                   {"a": 0.8, "b": 0.1},
                                   {"a": 0.3, "b": 0.6, "c": 0.4},
                                   {"a": 0.9}])
def test_starting_the_planned_search_does_not_move_the_estimate(probs):
    """A search the estimate already counts on costs the same once started.

    r1 stands at `a`, the first place its expected search visits; r2 is free,
    so after r1 commits the search is in flight at the same time. Planned and
    in-flight attempts are the same attempt: the in-flight one is the task's
    search (even when it is the last place left), and the pick that follows
    it happens wherever the box turns up.
    """
    actions = _actions(["r1", "r2"], ["box"], find_prob=lambda r, l, o: probs.get(l, 0.0))
    state = _state({"r1": "a", "r2": "c"}, extra={F("revealed goal")})
    goal = F("at box goal")
    planner = MCTSPlanner(actions)
    searching = _apply(actions, state, "search r1 a box")
    assert searching.time == 0.0
    assert planner.heuristic(searching, goal) == pytest.approx(planner.heuristic(state, goal))


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
        MCTSPlanner(actions_1, heuristic="concurrent", heuristic_options={"agent_aware": False}),
        _state({"r1": "a"}), goal)
    two = _breakdown(
        MCTSPlanner(actions_2, heuristic="concurrent", heuristic_options={"agent_aware": False}),
        _state({"r1": "a", "r2": "a"}), goal)
    assert dict(two["deltas"])["found box"] == pytest.approx(dict(one["deltas"])["found box"])

    # Counted as independent trials, the second robot "halves" the failure risk.
    phantom = _breakdown(
        MCTSPlanner(actions_2, heuristic="concurrent", heuristic_options={"agent_aware": False}),
        _state({"r1": "a", "r2": "a"}), goal, group_attempts=False)
    assert dict(phantom["deltas"])["found box"] < dict(one["deltas"])["found box"]


def test_expected_search_costs_delivery_unless_found_in_place():
    """Searching the target place can find the object already delivered.

    The box is at `goal` (20 away) or `b` (30 away), each with probability 0.5.
    The relaxation's cheapest support for `at box goal` is to search `goal`,
    whose success needs no delivery; the box may still turn up at `b`, from
    where it must be brought over. The greedy route searches `goal` (done at
    25), then `b` (done at 40): expected search time 25 + 0.5 * 15 = 32.5.
    The box turns up at `b` with probability 0.25, and delivering it from
    there takes pick + 10 + place = 14: 32.5 + 0.25 * 14 = 36.
    """
    def prob(robot, loc, obj):
        return {"goal": 0.5, "b": 0.5}.get(loc, 0.0)

    actions = _actions(["r1"], ["box"], find_prob=prob)
    state = _state({"r1": "start"}, extra={F("revealed a"), F("revealed c")})
    goal = F("at box goal")
    planner = MCTSPlanner(actions, heuristic="concurrent")
    assert _breakdown(planner, state, goal)["makespan"] == pytest.approx(36.0)


@pytest.mark.parametrize("probs", [{"a": 0.5, "b": 0.3, "c": 0.4},
                                   {"a": 0.6, "goal": 0.5, "b": 0.1},
                                   {"a": 0.9, "b": 0.2}])
def test_searching_where_the_route_starts_is_valued_by_its_outcomes(probs):
    """Before a search the expected route starts with, h is the search time
    plus the probability-weighted values of its two outcomes (one robot, one
    object): failing a likely search must not look better or worse than the
    estimate already counted on."""
    def prob(robot, loc, obj):
        return probs.get(loc, 0.0)

    actions = _actions(["r1"], ["box"], find_prob=prob)
    state = _state({"r1": "a"})
    goal = F("at box goal")
    planner = MCTSPlanner(actions, heuristic="concurrent")
    outcomes = transition(state, get_action_by_name(actions, "search r1 a box"))
    expected = sum(pr * (s.time - state.time + planner.heuristic(s, goal)) for s, pr in outcomes)
    assert planner.heuristic(state, goal) == pytest.approx(expected)


def test_single_precision_find_probabilities_are_handled_like_doubles():
    """Learned estimators return float32; `1 - p` then rounds in float32.

    The branches of a search then sum to just under one, which must not make
    what every branch adds (the robot free again) look uncertain -- that
    hid the delivery leg after a search where the robot stands.
    """
    np = pytest.importorskip("numpy")

    def value(cast):
        def prob(robot, loc, obj):
            return cast({"a": 0.455, "b": 0.455}.get(loc, 0.0))
        actions = _actions(["r1"], ["box"], find_prob=prob)
        state = _state({"r1": "a"}, extra={F("revealed goal"), F("revealed c")})
        planner = MCTSPlanner(actions, heuristic="concurrent")
        return _breakdown(planner, state, F("at box goal"))["makespan"]

    assert value(np.float32) == pytest.approx(value(float), rel=1e-5)


def test_a_free_robot_cannot_idle_until_its_own_flag_clears():
    """Blocking operators forbid putting an object straight back: `just-picked`
    clears 0.1 s after the pick. A free robot must act now, so it can place the
    box only after another action -- here a move of 10 -- not after 0.1 s. Each
    blocking action's last effect (clearing its own flag) comes 0.1 s after it
    ends: 10.1 + 2.1, against 2.1 when pending effects count from time 0.
    """
    objects_by_type = {"robot": {"r1"}, "location": set(POSITION), "object": {"box"}}
    ops = [
        operators.construct_move_operator_blocking(_move_time),
        operators.construct_pick_operator_blocking(2.0),
        operators.construct_place_operator_blocking(2.0),
    ]
    actions = sorted((a for op in ops for a in op.instantiate(objects_by_type)),
                     key=lambda a: a.name)
    state = _state({"r1": "goal"}, extra={F("at box goal"), F("found box")})
    held = _apply(actions, state, "pick r1 goal box")
    assert F("free r1") in held.fluents and held.upcoming_effects
    planner = MCTSPlanner(actions, heuristic="concurrent")
    assert planner.heuristic(held, F("at box goal")) == pytest.approx(10.1 + 2.1)
    assert (_breakdown(planner, held, F("at box goal"), timed_init=False)["makespan"]
            == pytest.approx(2.1))


def test_object_in_hand_is_delivered_before_fetching_another():
    """The relaxation frees a full hand by setting its object down anywhere.

    r1 stands at `a` holding the box, next to the cup; both belong at `goal`
    (10 away). Relaxed, the cup's plan drops the box to free the hand, and the
    box is then "placed" at `goal` again for free, since `holding r1 box` is
    never deleted: 16 in all. That plan destroys the holding the box's own
    delivery relies on, so the box goes first (10 + 2) and the cup is fetched
    from `goal` (10 + 2 + 10 + 2): 36, the true optimum.
    """
    actions = _actions(["r1"], ["box", "cup"])
    state = _state({"r1": "a"}, extra={
        F("holding r1 box"), F("hand-full r1"), F("at cup a"),
        F("found box"), F("found cup"), F("revealed a"), F("revealed goal")})
    goal = F("at box goal") & F("at cup goal")
    planner = MCTSPlanner(actions, heuristic="concurrent")
    assert _breakdown(planner, state, goal)["makespan"] == pytest.approx(36.0)
    assert _breakdown(planner, state, goal, order_conflicts=False)["makespan"] == pytest.approx(16.0)


def test_expected_search_walks_a_route_over_candidate_places():
    """An uncertain fetch is costed over where the object might turn up.

    The box is at `a` (10 away) or `b` (30 away), each with probability 0.5.
    The greedy route searches `a` first (done at 15), then `b` (done at 40):
    expected search time 15 + 0.5 * 25 = 27.5. As independent attempts, the
    searches find the box with probability 0.75, and delivery to `goal`
    (pick + 10 + place = 14 from either place) is charged in proportion:
    27.5 + 0.75 * 14 = 38.
    """
    def prob(robot, loc, obj):
        return {"a": 0.5, "b": 0.5}.get(loc, 0.0)

    actions = _actions(["r1"], ["box"], find_prob=prob)
    state = _state({"r1": "start"}, extra={F("revealed goal"), F("revealed c")})
    goal = F("at box goal")
    planner = MCTSPlanner(actions, heuristic="concurrent",
                          heuristic_options={"expected_search": True})
    d = _breakdown(planner, state, goal, expected_search=True)
    assert d["makespan"] == pytest.approx(38.0)


def test_goal_already_true_is_zero_and_unreachable_is_inf():
    actions = _actions(["r1"], ["box"])
    planner = MCTSPlanner(actions, heuristic="concurrent")
    done = _state({"r1": "start"}, extra={F("at box goal"), F("found box")})
    assert planner.heuristic(done, F("at box goal")) == 0.0
    # Nothing can ever reveal where `box` is: no search operator.
    lost = _state({"r1": "start"})
    assert math.isinf(planner.heuristic(lost, F("at box goal")))


@pytest.mark.parametrize("backup", ["mean", "max"])
def test_mcts_with_concurrent_heuristic_completes_two_fetches(backup):
    actions = _actions(["r1", "r2"], ["box", "cup"])
    state = _state(
        {"r1": "start", "r2": "start"},
        extra={F("at box b"), F("at cup c"), F("found box"), F("found cup")},
    )
    goal = F("at box goal") & F("at cup goal")
    planner = MCTSPlanner(actions, heuristic="concurrent", backup=backup)
    # The cup's delivery fixes the finish time, so a robot with slack may fill
    # it with short actions at no cost: allow more decisions than moves.
    for _ in range(40):
        if goal.evaluate(state.fluents):
            break
        name = planner(state, goal, max_iterations=1000, c=50, max_depth=20,
                       heuristic_multiplier=1)
        assert name != "NONE"
        state = _apply(actions, state, name)
    assert goal.evaluate(state.fluents)
    # The work is split: faster than one robot doing both (104 + 24).
    assert state.time < 128


def test_max_backup_values_a_rarely_visited_outcome_by_more_than_one_child():
    """A node keeps its own value until each of its actions has been tried.

    The robot stands at `x`, where the box is with probability 0.8; `y` (0.5)
    is 30 away, and `start` is far from everything. Expansion takes actions
    from the back of the list, ordered here so that the first action tried
    after the search succeeds is a long detour to `a`. If the likely outcome is
    valued by that one child, searching here looks worse than walking to `y`.
    """
    near = {("x", "goal"): 10.0, ("x", "y"): 30.0, ("y", "a"): 5.0, ("y", "goal"): 32.0}

    def move_time(robot, frm, to):
        del robot
        return 0.0 if frm == to else near.get((frm, to), near.get((to, frm), 500.0))

    def prob(robot, loc, obj):
        return {"x": 0.8, "y": 0.5}.get(loc, 0.0)

    objects_by_type = {"robot": {"r1"}, "location": {"x", "y", "a", "goal", "start"},
                       "object": {"box"}}
    ops = [
        operators.construct_move_operator(move_time),
        operators.construct_pick_operator(2.0),
        operators.construct_place_operator(2.0),
        operators.construct_search_operator(prob, 5.0),
    ]
    actions = sorted((a for op in ops for a in op.instantiate(objects_by_type)),
                     key=lambda a: a.name, reverse=True)
    state = State(0.0, {F("revealed goal"), F("at r1 x"), F("free r1")}, [])
    planner = MCTSPlanner(actions, heuristic="concurrent", backup="max",
                          heuristic_options={"expected_search": True})
    name = planner(state, F("at box goal"), max_iterations=200, c=50, max_depth=20,
                   heuristic_multiplier=1)
    assert name == "search r1 x box"


@pytest.mark.parametrize("backup", ["mean", "max"])
def test_mcts_does_not_expand_past_the_goal(backup):
    """A goal state ends the episode. Expanding it valued the goal by
    continuations past it, which under max backup (once every action had been
    tried) made reaching the goal look worse than it is."""
    actions = _actions(["r1"], ["box"])
    state = _state({"r1": "goal"}, extra={F("holding r1 box"), F("hand-full r1"), F("found box")})
    planner = MCTSPlanner(actions, heuristic="concurrent", backup=backup)
    name = planner(state, F("at box goal"), max_iterations=300, c=50, max_depth=20,
                   heuristic_multiplier=1)
    assert name == "place r1 goal box"
    trace = planner.get_trace_from_last_mcts_tree()
    goal_node = next(line for line in trace.splitlines() if "D:1|" in line)
    assert goal_node.rstrip().endswith("#A=0")


def test_unknown_heuristic_or_backup_is_rejected():
    with pytest.raises(ValueError):
        MCTSPlanner(_actions(["r1"], ["box"]), heuristic="nope")
    with pytest.raises(ValueError):
        MCTSPlanner(_actions(["r1"], ["box"]), backup="median")
    misspelled = cast(Any, {"agent_awareness": False})
    with pytest.raises(ValueError):
        MCTSPlanner(_actions(["r1"], ["box"]), heuristic="concurrent",
                    heuristic_options=misspelled)
    with pytest.raises(ValueError):  # options only mean something to "concurrent"
        MCTSPlanner(_actions(["r1"], ["box"]), heuristic="ff",
                    heuristic_options={"agent_aware": False})
