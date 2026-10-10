"""Tests for the concurrency-aware heuristic (heuristic_concurrent.hpp).

The scenarios are small fetch problems on a line of locations, so every
expected value can be worked out by hand from the move times.
"""

import math

import pytest

from railroad import operators
from railroad.core import Effect, Fluent, Operator, State, get_action_by_name, transition
from railroad.planner import MCTSPlanner

F = Fluent

# Locations on a line; a move costs the distance between them.
POSITION = {"start": 0.0, "a": 10.0, "b": 30.0, "c": -40.0, "goal": 20.0}


def _move_time(robot, frm, to):
    del robot
    return abs(POSITION[frm] - POSITION[to])


def _actions(robots, objects, find_prob=None, blocking=False, search_time=5.0):
    """Ground move / pick / place (and optionally search) for the line world."""
    objects_by_type = {"robot": set(robots), "location": set(POSITION), "object": set(objects)}
    if blocking:
        ops = [operators.construct_move_operator_blocking(_move_time),
               operators.construct_pick_operator_blocking(2.0),
               operators.construct_place_operator_blocking(2.0)]
    else:
        ops = [operators.construct_move_operator(_move_time),
               operators.construct_pick_operator(2.0),
               operators.construct_place_operator(2.0)]
    if find_prob is not None:
        ops.append(operators.construct_search_operator(find_prob, search_time))
    # Grounding iterates sets, whose order follows Python's per-process string
    # hashing; fix it so MCTS tie-breaking is reproducible.
    return sorted((a for op in ops for a in op.instantiate(objects_by_type)), key=lambda a: a.name)


def _state(robots, extra=()):
    fluents = {F("revealed start")} | set(extra)
    for r, loc in robots.items():
        fluents |= {F(f"at {r} {loc}"), F(f"free {r}")}
    return State(0.0, fluents, [])


def _breakdown(actions, state, goal, **options):
    """The heuristic's schedule, with the given switches changed."""
    return MCTSPlanner(actions, heuristic_options=options).heuristic_breakdown(state, goal)


def _apply(actions, state, name):
    return transition(state, get_action_by_name(actions, name))[0][0]


KNOWN = {F("at box b"), F("at cup c"), F("found box"), F("found cup")}


@pytest.mark.parametrize("start, options, expected", [
    ("start", {}, 10 + 2 + 10 + 2),                         # start -> a -> goal
    ("goal", {}, 10 + 2 + 10 + 2),                          # must walk back from a
    ("start", {"route_chaining": False}, 10 + 2 + 20 + 2),  # relaxed: start -> goal
])
def test_a_fetch_is_costed_as_one_route(start, options, expected):
    """A delete relaxation reaches every place from where the robot starts;
    route chaining visits them in order."""
    actions = _actions(["r1"], ["box"])
    state = _state({"r1": start}, extra={F("at box a"), F("found box")})
    d = _breakdown(actions, state, F("at box goal"), **options)
    assert d["makespan"] == pytest.approx(expected)
    assert d["value"] == pytest.approx(expected)  # one goal: sum == max


def _timed(name, params, pre, adds, duration, dels=()):
    """An operator that holds its agent (the first parameter) for `duration`."""
    agent = params[0][0]
    return Operator(name=name, parameters=params, preconditions=pre,
                    effects=[Effect(time=0, resulting_fluents={F(f"not free {agent}"), *dels}),
                             Effect(time=duration, resulting_fluents={F(f"free {agent}"), *adds})])


def test_a_route_does_last_what_uses_up_another_actions_precondition():
    """Serving needs the egg boiled (at the pot, b) and in the bowl (at the
    goal). The relaxation has no deletes, so it may put the egg in the bowl
    (10 from a) before boiling it (20 from a); but putting it in ends holding
    it, which boiling needs. The route boils first: pick the egg up at a (2),
    boil it at b (20 + 10), put it in at the goal (10 + 2), serve (1): 45 --
    not 2 + 10 + 2 + 10 + 10 + 1 = 35."""
    objects = {"robot": {"r1"}, "location": set(POSITION), "object": {"egg"}}
    holds = [F("free ?r"), F("holding ?r egg")]
    ops = [operators.construct_move_operator(_move_time), operators.construct_pick_operator(2.0),
           _timed("boil", [("?r", "robot"), ("?l", "location")],
                  [F("at ?r ?l"), F("at pot ?l"), *holds], [F("boiled egg")], 10.0),
           _timed("put-in", [("?r", "robot"), ("?l", "location")],
                  [F("at ?r ?l"), F("at bowl ?l"), *holds], [F("in egg bowl"), F("not hand-full ?r")],
                  2.0, dels={F("not holding ?r egg")}),
           _timed("serve", [("?r", "robot")], [F("free ?r"), F("boiled egg"), F("in egg bowl")],
                  [F("served")], 1.0)]
    actions = sorted((a for op in ops for a in op.instantiate(objects)), key=lambda a: a.name)
    state = _state({"r1": "a"}, extra={F("at egg a"), F("found egg"), F("at pot b"), F("at bowl goal")})
    assert _breakdown(actions, state, F("served"))["makespan"] == pytest.approx(45.0)


def test_fetches_are_split_across_robots_and_sequenced_on_one():
    goal = F("at box goal") & F("at cup goal")
    box, cup = 30 + 2 + 10 + 2, 40 + 2 + 60 + 2  # start -> b -> goal, start -> c -> goal
    two = _breakdown(_actions(["r1", "r2"], ["box", "cup"]),
                     _state({"r1": "start", "r2": "start"}, KNOWN), goal)
    assert {agent for _, agent in two["assignment"]} == {"r1", "r2"}
    assert two["makespan"] == pytest.approx(max(box, cup))
    assert two["completion_sum"] == pytest.approx(box + cup)
    one = _breakdown(_actions(["r1"], ["box", "cup"]), _state({"r1": "start"}, KNOWN), goal)
    assert one["makespan"] == pytest.approx(cup + (10 + 2 + 10 + 2))  # then goal -> b -> goal


@pytest.mark.parametrize("probs", [None, {"a": 0.5, "b": 0.3}, {"a": 0.8, "b": 0.1},
                                   {"a": 0.3, "b": 0.6, "c": 0.4}, {"a": 0.9}])
def test_starting_the_planned_action_does_not_move_the_estimate(probs):
    """With r2 free, r1's action is still in flight after it starts (t = 0).

    A move counts until it arrives (the FF heuristic's relaxed transition
    treats it as done). A search r1's expected route starts with is the same
    attempt once in flight -- even the last place left -- and what follows it
    happens wherever the box turns up.
    """
    goal = F("at box goal")
    if probs is None:
        actions = _actions(["r1", "r2"], ["box"])
        state = _state({"r1": "start", "r2": "c"}, extra={F("at box b"), F("found box")})
        name = "move r1 start b"
    else:
        actions = _actions(["r1", "r2"], ["box"], find_prob=lambda r, l, o: probs.get(l, 0.0))
        state = _state({"r1": "a", "r2": "c"}, extra={F("revealed goal")})
        name = "search r1 a box"
    planner = MCTSPlanner(actions)
    after = _apply(actions, state, name)
    assert after.time == 0.0
    assert planner.heuristic(after, goal) == pytest.approx(planner.heuristic(state, goal))


@pytest.mark.parametrize("probs, revealed, expected", [
    # Greedy route: a (done at 15), then b (done at 40): 15 + 0.5 * 25 = 27.5.
    # The box turns up with probability 0.75, and delivery from either place
    # takes pick + 10 + place = 14: 27.5 + 0.75 * 14 = 38.
    ({"a": 0.5, "b": 0.5}, {"goal", "c"}, 38.0),
    # Searching `goal` would find the box already delivered. Route: goal (done
    # at 25), then b (done at 40): 25 + 0.5 * 15 = 32.5; only the box found at
    # b (probability 0.25) needs delivering: 32.5 + 0.25 * 14 = 36.
    ({"goal": 0.5, "b": 0.5}, {"a", "c"}, 36.0),
])
def test_expected_search_is_a_route_over_the_candidate_places(probs, revealed, expected):
    actions = _actions(["r1"], ["box"], find_prob=lambda r, l, o: probs.get(l, 0.0))
    state = _state({"r1": "start"}, extra={F(f"revealed {p}") for p in revealed})
    assert _breakdown(actions, state, F("at box goal"))["makespan"] == pytest.approx(expected)


@pytest.mark.parametrize("order", [("search r1 a box", "search r2 c box"),
                                   ("search r2 c box", "search r1 a box")])
@pytest.mark.parametrize("c_time, expected", [
    (6.0, 5.5 + 7 + 16),  # r1, ready at 5, waits for `c` with probability 0.5
    (5.0, 5.0 + 7 + 16),  # both end at 5; if both succeed, deliver from `a`
])
def test_in_flight_searches_are_costed_where_they_happen(order, c_time, expected):
    """r1 searches `a` (done at 5) and r2 searches `c` while r3 is free, in
    either order. Delivery from `a` (0.5) takes 2 + 10 + 2 = 14 and from `c`
    alone (0.25) 2 + 60 + 2 = 64, so 7 + 16 after the searches. Costing `c`
    at `a`, or crediting `c` first when both end together, changes it."""
    actions = _actions(["r1", "r2", "r3"], ["box"],
                       find_prob=lambda r, l, o: {"a": 0.5, "c": 0.5}.get(l, 0.0),
                       search_time=lambda r, l, o: c_time if l == "c" else 5.0)
    state = _state({"r1": "a", "r2": "c", "r3": "start"},
                   extra={F("revealed goal"), F("revealed b")})
    for name in order:
        state = _apply(actions, state, name)
    assert state.time == 0.0 and len(state.upcoming_effects) >= 2
    assert _breakdown(actions, state, F("at box goal"))["value"] == pytest.approx(expected)


@pytest.mark.parametrize("probs", [{"a": 0.5, "b": 0.3, "c": 0.4},
                                   {"a": 0.6, "goal": 0.5, "b": 0.1},
                                   {"a": 0.9, "b": 0.2}])
def test_a_search_is_valued_by_its_outcomes(probs):
    """Before the search its expected route starts with, h is the search time
    plus the probability-weighted values of its outcomes: failing a likely
    search must not look better or worse than the estimate counted on."""
    actions = _actions(["r1"], ["box"], find_prob=lambda r, l, o: probs.get(l, 0.0))
    state = _state({"r1": "a"})
    goal = F("at box goal")
    planner = MCTSPlanner(actions)
    outcomes = transition(state, get_action_by_name(actions, "search r1 a box"))
    expected = sum(pr * (s.time + planner.heuristic(s, goal)) for s, pr in outcomes)
    assert planner.heuristic(state, goal) == pytest.approx(expected)


@pytest.mark.parametrize("robot", ["r1", "r2", "r3", "robot1", "robot2", "robot3"])
def test_the_value_does_not_depend_on_the_robots_name(robot):
    """The box and the cup may both be at `a`. Holding the box and filling the
    hand then tie in cost (17), and the cup, likelier there, is the better way
    to fill the hand by c / rho. Extracted first, `hand-full` pulled picking up
    the cup into the box's plan, and the fluents' hash order decided which came
    first, so the value depended on the robot's name. Search `a` (done at 15);
    with probability 0.5 the box is there: pick, move 10, place = 14.
    15 + 0.5 * 14 = 22."""
    actions = _actions([robot], ["box", "cup"],
                       find_prob=lambda r, l, o: {("a", "box"): 0.5, ("a", "cup"): 0.8}.get((l, o), 0.0))
    state = _state({robot: "start"})
    assert MCTSPlanner(actions).heuristic(state, F("at box goal")) == pytest.approx(22.0)


def test_single_precision_find_probabilities_are_handled_like_doubles():
    """Learned estimators return float32, and `1 - p` rounds in float32, so a
    search's branches sum to just under one. What every branch adds (the robot
    free again) must still be certain, or the delivery leg disappears."""
    np = pytest.importorskip("numpy")

    def makespan(cast):
        probs = {"a": 0.455, "b": 0.455}
        actions = _actions(["r1"], ["box"], find_prob=lambda r, l, o: cast(probs.get(l, 0.0)))
        state = _state({"r1": "a"}, extra={F("revealed goal"), F("revealed c")})
        return _breakdown(actions, state, F("at box goal"))["makespan"]

    assert makespan(np.float32) == pytest.approx(makespan(float), rel=1e-5)


def test_a_free_robot_cannot_idle_until_its_own_flag_clears():
    """Blocking operators clear a robot's flag 0.1 s after each action, so it
    cannot put an object straight back. A free robot must act now, so it can
    place the box only after another action (a move of 10), not after 0.1 s:
    10.1 + 2.1, against 2.1 when pending effects count from time 0."""
    actions = _actions(["r1"], ["box"], blocking=True)
    state = _state({"r1": "goal"}, extra={F("at box goal"), F("found box")})
    held = _apply(actions, state, "pick r1 goal box")
    assert F("free r1") in held.fluents and held.upcoming_effects
    assert MCTSPlanner(actions).heuristic(held, F("at box goal")) == pytest.approx(10.1 + 2.1)
    assert _breakdown(actions, held, F("at box goal"), timed_init=False)["makespan"] == pytest.approx(2.1)


def test_goal_already_true_is_zero_and_unreachable_is_inf():
    planner = MCTSPlanner(_actions(["r1"], ["box"]))
    done = _state({"r1": "start"}, extra={F("at box goal"), F("found box")})
    assert planner.heuristic(done, F("at box goal")) == 0.0
    lost = _state({"r1": "start"})  # nothing can find the box: no search operator
    assert math.isinf(planner.heuristic(lost, F("at box goal")))


@pytest.mark.parametrize("backup", ["mean", "max"])
def test_mcts_splits_two_fetches_between_two_robots(backup):
    actions = _actions(["r1", "r2"], ["box", "cup"])
    state = _state({"r1": "start", "r2": "start"}, KNOWN)
    goal = F("at box goal") & F("at cup goal")
    planner = MCTSPlanner(actions, backup=backup)
    for _ in range(40):  # a robot with slack may fill it with short actions
        if goal.evaluate(state.fluents):
            break
        name = planner(state, goal, max_iterations=1000, c=50, max_depth=20)
        assert name != "NONE"
        state = _apply(actions, state, name)
    assert goal.evaluate(state.fluents)
    assert state.time < 104 + 24  # faster than one robot doing both


def test_max_backup_values_a_rarely_visited_outcome_by_more_than_one_child():
    """A node keeps its own value until each of its actions has been tried.

    The robot stands at `x`, where the box is with probability 0.8; `y` (0.5)
    is 30 away. Expansion takes actions from the back of the list, ordered so
    that the first action tried after the search succeeds is a long detour to
    `a`. Valued by that one child, searching here looks worse than walking to `y`.
    """
    near = {("x", "goal"): 10.0, ("x", "y"): 30.0, ("y", "a"): 5.0, ("y", "goal"): 32.0}

    def move_time(robot, frm, to):
        return 0.0 if frm == to else near.get((frm, to), near.get((to, frm), 500.0))

    objects_by_type = {"robot": {"r1"}, "location": {"x", "y", "a", "goal", "start"},
                       "object": {"box"}}
    ops = [operators.construct_move_operator(move_time),
           operators.construct_pick_operator(2.0),
           operators.construct_place_operator(2.0),
           operators.construct_search_operator(lambda r, l, o: {"x": 0.8, "y": 0.5}.get(l, 0.0), 5.0)]
    actions = sorted((a for op in ops for a in op.instantiate(objects_by_type)),
                     key=lambda a: a.name, reverse=True)
    state = State(0.0, {F("revealed goal"), F("at r1 x"), F("free r1")}, [])
    name = MCTSPlanner(actions)(state, F("at box goal"), max_iterations=200, c=50, max_depth=20)
    assert name == "search r1 x box"


@pytest.mark.parametrize("backup", ["mean", "max"])
def test_mcts_does_not_expand_past_the_goal(backup):
    """A goal state ends the episode; expanded, it was valued by continuations
    past the goal, which under max backup made reaching it look worse."""
    actions = _actions(["r1"], ["box"])
    state = _state({"r1": "goal"}, extra={F("holding r1 box"), F("hand-full r1"), F("found box")})
    planner = MCTSPlanner(actions, backup=backup)
    assert planner(state, F("at box goal"), max_iterations=300, c=50, max_depth=20) == "place r1 goal box"
    trace = planner.get_trace_from_last_mcts_tree()
    goal_node = next(line for line in trace.splitlines() if "D:1|" in line)
    assert goal_node.rstrip().endswith("#A=0")


@pytest.mark.parametrize("kwargs", [
    {"heuristic": "nope"},
    {"backup": "median"},
    {"heuristic_options": {"agent_awareness": False}},             # misspelled
    {"heuristic_options": {"lambda_add": 1.0}},                    # a planner argument
    {"heuristic": "ff", "heuristic_options": {"agent_aware": False}},  # concurrent only
])
def test_invalid_planner_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MCTSPlanner(_actions(["r1"], ["box"]), **kwargs)


# -- Jobs: one goal's work shared between agents --------------------------------

def test_a_job_waits_for_what_another_agent_provides():
    """Lighting the lamp needs the power on, which another agent can switch on
    at the panel: one walks to the panel (30) and switches it on (5) while the
    other walks to the lamp (10), waits for the power (35) and lights it (5):
    40. On its own an agent goes on from the panel: 35 + 20 + 5 = 60."""
    where = {"start": 0.0, "panel": 30.0, "lamp": 10.0}
    ops = [operators.construct_move_operator(lambda r, a, b: abs(where[a] - where[b])),
           _timed("switch-on", [("?r", "robot")], [F("at ?r panel"), F("free ?r")], [F("powered")], 5.0),
           _timed("light", [("?r", "robot")], [F("at ?r lamp"), F("free ?r"), F("powered")], [F("lit")], 5.0)]

    def value(robots):
        objects = {"robot": set(robots), "location": set(where)}
        actions = sorted((a for op in ops for a in op.instantiate(objects)), key=lambda a: a.name)
        state = State(0.0, {F(f"at {r} start") for r in robots} | {F(f"free {r}") for r in robots}, [])
        return MCTSPlanner(actions).heuristic(state, F("lit"))

    assert value(["r1", "r2"]) == pytest.approx(40.0)
    assert value(["r1"]) == pytest.approx(60.0)


def _dish_actions(robots):
    """Line world with boiling and serving: the egg at `a`, the pot at `b`, the
    bowl at `c`, all already found; serve at `goal`."""
    objects = {"robot": set(robots), "location": set(POSITION), "object": {"egg", "pot", "bowl"},
               "egg": {"egg"}, "vessel": {"pot"}, "bowl": {"bowl"}}
    ops = [operators.construct_move_operator(_move_time),
           operators.construct_pick_operator(2.0), operators.construct_place_operator(2.0),
           _timed("boil", [("?r", "robot"), ("?l", "location"), ("?e", "egg"), ("?v", "vessel")],
                  [F("at ?r ?l"), F("free ?r"), F("holding ?r ?e"), F("at ?v ?l")], [F("boiled ?e")], 10.0),
           _timed("put-in", [("?r", "robot"), ("?l", "location"), ("?e", "egg"), ("?b", "bowl")],
                  [F("at ?r ?l"), F("free ?r"), F("holding ?r ?e"), F("at ?b ?l")],
                  [F("in ?e ?b"), F("not hand-full ?r")], 2.0, dels={F("not holding ?r ?e")})]
    return sorted((a for op in ops for a in op.instantiate(objects)), key=lambda a: a.name)


def test_a_dish_is_shared_between_two_robots():
    """Boiling the egg and putting it in the bowl are one robot's job (it holds
    the egg throughout); bringing the bowl to the goal is another's, which the
    first waits for. Bowl: 40 + 2 + 60 + 2 = 104. Egg: 10 + 2 to pick it up,
    20 + 10 to boil it at the pot, 10 to the goal (52), the bowl there at 104,
    then 2 to put it in: 106. h = 0.5 * (104 + 106 + 106) + 0.5 * 106."""
    robots = ["r1", "r2"]
    state = _state({r: "start" for r in robots},
                   extra={F("at egg a"), F("at pot b"), F("at bowl c"),
                          F("found egg"), F("found pot"), F("found bowl")})
    goal = F("boiled egg") & F("in egg bowl") & F("at bowl goal")
    d = MCTSPlanner(_dish_actions(robots)).heuristic_breakdown(state, goal)
    finish = dict(d["goal_finish"])
    assert finish["at bowl goal"] == pytest.approx(104.0)
    assert finish["in egg bowl"] == pytest.approx(106.0)
    assert d["makespan"] == pytest.approx(106.0)
    assert d["value"] == pytest.approx(0.5 * (104 + 106 + 106) + 0.5 * 106)
    assigned = dict(d["assignment"])
    assert assigned["boiled egg"] == assigned["in egg bowl"] != assigned["at bowl goal"]


@pytest.mark.parametrize("held, makespan", [(False, 44.0), (True, 42.0)])
def test_the_egg_is_boiled_before_it_goes_in_the_bowl(held, makespan):
    """The relaxation has no deletes, so it may put the egg in the bowl (at the
    goal, 10 from a) before boiling it (at b, 20 from a). Putting it in ends
    holding it, which boiling needs, so that comes last: pick the egg up (2),
    boil it at b (20 + 10), put it in at the goal (10 + 2): 44, or 42 when it
    is already held. Either way both facts are one job, for whoever holds it."""
    extra = {F("at pot b"), F("at bowl goal"), F("found egg"), F("found pot"), F("found bowl")}
    extra |= {F("holding r1 egg"), F("hand-full r1")} if held else {F("at egg a")}
    state = _state({"r1": "a", "r2": "c"}, extra=extra)
    d = MCTSPlanner(_dish_actions(["r1", "r2"])).heuristic_breakdown(state, F("boiled egg") & F("in egg bowl"))
    finish = dict(d["goal_finish"])
    assert finish["boiled egg"] == finish["in egg bowl"] == pytest.approx(makespan)
    assert dict(d["assignment"]) == {"boiled egg": "r1", "in egg bowl": "r1"}


def test_a_provider_leaves_the_robot_its_consumer_needs():
    """Bringing the bowl to the goal is a job the egg job waits for. r1, at the
    bowl, would bring it soonest (2 + 20 + 2 = 24), but then the egg job, for
    r1 or for r2 far off at c, ends at 78. Looking ahead, the bowl goes to r2
    (40 + 2 + 20 + 2 = 64) while r1 picks up the egg (10 + 2), boils it at b
    (20 + 10) and waits for the bowl at the goal (10, then 64): 66."""
    extra = {F("at egg a"), F("at pot b"), F("at bowl start"), F("found egg"), F("found pot"), F("found bowl")}
    state = _state({"r1": "start", "r2": "c"}, extra=extra)
    goal = F("boiled egg") & F("in egg bowl") & F("at bowl goal")
    d = MCTSPlanner(_dish_actions(["r1", "r2"])).heuristic_breakdown(state, goal)
    assert dict(d["assignment"]) == {"boiled egg": "r1", "in egg bowl": "r1", "at bowl goal": "r2"}
    assert d["makespan"] == pytest.approx(66.0)


@pytest.mark.parametrize("robots, makespan", [(["r1"], 58.0), (["r1", "r2"], 46.0)])
def test_a_bowl_is_filled_where_it_is_or_where_it_goes(robots, makespan):
    """The egg and the pot are at a, the bowl at b, past the goal. Alone, a
    robot boils the egg (10 + 2 + 10), puts it into the bowl at b (20 + 2) and
    carries the bowl to the goal (2 + 10 + 2): 58 -- not the bowl first (44)
    and then the egg (10 + 2 + 10 + 10 + 2): 78. With two, one brings the bowl
    (44) while the other boils the egg and waits for it at the goal: 46."""
    extra = {F("at egg a"), F("at pot a"), F("at bowl b"), F("found egg"), F("found pot"), F("found bowl")}
    state = _state({r: "start" for r in robots}, extra=extra)
    goal = F("boiled egg") & F("in egg bowl") & F("at bowl goal")
    d = MCTSPlanner(_dish_actions(robots)).heuristic_breakdown(state, goal)
    assert d["makespan"] == pytest.approx(makespan)


def test_a_spare_robot_hedges_on_a_second_way():
    """Either object at the goal will do. Each is at one of two places with
    probability 0.5. One robot brings the box; the other, with nothing to do
    for that, goes for the cup, which shortens the expected time. With one
    robot there is no one to spare."""
    probs = {("a", "box"): 0.5, ("b", "box"): 0.5, ("c", "cup"): 0.5, ("goal", "cup"): 0.5}
    goal = F("at box goal") | F("at cup goal")
    two = MCTSPlanner(_actions(["r1", "r2"], ["box", "cup"], find_prob=lambda r, l, o: probs.get((l, o), 0.0)))
    d = two.heuristic_breakdown(_state({"r1": "start", "r2": "start"}), goal)
    assert d["hedge_gain"] > 0.0 and d["hedge"]
    assert d["value"] == pytest.approx(0.5 * d["completion_sum"] + 0.5 * (d["makespan"] - d["hedge_gain"]))
    one = MCTSPlanner(_actions(["r1"], ["box", "cup"], find_prob=lambda r, l, o: probs.get((l, o), 0.0)))
    assert one.heuristic_breakdown(_state({"r1": "start"}), goal)["hedge_gain"] == 0.0
