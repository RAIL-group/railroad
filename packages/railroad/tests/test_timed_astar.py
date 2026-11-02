"""timed_astar_search (include/railroad/timed_astar.hpp): A* with clock-dependent costs from Python callbacks."""

from railroad._bindings import timed_astar_search
from railroad.core import Effect, LiteralGoal, Operator, State, transition
from railroad.core import Fluent as F
from railroad.operators import construct_move_operator

R = "r1"
TIMES = {("a", "x"): 10.0, ("a", "b"): 5.0, ("b", "x"): 15.0}


def travel(r, a, b):
    return TIMES.get((a, b)) or TIMES.get((b, a)) or 100.0


def moves():
    return construct_move_operator(travel).instantiate(
        {"robot": [R], "location": ["a", "b", "x"]}
    )


def elapsed(s, a, s2):
    return 0.0 if a.name.startswith("wait-") else s2.time - s.time


def zero(s):
    return 0.0


def test_fluent_names_are_the_fluents_as_strings():
    s = State(0.0, {F(f"at {R} a"), F("raw o1"), F("cups-clean-3"), F(f"free {R}")})
    assert s.fluent_names == frozenset(str(f)[1:-1] for f in s.fluents)
    assert f"at {R} a" in s.fluent_names


def test_shortest_plan():
    actions = moves()
    start = State(0.0, {F(f"free {R}"), F(f"at {R} a")})
    r = timed_astar_search(start, LiteralGoal(F(f"at {R} x")), actions, zero, elapsed)
    assert (
        r["success"]
        and [actions[i].name for i in r["plan"]] == [f"move {R} a x"]
        and r["cost"] == 10.0
    )


def test_out_of_budget_returns_the_best_partial_plan():
    actions = moves()
    start = State(0.0, {F(f"free {R}"), F(f"at {R} a")})
    r = timed_astar_search(
        start, LiteralGoal(F("never")), actions, zero, elapsed, budget=1
    )
    assert not r["success"] and r["expanded"] == 1 and len(r["plan"]) == 1


def bits(n):
    """2**n states: one action sets each bit, one clears it."""
    busy = Effect(0, resulting_fluents={F("not free ?r")})
    ops = []
    for i in range(n):
        b = f"bit{i}"
        ops.append(
            Operator(
                f"set-{i}",
                [("?r", "robot")],
                [F("free ?r"), F(f"not {b}")],
                [busy, Effect(1, resulting_fluents={F("free ?r"), F(b)})],
            )
        )
        ops.append(
            Operator(
                f"clear-{i}",
                [("?r", "robot")],
                [F("free ?r"), F(b)],
                [busy, Effect(1, resulting_fluents={F("free ?r"), F(f"not {b}")})],
            )
        )
    return [a for o in ops for a in o.instantiate({"robot": [R]})]


def test_deadline():
    actions = bits(10)
    start = State(0.0, {F(f"free {R}")})
    r = timed_astar_search(
        start,
        LiteralGoal(F("never")),
        actions,
        zero,
        elapsed,
        budget=10**6,
        seconds_left=0.0,
    )
    assert r["timed_out"] and not r["success"] and r["expanded"] == 256


def test_a_node_keeps_one_python_object():
    """The state a callback sees for a node is one Python object (so memos keyed on it hit)."""
    actions = moves()
    start = State(0.0, {F(f"free {R}"), F(f"at {R} a")})
    seen_h, seen_parent = set(), set()

    def h(s):
        seen_h.add(id(s))
        return 0.0

    def c(s, a, s2):
        seen_parent.add(id(s))
        return elapsed(s, a, s2)

    keep = []  # every state the heuristic saw, alive, so ids stay unique

    def h_keep(s):
        keep.append(s)
        return h(s)

    r = timed_astar_search(
        start, LiteralGoal(F("never")), actions, h_keep, c, budget=50
    )
    assert not r["success"] and len(seen_parent) > 1
    assert (
        seen_parent - {id(start)} <= seen_h
    )  # every expanded node is the object its heuristic call got


def door():
    """The robot at a must get in at x once a door opens at t = 30; waiting costs 2 per second. Plain time
    dominance keeps only the earliest copy of "at x" (route 1, waits 19 s: 49); pricing the wait in the dominance
    test (idle_cost) finds the optimum, 31: walk about until the door opens."""
    timer = Operator(
        "start-timer",
        [("?r", "robot")],
        [F("free ?r"), F("ready")],
        [
            Effect(0, resulting_fluents={F("not free ?r"), F("not ready")}),
            Effect(1, resulting_fluents={F("free ?r")}),
            Effect(
                30,
                resulting_fluents={F("door-open")},
                cond_effects=[
                    (
                        {F("waiting-door ?r")},
                        [
                            Effect(
                                0,
                                resulting_fluents={
                                    F("free ?r"),
                                    F("not waiting-door ?r"),
                                },
                            )
                        ],
                    )
                ],
            ),
        ],
    )
    wait = Operator(
        "wait-door",
        [("?r", "robot")],
        [F("free ?r"), F("at ?r x")],
        [Effect(0, resulting_fluents={F("not free ?r"), F("waiting-door ?r")})],
    )
    enter = Operator(
        "enter",
        [("?r", "robot")],
        [F("free ?r"), F("at ?r x"), F("door-open")],
        [
            Effect(0, resulting_fluents={F("not free ?r")}),
            Effect(1, resulting_fluents={F("free ?r"), F("inside")}),
        ],
    )
    start = State(0.0, {F(f"free {R}"), F(f"at {R} a"), F("ready")})
    ((s0, _),) = transition(start, timer.instantiate({"robot": [R]})[0])
    actions = moves() + [
        a for o in (wait, enter) for a in o.instantiate({"robot": [R]})
    ]
    return s0, actions


def test_waiting_must_be_priced_in_dominance():
    s0, actions = door()

    def priced(s, a, s2):
        return 2.0 * (s2.time - s.time) if a.name.startswith("wait-") else 0.0

    goal = LiteralGoal(F("inside"))
    plain = timed_astar_search(
        s0, goal, actions, zero, elapsed, interval_cost=priced, budget=200_000
    )
    slack = timed_astar_search(
        s0,
        goal,
        actions,
        zero,
        elapsed,
        interval_cost=priced,
        idle_cost=lambda s, t0, t1: 2.0 * (t1 - t0),
        budget=200_000,
    )
    assert plain["success"] and slack["success"]
    assert abs(plain["cost"] - 49.0) < 1e-6 and abs(slack["cost"] - 31.0) < 1e-6
