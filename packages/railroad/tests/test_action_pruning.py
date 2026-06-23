"""Tests for probabilistic-achiever action pruning.

These build small, positive-precondition operator sets (so no negative-
precondition conversion is needed) where a single probabilistic fluent
(`revealed goal`) has many achievers, and check that pruning keeps only
the most-probable / cheapest few per robot and drops the support actions
that become orphaned.
"""

from __future__ import annotations

from railroad._action_pruning import prune_probabilistic_achievers
from railroad._bindings import Fluent, LiteralGoal, OrGoal, State
from railroad.core import Effect, Operator

F = Fluent


def _explore_operator(prob_by_frontier: dict[str, float]) -> Operator:
    """`explore ?r ?f` reveals the goal with a per-frontier probability.

    Constant branch durations (1.0s) so achiever `exec_cost` is uniform and
    the cheapest ordering is driven purely by the time to reach the frontier.
    """
    return Operator(
        name="explore",
        parameters=[("?r", "robot"), ("?f", "frontier")],
        preconditions=[F("at ?r ?f"), F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(
                time=0.1,
                prob_effects=[
                    (
                        (lambda r, f: prob_by_frontier[f], ["?r", "?f"]),
                        [Effect(time=1.0, resulting_fluents={F("free ?r"), F("revealed goal")})],
                    ),
                    (
                        (lambda r, f: 1.0 - prob_by_frontier[f], ["?r", "?f"]),
                        [Effect(time=1.0, resulting_fluents={F("free ?r")})],
                    ),
                ],
            ),
        ],
    )


def _move_operator(dist_by_target: dict[str, float]) -> Operator:
    """`move ?r ?from ?to` between locations; duration depends on the target."""
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?r ?from")}),
            Effect(
                time=(lambda r, frm, to: dist_by_target.get(to, 5.0), ["?r", "?from", "?to"]),
                resulting_fluents={F("free ?r"), F("at ?r ?to")},
            ),
        ],
    )


def _move_to_goal_operator() -> Operator:
    """`reach ?r ?from goal` drives to the revealed goal from any location."""
    return Operator(
        name="reach",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "goal")],
        preconditions=[F("at ?r ?from"), F("free ?r"), F("revealed ?to")],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?r ?from")}),
            Effect(time=1.0, resulting_fluents={F("free ?r"), F("at ?r ?to")}),
        ],
    )


def _no_op_operator() -> Operator:
    return Operator(
        name="no_op",
        parameters=[("?r", "robot")],
        preconditions=[F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(time=5.0, resulting_fluents={F("free ?r")}),
        ],
    )


def _frontiers(n: int) -> list[str]:
    return [f"f{i}" for i in range(1, n + 1)]


def test_prunes_to_top_n_and_cheapest_per_robot() -> None:
    frontiers = _frontiers(8)
    # Highest probability: f1..f4. Cheapest to reach: f8, f7 (the two nearest,
    # but low probability) — so the kept union is {f1,f2,f3,f4} ∪ {f7,f8}.
    prob = {"f1": 0.95, "f2": 0.9, "f3": 0.85, "f4": 0.8,
            "f5": 0.7, "f6": 0.6, "f7": 0.5, "f8": 0.4}
    dist = {"f1": 20.0, "f2": 21.0, "f3": 22.0, "f4": 23.0,
            "f5": 24.0, "f6": 25.0, "f7": 2.0, "f8": 1.0}

    objects = {
        "robot": {"r1", "r2"},
        "location": {"start", *frontiers},
        "frontier": set(frontiers),
        "goal": {"goal"},
    }
    operators = [
        _move_operator(dist),
        _move_to_goal_operator(),
        _explore_operator(prob),
        _no_op_operator(),
    ]
    actions = [a for op in operators for a in op.instantiate(objects)]

    state = State(
        time=0,
        fluents={F("at r1 start"), F("free r1"), F("at r2 start"), F("free r2")},
    )
    goal = OrGoal([LiteralGoal(F("at r1 goal")), LiteralGoal(F("at r2 goal"))])

    pruned = prune_probabilistic_achievers(
        state, goal, actions, top_n=4, cheapest_m=2,
        prune_orphaned_supports=False,
    )
    pruned_names = {a.name for a in pruned}

    kept = {"f1", "f2", "f3", "f4", "f7", "f8"}
    dropped = {"f5", "f6"}
    for robot in ("r1", "r2"):
        for f in kept:
            assert f"explore {robot} {f}" in pruned_names
        for f in dropped:
            assert f"explore {robot} {f}" not in pruned_names

    # Only explore actions are touched; moves/reach/no_op are untouched here.
    original_non_explore = {a.name for a in actions if not a.name.startswith("explore ")}
    assert original_non_explore <= pruned_names
    # Exactly the four low-value explore actions were removed (2 frontiers × 2 robots).
    assert len(actions) - len(pruned) == 4


def test_dead_frontier_removal_drops_all_referencing_actions() -> None:
    # Dense graph (every frontier also a routable location), so the generic
    # closure can't drop transit moves -- but frontier_objects can: a frontier
    # with no surviving achiever and nothing at it is removed entirely.
    frontiers = _frontiers(8)
    prob = {"f1": 0.95, "f2": 0.9, "f3": 0.85, "f4": 0.8,
            "f5": 0.7, "f6": 0.6, "f7": 0.5, "f8": 0.4}
    dist = {"f1": 20.0, "f2": 21.0, "f3": 22.0, "f4": 23.0,
            "f5": 24.0, "f6": 25.0, "f7": 2.0, "f8": 1.0}
    objects = {
        "robot": {"r1"},
        "location": {"start", *frontiers},
        "frontier": set(frontiers),
        "goal": {"goal"},
    }
    operators = [
        _move_operator(dist),
        _move_to_goal_operator(),
        _explore_operator(prob),
        _no_op_operator(),
    ]
    actions = [a for op in operators for a in op.instantiate(objects)]
    state = State(time=0, fluents={F("at r1 start"), F("free r1")})
    goal = LiteralGoal(F("at r1 goal"))

    pruned = prune_probabilistic_achievers(
        state, goal, actions, top_n=4, cheapest_m=2,
        prune_orphaned_supports=False, frontier_objects=set(frontiers),
    )
    names = {a.name for a in pruned}

    # f5, f6 are dead: every action mentioning them is gone (explore + moves).
    for dead in ("f5", "f6"):
        assert not any(dead in name.split()[1:] for name in names)
    # Kept frontiers still have their explore + a move that reaches them.
    assert "explore r1 f1" in names
    assert "move r1 start f1" in names
    assert "explore r1 f8" in names  # kept as a cheapest frontier


def test_orphaned_support_actions_pruned() -> None:
    # Restricted (non-dense) domain: the only way to a frontier is the direct
    # move from start, and the goal is `revealed goal` itself. Dropping an
    # explore achiever then leaves its move with nothing to enable.
    prob = {"fa": 0.9, "fb": 0.1}
    dist = {"fa": 3.0, "fb": 4.0}
    objects = {
        "robot": {"r1"},
        "location": {"start", "fa", "fb"},
        "frontier": {"fa", "fb"},
    }
    move = Operator(
        name="move",
        parameters=[("?r", "robot"), ("?to", "frontier")],
        preconditions=[F("at ?r start"), F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(
                time=(lambda r, to: dist[to], ["?r", "?to"]),
                resulting_fluents={F("free ?r"), F("at ?r ?to")},
            ),
        ],
    )
    operators = [move, _explore_operator(prob), _no_op_operator()]
    actions = [a for op in operators for a in op.instantiate(objects)]

    state = State(time=0, fluents={F("at r1 start"), F("free r1")})
    goal = LiteralGoal(F("revealed goal"))

    pruned = prune_probabilistic_achievers(
        state, goal, actions, top_n=1, cheapest_m=0,
        prune_orphaned_supports=True,
    )
    names = {a.name for a in pruned}

    # Highest-probability achiever and its enabling move survive.
    assert "explore r1 fa" in names
    assert "move r1 fa" in names
    # The pruned achiever and its now-orphaned move are gone.
    assert "explore r1 fb" not in names
    assert "move r1 fb" not in names
    # no_op produces `free r1` (needed by the surviving explore/move), so it stays.
    assert "no_op r1" in names


def test_noop_when_no_probabilistic_fluents() -> None:
    # Deterministic-only domain: pruning returns the action list unchanged.
    objects = {"robot": {"r1"}, "location": {"start", "a", "b"}}
    operators = [_move_operator({"a": 1.0, "b": 1.0}), _no_op_operator()]
    actions = [a for op in operators for a in op.instantiate(objects)]

    state = State(time=0, fluents={F("at r1 start"), F("free r1")})
    goal = LiteralGoal(F("at r1 a"))

    pruned = prune_probabilistic_achievers(state, goal, actions)
    assert pruned is actions
