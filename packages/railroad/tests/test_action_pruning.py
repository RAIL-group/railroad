"""Tests for probabilistic-achiever action pruning."""
import pytest

from railroad._action_pruning import prune_probabilistic_achievers
from railroad._bindings import (
    Action,
    Fluent,
    GroundedEffect,
    LiteralGoal,
    OrGoal,
    State,
    get_achievers_for_fluent,
)
from railroad.planner import MCTSPlanner

F = Fluent


def make_search_action(
    name: str, obj: str, find_prob: float, search_cost: float, robot: str = "r1"
) -> Action:
    """Single-effect probabilistic search action that achieves ``found <obj>``."""
    preconditions = {F(f"free {robot}")}
    effects = [
        GroundedEffect(
            time=search_cost,
            resulting_fluents=set(),
            prob_effects=[
                (find_prob, [GroundedEffect(time=0, resulting_fluents={F(f"found {obj}")})]),
                (1.0 - find_prob, []),
            ],
        ),
    ]
    return Action(preconditions, effects, name=name)


def make_deterministic_action(name: str, fluent: Fluent, cost: float) -> Action:
    preconditions = {F("free r1")}
    effects = [GroundedEffect(time=cost, resulting_fluents={fluent})]
    return Action(preconditions, effects, name=name)


def _initial_state() -> State:
    return State(time=0, fluents={F("free r1")})


def test_prunes_lowest_quality_achiever():
    """Extra achiever that is neither top-3 by prob nor top-3 by time gets dropped."""
    actions = [
        make_search_action("search_A", "X", find_prob=0.9, search_cost=100.0),
        make_search_action("search_B", "X", find_prob=0.8, search_cost=100.0),
        make_search_action("search_C", "X", find_prob=0.7, search_cost=100.0),
        make_search_action("search_D", "X", find_prob=0.1, search_cost=1.0),
        make_search_action("search_E", "X", find_prob=0.1, search_cost=2.0),
        make_search_action("search_F", "X", find_prob=0.1, search_cost=3.0),
        make_search_action("search_G", "X", find_prob=0.05, search_cost=1000.0),
    ]
    goal = LiteralGoal(F("found X"))

    # Sanity: all 7 show up as achievers in the forward phase.
    assert len(get_achievers_for_fluent(_initial_state(), F("found X"), actions)) == 7

    pruned = prune_probabilistic_achievers(_initial_state(), goal, actions, top_k=3)
    names = {a.name for a in pruned}

    assert names == {f"search_{c}" for c in "ABCDEF"}
    assert "search_G" not in names


def test_preserves_top3_union_when_no_overlap():
    """When top-3-by-prob and top-3-by-time are disjoint, 6 achievers survive."""
    actions = [
        make_search_action("search_A", "X", find_prob=0.9, search_cost=100.0),
        make_search_action("search_B", "X", find_prob=0.8, search_cost=100.0),
        make_search_action("search_C", "X", find_prob=0.7, search_cost=100.0),
        make_search_action("search_D", "X", find_prob=0.1, search_cost=1.0),
        make_search_action("search_E", "X", find_prob=0.1, search_cost=2.0),
        make_search_action("search_F", "X", find_prob=0.1, search_cost=3.0),
    ]
    goal = LiteralGoal(F("found X"))

    pruned = prune_probabilistic_achievers(_initial_state(), goal, actions, top_k=3)
    assert len(pruned) == 6


def test_no_pruning_when_few_achievers():
    """With <= top_k achievers, nothing is pruned."""
    actions = [
        make_search_action("search_A", "X", find_prob=0.9, search_cost=100.0),
        make_search_action("search_B", "X", find_prob=0.5, search_cost=50.0),
        make_search_action("search_C", "X", find_prob=0.1, search_cost=1.0),
    ]
    goal = LiteralGoal(F("found X"))

    pruned = prune_probabilistic_achievers(_initial_state(), goal, actions, top_k=3)
    assert {a.name for a in pruned} == {"search_A", "search_B", "search_C"}


def test_deterministic_goal_is_not_pruned():
    """Achievers whose probabilities are all 1.0 should not be touched."""
    actions = [
        make_deterministic_action(f"det_{i}", F("goal_reached"), cost=float(i + 1))
        for i in range(6)
    ]
    goal = LiteralGoal(F("goal_reached"))

    pruned = prune_probabilistic_achievers(_initial_state(), goal, actions, top_k=3)
    assert len(pruned) == len(actions)


def test_non_achiever_actions_pass_through():
    """Actions that do not achieve any candidate goal fluent are preserved
    when the helpful-action filter is disabled. (With it enabled, a move
    whose effect is not backward-reachable from the goal is correctly dropped
    --- see ``test_helpful_action_filter_drops_unreachable_move``.)
    """
    search_actions = [
        make_search_action(f"search_{i}", "X", find_prob=0.1 * (i + 1), search_cost=5.0)
        for i in range(5)
    ]
    move_action = make_deterministic_action("move r1 a b", F("at r1 b"), cost=3.0)
    actions = search_actions + [move_action]
    goal = LiteralGoal(F("found X"))

    pruned = prune_probabilistic_achievers(
        _initial_state(), goal, actions, top_k=3,
        enable_helpful_action_filter=False,
    )
    names = {a.name for a in pruned}

    assert "move r1 a b" in names
    # Some searches should have been dropped (5 achievers, top_k=3 → at most 6 kept,
    # but prob and time rankings overlap here so we expect fewer).
    assert len([a for a in pruned if a.name.startswith("search_")]) < 5


def test_helpful_action_filter_drops_unreachable_move():
    """With the helpful-action filter enabled, a move whose effect is not
    backward-reachable from the goal (searches here only require `free r1`,
    not any `at` fluent) is correctly dropped alongside the pruned achievers.
    """
    search_actions = [
        make_search_action(f"search_{i}", "X", find_prob=0.1 * (i + 1), search_cost=5.0)
        for i in range(5)
    ]
    move_action = make_deterministic_action("move r1 a b", F("at r1 b"), cost=3.0)
    actions = search_actions + [move_action]
    goal = LiteralGoal(F("found X"))

    pruned = prune_probabilistic_achievers(_initial_state(), goal, actions, top_k=3)
    names = {a.name for a in pruned}

    assert "move r1 a b" not in names
    assert any(n.startswith("search_") for n in names)


def test_helpful_action_filter_keeps_backward_reachable_move():
    """A move whose destination is a precondition of a surviving achiever
    should be kept by the helpful-action filter."""
    # Search for X requires being at location `loc` (so at r1 loc is a
    # precondition of a kept achiever, making move-to-loc backward-reachable).
    def search_at_loc(name, find_prob, cost, loc="loc"):
        preconditions = {F("free r1"), F(f"at r1 {loc}")}
        effects = [
            GroundedEffect(
                time=cost,
                resulting_fluents=set(),
                prob_effects=[
                    (find_prob, [GroundedEffect(time=0, resulting_fluents={F("found X")})]),
                    (1.0 - find_prob, []),
                ],
            ),
        ]
        return Action(preconditions, effects, name=name)

    search = search_at_loc("search_loc", 0.8, 5.0)
    move_action = Action(
        {F("at r1 start"), F("free r1")},
        [GroundedEffect(time=3.0, resulting_fluents={F("at r1 loc"), F("free r1")})],
        name="move r1 start loc",
    )
    actions = [search, move_action]
    goal = LiteralGoal(F("found X"))
    state = State(time=0, fluents={F("free r1"), F("at r1 start")})

    pruned = prune_probabilistic_achievers(state, goal, actions, top_k=3)
    names = {a.name for a in pruned}

    assert "move r1 start loc" in names
    assert "search_loc" in names


def test_dnf_goal_considers_all_branches():
    """OR-goal: achievers of both disjuncts should be evaluated independently."""
    actions = [
        # Achievers of A (6 of them).
        make_search_action("searchA_hi", "A", find_prob=0.9, search_cost=100.0),
        make_search_action("searchA_mi", "A", find_prob=0.5, search_cost=50.0),
        make_search_action("searchA_lo", "A", find_prob=0.05, search_cost=1000.0),
        make_search_action("searchA_q1", "A", find_prob=0.1, search_cost=1.0),
        make_search_action("searchA_q2", "A", find_prob=0.1, search_cost=2.0),
        make_search_action("searchA_q3", "A", find_prob=0.1, search_cost=3.0),
        # Achievers of B (6 of them).
        make_search_action("searchB_hi", "B", find_prob=0.9, search_cost=100.0),
        make_search_action("searchB_mi", "B", find_prob=0.5, search_cost=50.0),
        make_search_action("searchB_lo", "B", find_prob=0.05, search_cost=1000.0),
        make_search_action("searchB_q1", "B", find_prob=0.1, search_cost=1.0),
        make_search_action("searchB_q2", "B", find_prob=0.1, search_cost=2.0),
        make_search_action("searchB_q3", "B", find_prob=0.1, search_cost=3.0),
    ]
    goal = OrGoal([LiteralGoal(F("found A")), LiteralGoal(F("found B"))])

    pruned = prune_probabilistic_achievers(_initial_state(), goal, actions, top_k=3)
    names = {a.name for a in pruned}

    # The *_lo achievers should be pruned from both branches.
    assert "searchA_lo" not in names
    assert "searchB_lo" not in names
    # Top-ranked achievers from both branches are kept.
    assert {"searchA_hi", "searchB_hi"}.issubset(names)


def test_mcts_planner_flag_disables_pruning():
    """MCTSPlanner(enable_action_pruning=False) should not prune."""
    actions = [
        make_search_action(f"search_{i}", "X", find_prob=0.01, search_cost=1000.0)
        for i in range(8)
    ]
    # Mix in one attractive achiever.
    actions.append(make_search_action("search_best", "X", find_prob=0.99, search_cost=1.0))
    goal = LiteralGoal(F("found X"))

    pruned = prune_probabilistic_achievers(_initial_state(), goal, actions, top_k=3)
    assert len(pruned) < len(actions)

    planner_off = MCTSPlanner(actions, enable_action_pruning=False)
    planner_on = MCTSPlanner(actions, enable_action_pruning=True)

    # Run a few iterations so the C++ side executes; we only check the flag wiring
    # does not error and returns a string action name.
    name_off = planner_off(_initial_state(), goal, max_iterations=10)
    name_on = planner_on(_initial_state(), goal, max_iterations=10)
    assert isinstance(name_off, str)
    assert isinstance(name_on, str)


def test_pruning_is_per_robot():
    """Each robot keeps its own top-k; one robot can't monopolize the keepers."""
    # Robot r1's achievers are uniformly better (higher prob, lower cost) than
    # r2's. Under global (non-per-robot) pruning, all keepers would be r1 and
    # r2's achievers would all be dropped. Per-robot pruning must retain a
    # fair share on each side.
    actions = [
        # r1: 6 achievers of (found X), all strong.
        make_search_action("s_r1_a", "X", 0.9, 100.0, robot="r1"),
        make_search_action("s_r1_b", "X", 0.8, 100.0, robot="r1"),
        make_search_action("s_r1_c", "X", 0.7, 100.0, robot="r1"),
        make_search_action("s_r1_d", "X", 0.6, 10.0, robot="r1"),
        make_search_action("s_r1_e", "X", 0.5, 20.0, robot="r1"),
        make_search_action("s_r1_f", "X", 0.4, 30.0, robot="r1"),
        # r2: 5 achievers, weaker but should still be kept per-robot.
        make_search_action("s_r2_a", "X", 0.3, 200.0, robot="r2"),
        make_search_action("s_r2_b", "X", 0.2, 200.0, robot="r2"),
        make_search_action("s_r2_c", "X", 0.1, 500.0, robot="r2"),
        make_search_action("s_r2_d", "X", 0.05, 40.0, robot="r2"),
        make_search_action("s_r2_e", "X", 0.05, 50.0, robot="r2"),
    ]
    # Need r2's free fluent in the state so forward phase reaches r2's actions.
    state = State(time=0, fluents={F("free r1"), F("free r2")})
    goal = LiteralGoal(F("found X"))

    pruned = prune_probabilistic_achievers(state, goal, actions, top_k=2)
    names = {a.name for a in pruned}

    r1_kept = [n for n in names if n.startswith("s_r1_")]
    r2_kept = [n for n in names if n.startswith("s_r2_")]

    # Each robot's group must retain at least one survivor (not monopolized).
    assert len(r1_kept) >= 1, f"r1 swept entirely; got {names}"
    assert len(r2_kept) >= 1, f"r2 swept entirely; got {names}"
    # Some per-robot pruning did happen on r1 (6 -> at most 4 via top-2 x 2).
    assert len(r1_kept) < 6, f"expected r1 to be pruned, kept {r1_kept}"


def test_at_goal_expands_to_found_landmark_for_pruning():
    """Goals of the form `at <obj> <loc>` should add `found <obj>` as a pruning
    target, so probabilistic search actions get pruned even when the direct
    goal achievers are all deterministic place actions.
    """
    # 6 search achievers of found X, only 2 to keep (one hi-prob, one lo-time).
    search_actions = [
        make_search_action("search_hi", "X", find_prob=0.9, search_cost=100.0),
        make_search_action("search_mi", "X", find_prob=0.5, search_cost=50.0),
        make_search_action("search_lo", "X", find_prob=0.05, search_cost=1000.0),
        make_search_action("search_q1", "X", find_prob=0.1, search_cost=1.0),
        make_search_action("search_q2", "X", find_prob=0.1, search_cost=2.0),
        make_search_action("search_q3", "X", find_prob=0.1, search_cost=3.0),
    ]
    # A deterministic `place r1 loc X` achieving `at X loc`, requiring `found X`
    # as a precondition so the action chain ties the two together.
    place = Action(
        {F("found X"), F("free r1")},
        [GroundedEffect(time=5.0, resulting_fluents={F("at X loc"), F("free r1")})],
        name="place r1 loc X",
    )
    actions = search_actions + [place]
    goal = LiteralGoal(F("at X loc"))

    pruned = prune_probabilistic_achievers(_initial_state(), goal, actions, top_k=1)
    names = {a.name for a in pruned}

    # Place is not an achiever of `found X`, so it survives.
    assert "place r1 loc X" in names
    # Low-quality searches are dropped.
    assert "search_lo" not in names
    assert len([n for n in names if n.startswith("search_")]) < len(search_actions)
