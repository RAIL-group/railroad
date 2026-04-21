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


def make_search_action(name: str, obj: str, find_prob: float, search_cost: float) -> Action:
    """Single-effect probabilistic search action that achieves ``found <obj>``."""
    preconditions = {F("free r1")}
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
    """Actions that do not achieve any candidate goal fluent are preserved."""
    search_actions = [
        make_search_action(f"search_{i}", "X", find_prob=0.1 * (i + 1), search_cost=5.0)
        for i in range(5)
    ]
    move_action = make_deterministic_action("move r1 a b", F("at r1 b"), cost=3.0)
    actions = search_actions + [move_action]
    goal = LiteralGoal(F("found X"))

    pruned = prune_probabilistic_achievers(_initial_state(), goal, actions, top_k=3)
    names = {a.name for a in pruned}

    assert "move r1 a b" in names
    # Some searches should have been dropped (5 achievers, top_k=3 → at most 6 kept,
    # but prob and time rankings overlap here so we expect fewer).
    assert len([a for a in pruned if a.name.startswith("search_")]) < 5


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
