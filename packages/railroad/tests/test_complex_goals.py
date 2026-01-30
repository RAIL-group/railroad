"""
Tests for complex goal expressions (AND, OR) following the design in complex_goals.md.

Updated to match the revised C++ goal implementation:
- Structural equality (==) is structural, not hash-based
- Deduplication is structural, not "hash-only"
- Negated literal evaluation works directly against positive-only state sets:
    LiteralGoal(~P) is satisfied iff P is absent from the state
- Normalization includes:
    AND(P, ~P) -> FALSE
    OR(P, ~P)  -> TRUE
- goal_count semantics are structure-aware:
    Literal: 1 if satisfied else 0 (negation-correct)
    AND: sum of children
    OR: max of children (best branch progress)
    TRUE/FALSE: 0
- Heuristic semantics for OR are "cheapest alternative":
    h(OR(...)) = min(h(branch_i))
    and nested AND-with-OR should respect that structure (no flattening of OR into AND)

Organized into themed test classes:
- TestGoalSatisfaction: Core truth-table tests for goal evaluation (incl. negation)
- TestNormalization: Verify preprocessing produces correct structures and canonical forms
- TestBranchAccess: Verify heuristic can access goal structure + goal_count semantics
- TestHouseholdScenarios: Regression tests using realistic goal expressions
- TestPlannerWithGoals: Integration tests for planner integration with Goal objects
- TestHeuristicORBranches: Heuristic properties for OR branching
- TestGoalNegativeConversion: Tests for optional conversion utilities (if used elsewhere)
- TestFluentOperatorOverloading: & and | operator overloading on Fluent / Goal
"""

import math
import pytest

from railroad.core import Fluent, State, transition, get_action_by_name
from railroad.operators import construct_move_visited_operator

# Goal classes from bindings
from railroad._bindings import (
    GoalType,
    LiteralGoal,
    AndGoal,
    OrGoal,
    TrueGoal,
    FalseGoal,
    ff_heuristic,
)

from railroad.planner import MCTSPlanner, get_usable_actions

F = Fluent

# Helper
def _run_mcts_until(goal, initial_state, all_actions, max_steps=30, max_iterations=800):
    """Helper to run MCTS until goal is satisfied or max_steps reached."""
    mcts = MCTSPlanner(all_actions)
    state = initial_state

    for _ in range(max_steps):
        if goal.evaluate(state.fluents):
            break
        action_name = mcts(state, goal, max_iterations=max_iterations, c=10)
        if action_name == "NONE":
            break
        action = get_action_by_name(all_actions, action_name)
        state = transition(state, action)[0][0]

    return state


# -----------------------------
# Core satisfaction semantics
# -----------------------------

class TestGoalSatisfaction:
    """Core truth-table tests for goal evaluation."""

    @pytest.mark.parametrize("goal_factory,state_fluents,expected", [
        # Literal goal - satisfied
        (lambda: LiteralGoal(F("at r1 kitchen")), {F("at r1 kitchen"), F("free r1")}, True),
        # Literal goal - not satisfied
        (lambda: LiteralGoal(F("at r1 kitchen")), {F("at r1 bedroom"), F("free r1")}, False),
        # AND goal - all true
        (lambda: AndGoal([LiteralGoal(F("at r1 kitchen")), LiteralGoal(F("free r1"))]),
         {F("at r1 kitchen"), F("free r1"), F("visited kitchen")}, True),
        # AND goal - any false
        (lambda: AndGoal([LiteralGoal(F("at r1 kitchen")), LiteralGoal(F("holding r1 cup"))]),
         {F("at r1 kitchen"), F("free r1")}, False),
        # OR goal - any true
        (lambda: OrGoal([LiteralGoal(F("at r1 kitchen")), LiteralGoal(F("at r1 bedroom"))]),
         {F("at r1 bedroom"), F("free r1")}, True),
        # OR goal - all false
        (lambda: OrGoal([LiteralGoal(F("at r1 kitchen")), LiteralGoal(F("at r1 bedroom"))]),
         {F("at r1 living_room"), F("free r1")}, False),
        # TRUE goal - always satisfied
        (lambda: TrueGoal(), set(), True),
        # FALSE goal - never satisfied
        (lambda: FalseGoal(), {F("any fluent")}, False),
    ], ids=[
        "literal_satisfied",
        "literal_not_satisfied",
        "and_all_true",
        "and_any_false",
        "or_any_true",
        "or_all_false",
        "true_goal",
        "false_goal",
    ])
    def test_evaluate_semantics(self, goal_factory, state_fluents, expected):
        goal = goal_factory()
        assert goal.evaluate(state_fluents) is expected

    def test_negated_literal_goal_satisfied_when_positive_absent(self):
        """Negated literal is satisfied iff the positive form is absent."""
        goal = LiteralGoal(~F("at r1 kitchen"))

        assert goal.evaluate(set()) is True
        assert goal.evaluate({F("at r1 bedroom")}) is True
        assert goal.evaluate({F("at r1 kitchen")}) is False

    def test_nested_and_or_satisfied(self):
        """AND(table_set, OR(toast_ready, cereal_ready)) - mixed nesting evaluation."""
        a = LiteralGoal(F("table_set"))
        b = LiteralGoal(F("toast_ready"))
        c = LiteralGoal(F("cereal_ready"))
        goal = AndGoal([a, OrGoal([b, c])])

        assert goal.evaluate({F("table_set"), F("toast_ready")}) is True
        assert goal.evaluate({F("table_set"), F("cereal_ready")}) is True
        assert goal.evaluate({F("table_set"), F("toast_ready"), F("cereal_ready")}) is True
        assert goal.evaluate({F("toast_ready")}) is False
        assert goal.evaluate({F("table_set")}) is False


# -----------------------------
# Normalization invariants
# -----------------------------

class TestNormalization:
    """Verify preprocessing produces correct structures and canonical forms."""

    def test_flatten_nested_and(self):
        a = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        c = LiteralGoal(F("c"))

        nested = AndGoal([AndGoal([a, b]), c])
        normalized = nested.normalize()

        assert normalized.get_type() == GoalType.AND
        assert len(list(normalized.children())) == 3
        assert normalized.evaluate({F("a"), F("b"), F("c")}) is True
        assert normalized.evaluate({F("a"), F("b")}) is False

    def test_flatten_nested_or(self):
        a = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        c = LiteralGoal(F("c"))

        nested = OrGoal([OrGoal([a, b]), c])
        normalized = nested.normalize()

        assert normalized.get_type() == GoalType.OR
        assert len(list(normalized.children())) == 3
        assert normalized.evaluate({F("a")}) is True
        assert normalized.evaluate({F("c")}) is True
        assert normalized.evaluate({F("d")}) is False

    @pytest.mark.parametrize("ctor,expected_type,eval_empty", [
        (AndGoal, GoalType.TRUE_GOAL, True),
        (OrGoal, GoalType.FALSE_GOAL, False),
    ], ids=["empty_and_to_true", "empty_or_to_false"])
    def test_empty_goal_normalization(self, ctor, expected_type, eval_empty):
        """Empty AND normalizes to TRUE, empty OR normalizes to FALSE."""
        goal = ctor([]).normalize()
        assert goal.get_type() == expected_type
        assert goal.evaluate(set()) is eval_empty
        assert goal.evaluate({F("random fluent")}) is eval_empty

    @pytest.mark.parametrize("ctor,constant,expected_type,eval_a,eval_empty", [
        # AND([a, TRUE]) => a (literal)
        (AndGoal, TrueGoal(), GoalType.LITERAL, True, False),
        # AND([a, FALSE]) => FALSE
        (AndGoal, FalseGoal(), GoalType.FALSE_GOAL, False, False),
        # OR([a, FALSE]) => a (literal)
        (OrGoal, FalseGoal(), GoalType.LITERAL, True, False),
        # OR([a, TRUE]) => TRUE
        (OrGoal, TrueGoal(), GoalType.TRUE_GOAL, True, True),
    ], ids=[
        "and_with_true",
        "and_with_false",
        "or_with_false",
        "or_with_true",
    ])
    def test_constant_folding(self, ctor, constant, expected_type, eval_a, eval_empty):
        """Constant folding: AND/OR with TRUE/FALSE constants."""
        a = LiteralGoal(F("a"))
        goal = ctor([a, constant])
        normalized = goal.normalize()

        assert normalized.get_type() == expected_type
        assert normalized.evaluate({F("a")}) is eval_a
        assert normalized.evaluate(set()) is eval_empty

    @pytest.mark.parametrize("ctor", [AndGoal, OrGoal], ids=["and", "or"])
    def test_single_child_collapses(self, ctor):
        """Single-child AND/OR collapses to the child."""
        g = ctor([LiteralGoal(F("a"))]).normalize()
        assert g.get_type() == GoalType.LITERAL

    @pytest.mark.parametrize("ctor,expected_type", [
        (AndGoal, GoalType.AND),
        (OrGoal, GoalType.OR),
    ], ids=["and_dedup", "or_dedup"])
    def test_deduplication_structural(self, ctor, expected_type):
        """Structural deduplication removes duplicate literals."""
        a1 = LiteralGoal(F("a"))
        a2 = LiteralGoal(F("a"))  # Duplicate structurally
        b = LiteralGoal(F("b"))

        normalized = ctor([a1, a2, b]).normalize()
        assert normalized.get_type() == expected_type

        lits = normalized.get_all_literals()
        assert F("a") in lits
        assert F("b") in lits
        assert len(lits) == 2

    def test_no_distribution_and_over_or(self):
        a = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        c = LiteralGoal(F("c"))

        normalized = AndGoal([a, OrGoal([b, c])]).normalize()

        assert normalized.get_type() == GoalType.AND
        assert len(list(normalized.children())) == 2
        child_types = [child.get_type() for child in normalized.children()]
        assert GoalType.LITERAL in child_types
        assert GoalType.OR in child_types

    def test_contradiction_and_normalizes_to_false(self):
        normalized = AndGoal([LiteralGoal(F("a")), LiteralGoal(~F("a"))]).normalize()
        assert normalized.get_type() == GoalType.FALSE_GOAL
        assert normalized.evaluate(set()) is False
        assert normalized.evaluate({F("a")}) is False

    def test_tautology_or_normalizes_to_true(self):
        normalized = OrGoal([LiteralGoal(F("a")), LiteralGoal(~F("a"))]).normalize()
        assert normalized.get_type() == GoalType.TRUE_GOAL
        assert normalized.evaluate(set()) is True
        assert normalized.evaluate({F("a")}) is True

    def test_normalize_idempotent(self):
        g = AndGoal([
            LiteralGoal(F("a")),
            TrueGoal(),
            AndGoal([LiteralGoal(F("b"))]),
            OrGoal([FalseGoal(), LiteralGoal(F("c"))]),
        ])
        n1 = g.normalize()
        n2 = n1.normalize()

        # This asserts the pybind exposes structural __eq__.
        assert n1 == n2, "Expected normalize() to be idempotent"

    def test_canonical_ordering_strict_structural_equality(self):
        a = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        c = LiteralGoal(F("c"))

        g1 = AndGoal([a, b, c]).normalize()
        g2 = AndGoal([c, b, a]).normalize()
        g3 = AndGoal([b, a, c]).normalize()

        assert g1 == g2 == g3, "Canonical ordering failed"

    def test_example_from_design_doc(self):
        # Input: AND(table_set, AND(toast_ready, TRUE), OR(FALSE, coffee_ready))
        # Expected: AND(table_set, toast_ready, coffee_ready)
        table_set = LiteralGoal(F("table_set"))
        toast_ready = LiteralGoal(F("toast_ready"))
        coffee_ready = LiteralGoal(F("coffee_ready"))

        goal = AndGoal([
            table_set,
            AndGoal([toast_ready, TrueGoal()]),
            OrGoal([FalseGoal(), coffee_ready]),
        ])
        normalized = goal.normalize()

        assert normalized.get_type() == GoalType.AND
        assert len(list(normalized.children())) == 3

        assert normalized.evaluate({F("table_set"), F("toast_ready"), F("coffee_ready")}) is True
        assert normalized.evaluate({F("table_set"), F("toast_ready")}) is False


# -----------------------------
# Structure access + goal_count
# -----------------------------

class TestBranchAccess:
    """Verify heuristic can access goal structure, and goal_count semantics are correct."""

    def test_get_all_literals_nested(self):
        goal = AndGoal([LiteralGoal(F("a")), OrGoal([LiteralGoal(F("b")), LiteralGoal(F("c"))])])
        lits = goal.get_all_literals()
        assert len(lits) == 3
        assert F("a") in lits and F("b") in lits and F("c") in lits

    def test_goal_count_and_is_sum(self):
        goal = AndGoal([LiteralGoal(F("a")), LiteralGoal(F("b")), LiteralGoal(F("c"))])
        assert goal.goal_count({F("a"), F("b")}) == 2
        assert goal.goal_count({F("a"), F("b"), F("c")}) == 3
        assert goal.goal_count({F("d")}) == 0

    def test_goal_count_or_is_max(self):
        goal = OrGoal([LiteralGoal(F("a")), LiteralGoal(F("b"))])
        assert goal.goal_count(set()) == 0
        assert goal.goal_count({F("a")}) == 1
        assert goal.goal_count({F("b")}) == 1
        assert goal.goal_count({F("a"), F("b")}) == 1  # max, not sum

    def test_goal_count_negation_correct(self):
        goal = AndGoal([LiteralGoal(~F("a")), LiteralGoal(F("b"))])

        # a absent, b absent => only ~a satisfied
        assert goal.goal_count(set()) == 1
        # a present, b absent => none satisfied
        assert goal.goal_count({F("a")}) == 0
        # a absent, b present => both satisfied
        assert goal.goal_count({F("b")}) == 2


# -----------------------------
# Realistic regression scenarios
# -----------------------------

class TestHouseholdScenarios:
    """Regression tests using realistic goal expressions from the design doc."""

    def test_set_table_and_choose_meal(self):
        """AND(table_set, OR(meal_a, meal_b)) - realistic shape."""
        table_set = LiteralGoal(F("table_set"))
        meal_a = AndGoal([LiteralGoal(F("pasta_ready")), LiteralGoal(F("sauce_ready"))])
        meal_b = AndGoal([LiteralGoal(F("salad_ready")), LiteralGoal(F("dressing_ready"))])

        goal = AndGoal([table_set, OrGoal([meal_a, meal_b])])

        assert goal.evaluate({F("table_set"), F("pasta_ready"), F("sauce_ready")}) is True
        assert goal.evaluate({F("table_set"), F("salad_ready"), F("dressing_ready")}) is True
        assert goal.evaluate({F("pasta_ready"), F("sauce_ready")}) is False
        assert goal.evaluate({F("table_set"), F("pasta_ready")}) is False


# -----------------------------
# Planner integration (smoke tests)
# -----------------------------

@pytest.mark.slow
class TestPlannerWithGoals:
    """Integration tests for planner + Goal objects.

    These can be stochastic (depending on planner implementation). Marked slow.
    """

    @pytest.mark.parametrize("goal_factory,assertion_msg", [
        # AND goal: must visit both locations
        (lambda: AndGoal([LiteralGoal(F("visited a")), LiteralGoal(F("visited b"))]),
         "AND goal should be satisfied"),
        # OR goal: must visit at least one location
        (lambda: OrGoal([LiteralGoal(F("visited a")), LiteralGoal(F("visited b"))]),
         "OR goal should be satisfied"),
    ], ids=["and_goal", "or_goal"])
    def test_planner_with_goal(self, goal_factory, assertion_msg):
        objects_by_type = {"robot": ["r1"], "location": ["start", "a", "b"]}
        move_op = construct_move_visited_operator(lambda *args: 5.0)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(time=0, fluents={F("at r1 start"), F("free r1"), F("visited start")})
        goal = goal_factory()

        # Prune actions using forward reachability.
        all_actions = get_usable_actions(initial_state, all_actions)

        state = _run_mcts_until(goal, initial_state, all_actions, max_steps=30, max_iterations=800)
        assert goal.evaluate(state.fluents), assertion_msg

    def test_planner_with_negative_literal_goal(self):
        """Goal: robot is NOT at start."""
        objects_by_type = {"robot": ["r1"], "location": ["start", "kitchen"]}
        move_op = construct_move_visited_operator(lambda *args: 5.0)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(time=0, fluents={F("at r1 start"), F("free r1")})
        goal = LiteralGoal(~F("at r1 start"))

        state = _run_mcts_until(goal, initial_state, all_actions, max_steps=15, max_iterations=400)
        assert goal.evaluate(state.fluents), "Negated literal goal should become satisfied after moving away"

    def test_ff_heuristic_with_goal_object(self):
        """Test ff_heuristic function with Goal objects."""
        objects_by_type = {
            "robot": ["r1"],
            "location": ["start", "a", "b"],
        }
        move_op = construct_move_visited_operator(lambda *args: 5.0)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(
            time=0,
            fluents={F("at r1 start"), F("free r1"), F("visited start")}
        )

        # Create goal
        goal = AndGoal([
            LiteralGoal(F("visited a")),
            LiteralGoal(F("visited b")),
        ])

        # Compute heuristic
        h_value = ff_heuristic(initial_state, goal, all_actions)

        # Heuristic should be positive (need to visit 2 locations)
        assert h_value > 0, f"Heuristic should be positive, got {h_value}"


# -----------------------------
# Heuristic properties
# -----------------------------

class TestHeuristicORBranches:
    """Tests for efficient OR branch handling in the heuristic."""

    def test_or_goal_returns_min_branch_cost(self):
        objects_by_type = {"robot": ["r1"], "location": ["start", "near", "far"]}

        def cost_fn(robot, loc1, loc2):
            if loc2 == "near":
                return 5.0
            if loc2 == "far":
                return 15.0
            return 10.0

        move_op = construct_move_visited_operator(cost_fn)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(time=0, fluents={F("at r1 start"), F("free r1"), F("visited start")})

        or_goal = OrGoal([LiteralGoal(F("visited near")), LiteralGoal(F("visited far"))])

        h_near = ff_heuristic(initial_state, LiteralGoal(F("visited near")), all_actions)
        h_far = ff_heuristic(initial_state, LiteralGoal(F("visited far")), all_actions)
        h_or = ff_heuristic(initial_state, or_goal, all_actions)

        assert h_or == pytest.approx(min(h_near, h_far)), f"h_or={h_or}, h_near={h_near}, h_far={h_far}"

    def test_true_goal_returns_zero(self):
        objects_by_type = {"robot": ["r1"], "location": ["start"]}
        move_op = construct_move_visited_operator(lambda *args: 5.0)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(time=0, fluents={F("at r1 start"), F("free r1")})

        h_value = ff_heuristic(initial_state, TrueGoal(), all_actions)
        assert h_value == 0.0

    def test_false_goal_returns_infinity(self):
        objects_by_type = {"robot": ["r1"], "location": ["start"]}
        move_op = construct_move_visited_operator(lambda *args: 5.0)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(time=0, fluents={F("at r1 start"), F("free r1")})

        h_value = ff_heuristic(initial_state, FalseGoal(), all_actions)
        assert math.isinf(h_value)

    def test_or_with_unreachable_branch_uses_reachable_branch(self):
        objects_by_type = {"robot": ["r1"], "location": ["start", "reachable"]}
        move_op = construct_move_visited_operator(lambda *args: 5.0)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(time=0, fluents={F("at r1 start"), F("free r1"), F("visited start")})

        or_goal = OrGoal([
            LiteralGoal(F("visited reachable")),
            LiteralGoal(F("visited nonexistent")),  # no such location/actions
        ])

        h_value = ff_heuristic(initial_state, or_goal, all_actions)
        assert not math.isinf(h_value)

    def test_and_with_nested_or_is_min_of_and_branches(self):
        """AND(A, OR(B,C)) should behave like min(AND(A,B), AND(A,C)) for FF heuristic.

        Uses asymmetric costs to ensure the test is discriminating:
        - Moving to b is cheap (5.0)
        - Moving to c is expensive (50.0)
        """
        objects_by_type = {"robot": ["r1"], "location": ["start", "a", "b", "c"]}

        def cost_fn(robot, loc1, loc2):
            if loc2 == "b":
                return 5.0
            if loc2 == "c":
                return 50.0
            return 10.0

        move_op = construct_move_visited_operator(cost_fn)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(time=0, fluents={F("at r1 start"), F("free r1"), F("visited start")})

        nested = AndGoal([
            LiteralGoal(F("visited a")),
            OrGoal([LiteralGoal(F("visited b")), LiteralGoal(F("visited c"))]),
        ])

        and_ab = AndGoal([LiteralGoal(F("visited a")), LiteralGoal(F("visited b"))])
        and_ac = AndGoal([LiteralGoal(F("visited a")), LiteralGoal(F("visited c"))])

        h_nested = ff_heuristic(initial_state, nested, all_actions)
        h_ab = ff_heuristic(initial_state, and_ab, all_actions)
        h_ac = ff_heuristic(initial_state, and_ac, all_actions)

        # With asymmetric costs, h_ab should be strictly less than h_ac
        assert h_ab < h_ac, f"Expected h_ab < h_ac but got h_ab={h_ab}, h_ac={h_ac}"
        assert h_nested == pytest.approx(min(h_ab, h_ac)), f"h_nested={h_nested}, h_ab={h_ab}, h_ac={h_ac}"


# -----------------------------
# Optional: conversion utilities
# -----------------------------

class TestGoalNegativeConversion:
    """Tests for converting negative fluents in goals to positive 'not-' equivalents.

    Note: With the revised goal evaluation semantics, negative literals can be
    evaluated directly (LiteralGoal(~P) is satisfied iff P is absent).
    These conversion utilities may still be used elsewhere (e.g., to compile away
    negation for planners/heuristics that require positive-only predicates).
    """

    def test_convert_simple_negative_literal(self):
        from railroad.core import create_positive_fluent_mapping, convert_goal_to_positive_preconditions

        neg_fluent = ~F("free r1")
        goal = LiteralGoal(neg_fluent)

        neg_to_pos = create_positive_fluent_mapping({F("free r1")})
        converted_goal = convert_goal_to_positive_preconditions(goal, neg_to_pos)

        converted_literals = converted_goal.get_all_literals()
        assert len(converted_literals) == 1
        converted_fluent = list(converted_literals)[0]
        assert converted_fluent.name == "not-free"
        assert not converted_fluent.negated

    def test_convert_nested_goals_with_negatives(self):
        from railroad.core import create_positive_fluent_mapping, convert_goal_to_positive_preconditions

        goal = AndGoal([
            LiteralGoal(F("at r1 kitchen")),
            OrGoal([LiteralGoal(~F("free r1")), LiteralGoal(~F("holding r1 cup"))]),
        ])

        neg_to_pos = create_positive_fluent_mapping({F("free r1"), F("holding r1 cup")})
        converted_goal = convert_goal_to_positive_preconditions(goal, neg_to_pos)

        assert converted_goal.get_type() == GoalType.AND

        all_literals = converted_goal.get_all_literals()
        literal_names = {f.name for f in all_literals}
        assert "at" in literal_names
        assert "not-free" in literal_names
        assert "not-holding" in literal_names

        for lit in all_literals:
            if lit.name.startswith("not-"):
                assert not lit.negated

    def test_unconverted_fluents_pass_through(self):
        from railroad.core import create_positive_fluent_mapping, convert_goal_to_positive_preconditions

        goal = LiteralGoal(~F("at r1 kitchen"))
        neg_to_pos = create_positive_fluent_mapping({F("free r1")})  # doesn't include 'at'
        converted_goal = convert_goal_to_positive_preconditions(goal, neg_to_pos)

        converted_literals = converted_goal.get_all_literals()
        assert len(converted_literals) == 1
        converted_fluent = list(converted_literals)[0]
        assert converted_fluent.negated
        assert converted_fluent.name == "at"


# -----------------------------
# Operator overloading (&, |, ~)
# -----------------------------

class TestFluentOperatorOverloading:
    """Tests for & and | operator overloading on Fluent and Goal classes."""

    def test_mixed_and_or(self):
        goal = (Fluent("a") | Fluent("b")) & Fluent("c")
        assert goal.get_type() == GoalType.AND
        assert len(goal.get_all_literals()) == 3

    def test_mixed_or_and(self):
        goal = (Fluent("a") & Fluent("b")) | Fluent("c")
        assert goal.get_type() == GoalType.OR
        assert len(goal.get_all_literals()) == 3

    def test_negated_fluent_operator_evaluation(self):
        # Goal: at kitchen AND not holding cup
        goal = Fluent("at r1 kitchen") & ~Fluent("holding r1 cup")

        # Holding cup => negated literal fails => whole AND fails
        assert goal.evaluate({Fluent("at r1 kitchen"), Fluent("holding r1 cup")}) is False

        # Not holding cup => negated literal satisfied => AND satisfied
        assert goal.evaluate({Fluent("at r1 kitchen")}) is True

    def test_goal_and_fluent_chain_flattens(self):
        goal1 = Fluent("a") & Fluent("b")
        goal2 = goal1 & Fluent("c")
        assert goal2.get_type() == GoalType.AND
        assert len(list(goal2.children())) == 3

    def test_goal_or_fluent_chain_flattens(self):
        goal1 = Fluent("a") | Fluent("b")
        goal2 = goal1 | Fluent("c")
        assert goal2.get_type() == GoalType.OR
        assert len(list(goal2.children())) == 3

    def test_complex_nested_expression(self):
        goal = (Fluent("a") & Fluent("b")) | (Fluent("c") & Fluent("d"))
        assert goal.get_type() == GoalType.OR
        children = list(goal.children())
        assert len(children) == 2
        for child in children:
            assert child.get_type() == GoalType.AND

    @pytest.mark.slow
    def test_operator_with_planner(self):
        objects_by_type = {"robot": ["r1"], "location": ["start", "kitchen", "bedroom"]}
        move_op = construct_move_visited_operator(lambda *args: 5.0)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(time=0, fluents={Fluent("at r1 start"), Fluent("free r1")})

        # Goal: visit kitchen OR bedroom
        goal = Fluent("visited kitchen") | Fluent("visited bedroom")

        state = _run_mcts_until(goal, initial_state, all_actions, max_steps=20, max_iterations=400)
        assert goal.evaluate(state.fluents), "OR goal created with | should be achievable"
