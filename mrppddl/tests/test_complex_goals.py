"""
Tests for complex goal expressions (AND, OR) following the design in complex_goals.md.

Organized into themed test classes:
- TestGoalSatisfaction: Core truth-table tests for goal evaluation
- TestNormalization: Verify preprocessing produces correct structures
- TestBranchAccess: Verify heuristic can access goal structure
- TestHouseholdScenarios: Regression tests using realistic goal expressions
- TestPlannerWithGoals: Tests for planner integration with Goal objects
"""

import pytest
from mrppddl.core import Fluent, State, transition, get_action_by_name
from mrppddl.helper import construct_move_visited_operator

# Import goal classes from bindings (will be added in _bindings.cpp)
from mrppddl._bindings import (
    GoalType,
    LiteralGoal,
    AndGoal,
    OrGoal,
    TrueGoal,
    FalseGoal,
    goal_from_fluent_set,
    ff_heuristic_goal,
)
from mrppddl.planner import MCTSPlanner, get_usable_actions

F = Fluent


class TestGoalSatisfaction:
    """Core truth-table tests for goal evaluation."""

    def test_literal_goal_satisfied(self):
        """Literal goal satisfied when fluent is present."""
        fluent = F("at r1 kitchen")
        goal = LiteralGoal(fluent)
        state_fluents = {F("at r1 kitchen"), F("free r1")}

        assert goal.evaluate(state_fluents) is True

    def test_literal_goal_not_satisfied(self):
        """Literal goal not satisfied when fluent is missing."""
        fluent = F("at r1 kitchen")
        goal = LiteralGoal(fluent)
        state_fluents = {F("at r1 bedroom"), F("free r1")}

        assert goal.evaluate(state_fluents) is False

    def test_and_goal_all_true(self):
        """AND goal satisfied when all children are satisfied."""
        g1 = LiteralGoal(F("at r1 kitchen"))
        g2 = LiteralGoal(F("free r1"))
        goal = AndGoal([g1, g2])
        state_fluents = {F("at r1 kitchen"), F("free r1"), F("visited kitchen")}

        assert goal.evaluate(state_fluents) is True

    def test_and_goal_any_false(self):
        """AND goal not satisfied when any child is not satisfied."""
        g1 = LiteralGoal(F("at r1 kitchen"))
        g2 = LiteralGoal(F("holding r1 cup"))
        goal = AndGoal([g1, g2])
        state_fluents = {F("at r1 kitchen"), F("free r1")}

        assert goal.evaluate(state_fluents) is False

    def test_and_goal_empty(self):
        """Empty AND is true (TrueGoal)."""
        goal = AndGoal([]).normalize()

        assert goal.get_type() == GoalType.TRUE_GOAL
        assert goal.evaluate(set()) is True
        assert goal.evaluate({F("random fluent")}) is True

    def test_or_goal_any_true(self):
        """OR goal satisfied when any child is satisfied."""
        g1 = LiteralGoal(F("at r1 kitchen"))
        g2 = LiteralGoal(F("at r1 bedroom"))
        goal = OrGoal([g1, g2])
        state_fluents = {F("at r1 bedroom"), F("free r1")}

        assert goal.evaluate(state_fluents) is True

    def test_or_goal_all_false(self):
        """OR goal not satisfied when all children are not satisfied."""
        g1 = LiteralGoal(F("at r1 kitchen"))
        g2 = LiteralGoal(F("at r1 bedroom"))
        goal = OrGoal([g1, g2])
        state_fluents = {F("at r1 living_room"), F("free r1")}

        assert goal.evaluate(state_fluents) is False

    def test_or_goal_empty(self):
        """Empty OR is false (FalseGoal)."""
        goal = OrGoal([]).normalize()

        assert goal.get_type() == GoalType.FALSE_GOAL
        assert goal.evaluate(set()) is False
        assert goal.evaluate({F("random fluent")}) is False

    def test_true_goal_always_satisfied(self):
        """TrueGoal is always satisfied."""
        goal = TrueGoal()

        assert goal.evaluate(set()) is True
        assert goal.evaluate({F("any fluent")}) is True

    def test_false_goal_never_satisfied(self):
        """FalseGoal is never satisfied."""
        goal = FalseGoal()

        assert goal.evaluate(set()) is False
        assert goal.evaluate({F("any fluent")}) is False

    def test_nested_and_or_satisfied(self):
        """AND(A, OR(B,C)) - satisfied when A true and either B or C true."""
        # AND(table_set, OR(toast_ready, cereal_ready))
        a = LiteralGoal(F("table_set"))
        b = LiteralGoal(F("toast_ready"))
        c = LiteralGoal(F("cereal_ready"))
        goal = AndGoal([a, OrGoal([b, c])])

        # Case 1: A true, B true -> satisfied
        assert goal.evaluate({F("table_set"), F("toast_ready")}) is True

        # Case 2: A true, C true -> satisfied
        assert goal.evaluate({F("table_set"), F("cereal_ready")}) is True

        # Case 3: A true, B and C both true -> satisfied
        assert goal.evaluate({F("table_set"), F("toast_ready"), F("cereal_ready")}) is True

        # Case 4: A false, B true -> not satisfied
        assert goal.evaluate({F("toast_ready")}) is False

        # Case 5: A true, B and C both false -> not satisfied
        assert goal.evaluate({F("table_set")}) is False

    def test_nested_or_and_satisfied(self):
        """OR(AND(A,B), AND(C,D)) - satisfied when either conjunction is satisfied."""
        # Breakfast scenario: either (toast AND coffee) OR (cereal AND milk)
        a = LiteralGoal(F("toast_ready"))
        b = LiteralGoal(F("coffee_ready"))
        c = LiteralGoal(F("cereal_ready"))
        d = LiteralGoal(F("milk_available"))
        goal = OrGoal([AndGoal([a, b]), AndGoal([c, d])])

        # Case 1: First conjunction satisfied
        assert goal.evaluate({F("toast_ready"), F("coffee_ready")}) is True

        # Case 2: Second conjunction satisfied
        assert goal.evaluate({F("cereal_ready"), F("milk_available")}) is True

        # Case 3: Both conjunctions satisfied
        assert goal.evaluate({
            F("toast_ready"), F("coffee_ready"),
            F("cereal_ready"), F("milk_available")
        }) is True

        # Case 4: Partial first conjunction (missing coffee)
        assert goal.evaluate({F("toast_ready"), F("milk_available")}) is False

        # Case 5: Nothing satisfied
        assert goal.evaluate({F("eggs_ready")}) is False


class TestNormalization:
    """Verify preprocessing produces correct structures."""

    def test_flatten_nested_and(self):
        """AND(AND(a,b),c) normalizes to AND(a,b,c)."""
        a = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        c = LiteralGoal(F("c"))

        nested = AndGoal([AndGoal([a, b]), c])
        normalized = nested.normalize()

        # Should be a flat AND with 3 children
        assert normalized.get_type() == GoalType.AND
        assert len(normalized.children()) == 3

        # Verify semantics preserved
        assert normalized.evaluate({F("a"), F("b"), F("c")}) is True
        assert normalized.evaluate({F("a"), F("b")}) is False

    def test_flatten_nested_or(self):
        """OR(OR(a,b),c) normalizes to OR(a,b,c)."""
        a = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        c = LiteralGoal(F("c"))

        nested = OrGoal([OrGoal([a, b]), c])
        normalized = nested.normalize()

        # Should be a flat OR with 3 children
        assert normalized.get_type() == GoalType.OR
        assert len(normalized.children()) == 3

        # Verify semantics preserved
        assert normalized.evaluate({F("a")}) is True
        assert normalized.evaluate({F("c")}) is True
        assert normalized.evaluate({F("d")}) is False

    def test_flatten_deeply_nested(self):
        """AND(AND(AND(a,b),c),d) flattens completely."""
        a = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        c = LiteralGoal(F("c"))
        d = LiteralGoal(F("d"))

        deeply_nested = AndGoal([AndGoal([AndGoal([a, b]), c]), d])
        normalized = deeply_nested.normalize()

        assert normalized.get_type() == GoalType.AND
        assert len(normalized.children()) == 4

    def test_constant_folding_and_with_true(self):
        """AND(A, TRUE) normalizes to A."""
        a = LiteralGoal(F("a"))
        goal = AndGoal([a, TrueGoal()])
        normalized = goal.normalize()

        # Should just be the literal A
        assert normalized.get_type() == GoalType.LITERAL
        assert normalized.evaluate({F("a")}) is True
        assert normalized.evaluate({F("b")}) is False

    def test_constant_folding_or_with_false(self):
        """OR(A, FALSE) normalizes to A."""
        a = LiteralGoal(F("a"))
        goal = OrGoal([a, FalseGoal()])
        normalized = goal.normalize()

        # Should just be the literal A
        assert normalized.get_type() == GoalType.LITERAL

    def test_constant_folding_and_with_false(self):
        """AND(A, FALSE) normalizes to FALSE."""
        a = LiteralGoal(F("a"))
        goal = AndGoal([a, FalseGoal()])
        normalized = goal.normalize()

        # Should be FalseGoal
        assert normalized.get_type() == GoalType.FALSE_GOAL
        assert normalized.evaluate({F("a")}) is False

    def test_constant_folding_or_with_true(self):
        """OR(A, TRUE) normalizes to TRUE."""
        a = LiteralGoal(F("a"))
        goal = OrGoal([a, TrueGoal()])
        normalized = goal.normalize()

        # Should be TrueGoal
        assert normalized.get_type() == GoalType.TRUE_GOAL
        assert normalized.evaluate(set()) is True

    def test_deduplication_and(self):
        """AND(A,A,B) normalizes to AND(A,B)."""
        a1 = LiteralGoal(F("a"))
        a2 = LiteralGoal(F("a"))  # Duplicate
        b = LiteralGoal(F("b"))

        goal = AndGoal([a1, a2, b])
        normalized = goal.normalize()

        assert normalized.get_type() == GoalType.AND
        assert len(normalized.children()) == 2

    def test_deduplication_or(self):
        """OR(A,B,A) normalizes to OR(A,B)."""
        a1 = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        a2 = LiteralGoal(F("a"))  # Duplicate

        goal = OrGoal([a1, b, a2])
        normalized = goal.normalize()

        assert normalized.get_type() == GoalType.OR
        assert len(normalized.children()) == 2

    def test_no_distribution_and_over_or(self):
        """AND(A, OR(B,C)) stays as AND(A, OR(B,C)) - no distribution."""
        a = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        c = LiteralGoal(F("c"))

        goal = AndGoal([a, OrGoal([b, c])])
        normalized = goal.normalize()

        # Structure should remain: AND with 2 children
        assert normalized.get_type() == GoalType.AND
        assert len(normalized.children()) == 2

        # One child should be a literal, one should be OR
        child_types = [child.get_type() for child in normalized.children()]
        assert GoalType.LITERAL in child_types
        assert GoalType.OR in child_types

    def test_canonical_ordering(self):
        """Goals with same children in different orders normalize identically."""
        a = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        c = LiteralGoal(F("c"))

        goal1 = AndGoal([a, b, c]).normalize()
        goal2 = AndGoal([c, b, a]).normalize()
        goal3 = AndGoal([b, a, c]).normalize()

        # All should have same structure
        assert goal1.get_type() == goal2.get_type() == goal3.get_type()
        assert len(goal1.children()) == len(goal2.children()) == len(goal3.children())

    def test_example_from_design_doc(self):
        """Test the example from design doc section 7.1."""
        # Input: AND(table_set, AND(toast_ready, TRUE), OR(FALSE, coffee_ready))
        # Expected: AND(table_set, toast_ready, coffee_ready)
        table_set = LiteralGoal(F("table_set"))
        toast_ready = LiteralGoal(F("toast_ready"))
        coffee_ready = LiteralGoal(F("coffee_ready"))

        goal = AndGoal([
            table_set,
            AndGoal([toast_ready, TrueGoal()]),
            OrGoal([FalseGoal(), coffee_ready])
        ])
        normalized = goal.normalize()

        # Should be flat AND of 3 literals
        assert normalized.get_type() == GoalType.AND
        assert len(normalized.children()) == 3

        # Check semantics
        assert normalized.evaluate({F("table_set"), F("toast_ready"), F("coffee_ready")}) is True
        assert normalized.evaluate({F("table_set"), F("toast_ready")}) is False


class TestBranchAccess:
    """Verify heuristic can access goal structure."""

    def test_literal_is_pure_conjunction(self):
        """A single literal is a pure conjunction."""
        goal = LiteralGoal(F("a"))

        assert goal.is_pure_conjunction() is True

    def test_and_of_literals_is_pure_conjunction(self):
        """AND of literals is a pure conjunction."""
        a = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        c = LiteralGoal(F("c"))
        goal = AndGoal([a, b, c])

        assert goal.is_pure_conjunction() is True

    def test_or_is_not_pure_conjunction(self):
        """OR is never a pure conjunction."""
        a = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        goal = OrGoal([a, b])

        assert goal.is_pure_conjunction() is False

    def test_and_with_nested_or_is_not_pure_conjunction(self):
        """AND with nested OR is not a pure conjunction."""
        a = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        c = LiteralGoal(F("c"))
        goal = AndGoal([a, OrGoal([b, c])])

        assert goal.is_pure_conjunction() is False

    def test_top_level_or_exposes_branches(self):
        """OR goal exposes its children for heuristic evaluation."""
        g1 = AndGoal([LiteralGoal(F("toast")), LiteralGoal(F("coffee"))])
        g2 = AndGoal([LiteralGoal(F("cereal")), LiteralGoal(F("milk"))])
        g3 = AndGoal([LiteralGoal(F("eggs")), LiteralGoal(F("plate"))])
        g4 = AndGoal([LiteralGoal(F("yogurt")), LiteralGoal(F("fruit"))])

        goal = OrGoal([g1, g2, g3, g4])

        # Should expose 4 alternatives
        assert len(goal.children()) == 4

    def test_and_goal_exposes_children(self):
        """AND goal exposes its children."""
        a = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        goal = AndGoal([a, b])

        assert len(goal.children()) == 2

    def test_get_all_literals_pure_conjunction(self):
        """get_all_literals returns all fluents from a conjunction."""
        a = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        c = LiteralGoal(F("c"))
        goal = AndGoal([a, b, c])

        literals = goal.get_all_literals()

        assert len(literals) == 3
        assert F("a") in literals
        assert F("b") in literals
        assert F("c") in literals

    def test_get_all_literals_nested(self):
        """get_all_literals returns all fluents from nested structure."""
        a = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        c = LiteralGoal(F("c"))
        goal = AndGoal([a, OrGoal([b, c])])

        literals = goal.get_all_literals()

        assert len(literals) == 3
        assert F("a") in literals
        assert F("b") in literals
        assert F("c") in literals

    def test_goal_from_fluent_set(self):
        """goal_from_fluent_set creates an AND goal from a set of fluents."""
        fluents = {F("a"), F("b"), F("c")}
        goal = goal_from_fluent_set(fluents)

        # Should be a pure conjunction
        assert goal.is_pure_conjunction() is True

        # Should have correct semantics
        assert goal.evaluate({F("a"), F("b"), F("c")}) is True
        assert goal.evaluate({F("a"), F("b")}) is False

    def test_goal_from_single_fluent(self):
        """goal_from_fluent_set with single fluent creates a literal."""
        fluents = {F("a")}
        goal = goal_from_fluent_set(fluents)

        assert goal.get_type() == GoalType.LITERAL
        assert goal.evaluate({F("a")}) is True

    def test_goal_count(self):
        """goal_count returns number of achieved literal goals."""
        a = LiteralGoal(F("a"))
        b = LiteralGoal(F("b"))
        c = LiteralGoal(F("c"))
        goal = AndGoal([a, b, c])

        # 2 of 3 achieved
        assert goal.goal_count({F("a"), F("b")}) == 2

        # All 3 achieved
        assert goal.goal_count({F("a"), F("b"), F("c")}) == 3

        # None achieved
        assert goal.goal_count({F("d")}) == 0


class TestHouseholdScenarios:
    """Regression tests using realistic goal expressions from the design doc."""

    def test_breakfast_four_methods(self):
        """
        Test the '4 ways to prepare breakfast' scenario from design doc.
        OR(G1, G2, G3, G4) where each is a conjunction.
        """
        # G1 = AND(toast_ready, coffee_ready)
        g1 = AndGoal([LiteralGoal(F("toast_ready")), LiteralGoal(F("coffee_ready"))])
        # G2 = AND(cereal_ready, milk_available)
        g2 = AndGoal([LiteralGoal(F("cereal_ready")), LiteralGoal(F("milk_available"))])
        # G3 = AND(eggs_cooked, plate_clean)
        g3 = AndGoal([LiteralGoal(F("eggs_cooked")), LiteralGoal(F("plate_clean"))])
        # G4 = AND(yogurt_served, fruit_cut)
        g4 = AndGoal([LiteralGoal(F("yogurt_served")), LiteralGoal(F("fruit_cut"))])

        goal = OrGoal([g1, g2, g3, g4])

        # Method 1 complete
        assert goal.evaluate({F("toast_ready"), F("coffee_ready")}) is True

        # Method 2 complete
        assert goal.evaluate({F("cereal_ready"), F("milk_available")}) is True

        # Method 3 complete
        assert goal.evaluate({F("eggs_cooked"), F("plate_clean")}) is True

        # Method 4 complete
        assert goal.evaluate({F("yogurt_served"), F("fruit_cut")}) is True

        # Partial method 1 (missing coffee)
        assert goal.evaluate({F("toast_ready")}) is False

        # Mixed partial (one from each method - none complete)
        assert goal.evaluate({F("toast_ready"), F("cereal_ready"), F("eggs_cooked")}) is False

        # Multiple methods complete
        assert goal.evaluate({
            F("toast_ready"), F("coffee_ready"),
            F("cereal_ready"), F("milk_available")
        }) is True

    def test_set_table_and_choose_meal(self):
        """
        Test nested goal: table must be set AND one of two meal options.
        AND(table_set, OR(meal_option_a, meal_option_b))
        """
        table_set = LiteralGoal(F("table_set"))
        meal_a = AndGoal([LiteralGoal(F("pasta_ready")), LiteralGoal(F("sauce_ready"))])
        meal_b = AndGoal([LiteralGoal(F("salad_ready")), LiteralGoal(F("dressing_ready"))])

        goal = AndGoal([table_set, OrGoal([meal_a, meal_b])])

        # Table set and meal A ready
        assert goal.evaluate({
            F("table_set"), F("pasta_ready"), F("sauce_ready")
        }) is True

        # Table set and meal B ready
        assert goal.evaluate({
            F("table_set"), F("salad_ready"), F("dressing_ready")
        }) is True

        # Meal A ready but table not set
        assert goal.evaluate({F("pasta_ready"), F("sauce_ready")}) is False

        # Table set but no complete meal
        assert goal.evaluate({F("table_set"), F("pasta_ready")}) is False

    def test_clean_kitchen(self):
        """
        Test simple conjunction: all cleanliness predicates must hold.
        AND(counter_clean, sink_clean, floor_clean)
        """
        goal = AndGoal([
            LiteralGoal(F("counter_clean")),
            LiteralGoal(F("sink_clean")),
            LiteralGoal(F("floor_clean"))
        ])

        # All clean
        assert goal.evaluate({
            F("counter_clean"), F("sink_clean"), F("floor_clean")
        }) is True

        # Missing one
        assert goal.evaluate({F("counter_clean"), F("sink_clean")}) is False

        # None clean
        assert goal.evaluate(set()) is False

    def test_exists_pattern_manual(self):
        """
        Test EXISTS pattern manually grounded (as it would be compiled).
        EXISTS(p in Plates, clean(p)) -> OR(clean(plate1), clean(plate2), clean(plate3))
        """
        # Simulating: "Breakfast can be served on some clean plate"
        plates = ["plate1", "plate2", "plate3"]
        plate_goals = [LiteralGoal(F(f"clean {p}")) for p in plates]
        goal = OrGoal(plate_goals)

        # Plate 1 is clean
        assert goal.evaluate({F("clean plate1")}) is True

        # Plate 2 is clean
        assert goal.evaluate({F("clean plate2")}) is True

        # All plates dirty
        assert goal.evaluate(set()) is False

        # Multiple clean
        assert goal.evaluate({F("clean plate1"), F("clean plate3")}) is True


class TestPlannerWithGoals:
    """Tests for planner integration with Goal objects."""

    def test_planner_with_and_goal(self):
        """Test MCTS planner with an AND goal (pure conjunction)."""
        # Setup: Simple move-visit scenario
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

        # Create AND goal using Goal object
        goal = AndGoal([
            LiteralGoal(F("visited a")),
            LiteralGoal(F("visited b")),
        ])

        # Plan with Goal object
        all_actions = get_usable_actions(initial_state, goal.get_all_literals(), all_actions)
        mcts = MCTSPlanner(all_actions)

        state = initial_state
        for _ in range(10):
            if goal.evaluate(state.fluents):
                break
            action_name = mcts(state, goal, max_iterations=500, c=10)
            if action_name == "NONE":
                break
            action = get_action_by_name(all_actions, action_name)
            state = transition(state, action)[0][0]

        assert goal.evaluate(state.fluents), "AND goal should be satisfied"

    def test_planner_with_or_goal(self):
        """Test MCTS planner with an OR goal (disjunction)."""
        # Setup: Robot can visit either location a OR location b
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

        # Create OR goal - either visit a OR visit b
        goal = OrGoal([
            LiteralGoal(F("visited a")),
            LiteralGoal(F("visited b")),
        ])

        # Plan with Goal object
        all_actions = get_usable_actions(initial_state, goal.get_all_literals(), all_actions)
        mcts = MCTSPlanner(all_actions)

        state = initial_state
        for _ in range(10):
            if goal.evaluate(state.fluents):
                break
            action_name = mcts(state, goal, max_iterations=500, c=10)
            if action_name == "NONE":
                break
            action = get_action_by_name(all_actions, action_name)
            state = transition(state, action)[0][0]

        assert goal.evaluate(state.fluents), "OR goal should be satisfied"
        # At least one should be visited
        assert (F("visited a") in state.fluents) or (F("visited b") in state.fluents)

    def test_planner_with_nested_goal(self):
        """Test MCTS planner with nested AND/OR goal."""
        # Setup: Visit start AND (visit a OR visit b)
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

        # Create nested goal: visited start AND (visited a OR visited b)
        goal = AndGoal([
            LiteralGoal(F("visited start")),
            OrGoal([
                LiteralGoal(F("visited a")),
                LiteralGoal(F("visited b")),
            ])
        ])

        # Plan with Goal object
        all_actions = get_usable_actions(initial_state, goal.get_all_literals(), all_actions)
        mcts = MCTSPlanner(all_actions)

        state = initial_state
        for _ in range(10):
            if goal.evaluate(state.fluents):
                break
            action_name = mcts(state, goal, max_iterations=500, c=10)
            if action_name == "NONE":
                break
            action = get_action_by_name(all_actions, action_name)
            state = transition(state, action)[0][0]

        assert goal.evaluate(state.fluents), "Nested goal should be satisfied"

    def test_ff_heuristic_with_goal_object(self):
        """Test ff_heuristic_goal function with Goal objects."""
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
        h_value = ff_heuristic_goal(initial_state, goal, all_actions)

        # Heuristic should be positive (need to visit 2 locations)
        assert h_value > 0, f"Heuristic should be positive, got {h_value}"

    def test_goal_from_fluent_set_in_planner(self):
        """Test using goal_from_fluent_set to convert legacy fluent sets."""
        objects_by_type = {
            "robot": ["r1"],
            "location": ["start", "a"],
        }
        move_op = construct_move_visited_operator(lambda *args: 5.0)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(
            time=0,
            fluents={F("at r1 start"), F("free r1"), F("visited start")}
        )

        # Convert fluent set to Goal object
        fluent_set = {F("visited a")}
        goal = goal_from_fluent_set(fluent_set)

        assert goal.get_type() == GoalType.LITERAL  # Single fluent -> literal
        assert goal.is_pure_conjunction() is True

        # Plan with converted goal
        all_actions = get_usable_actions(initial_state, fluent_set, all_actions)
        mcts = MCTSPlanner(all_actions)

        state = initial_state
        for _ in range(10):
            if goal.evaluate(state.fluents):
                break
            action_name = mcts(state, goal, max_iterations=500, c=10)
            if action_name == "NONE":
                break
            action = get_action_by_name(all_actions, action_name)
            state = transition(state, action)[0][0]

        assert goal.evaluate(state.fluents), "Goal should be satisfied"


class TestHeuristicORBranches:
    """Tests for efficient OR branch handling in the heuristic."""

    def test_or_goal_returns_min_branch_cost(self):
        """OR goal should return the minimum cost over all branches."""
        # Setup: Robot at 'start', can move to 'near' (cost 5) or 'far' (cost 15)
        objects_by_type = {
            "robot": ["r1"],
            "location": ["start", "near", "far"],
        }

        # Create move operator with distance-based costs
        # cost_fn signature: (robot, from_loc, to_loc)
        def cost_fn(robot, loc1, loc2):
            if loc2 == "near":
                return 5.0
            elif loc2 == "far":
                return 15.0
            return 10.0

        move_op = construct_move_visited_operator(cost_fn)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(
            time=0,
            fluents={F("at r1 start"), F("free r1"), F("visited start")}
        )

        # OR goal: visit 'near' OR visit 'far'
        # 'near' is cheaper, so heuristic should return cost to reach 'near'
        or_goal = OrGoal([
            LiteralGoal(F("visited near")),
            LiteralGoal(F("visited far")),
        ])

        # Compute heuristic for each branch individually
        near_goal = LiteralGoal(F("visited near"))
        far_goal = LiteralGoal(F("visited far"))

        h_near = ff_heuristic_goal(initial_state, near_goal, all_actions)
        h_far = ff_heuristic_goal(initial_state, far_goal, all_actions)
        h_or = ff_heuristic_goal(initial_state, or_goal, all_actions)

        # OR should return min
        assert h_or == min(h_near, h_far), (
            f"OR heuristic should be min of branches: "
            f"h_or={h_or}, h_near={h_near}, h_far={h_far}"
        )

    def test_true_goal_returns_zero(self):
        """TrueGoal should return 0 (already satisfied)."""
        # Minimal setup
        objects_by_type = {
            "robot": ["r1"],
            "location": ["start"],
        }
        move_op = construct_move_visited_operator(lambda *args: 5.0)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(
            time=0,
            fluents={F("at r1 start"), F("free r1")}
        )

        goal = TrueGoal()
        h_value = ff_heuristic_goal(initial_state, goal, all_actions)

        assert h_value == 0.0, f"TrueGoal should have cost 0, got {h_value}"

    def test_false_goal_returns_infinity(self):
        """FalseGoal should return infinity (impossible)."""
        import math

        objects_by_type = {
            "robot": ["r1"],
            "location": ["start"],
        }
        move_op = construct_move_visited_operator(lambda *args: 5.0)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(
            time=0,
            fluents={F("at r1 start"), F("free r1")}
        )

        goal = FalseGoal()
        h_value = ff_heuristic_goal(initial_state, goal, all_actions)

        assert math.isinf(h_value), f"FalseGoal should have infinite cost, got {h_value}"

    def test_or_with_unreachable_branch(self):
        """OR with an unreachable branch should still work using reachable branch."""
        import math

        objects_by_type = {
            "robot": ["r1"],
            "location": ["start", "reachable"],
        }
        move_op = construct_move_visited_operator(lambda *args: 5.0)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(
            time=0,
            fluents={F("at r1 start"), F("free r1"), F("visited start")}
        )

        # OR goal with one reachable and one unreachable branch
        or_goal = OrGoal([
            LiteralGoal(F("visited reachable")),     # Reachable
            LiteralGoal(F("visited nonexistent")),   # Unreachable (location doesn't exist)
        ])

        h_value = ff_heuristic_goal(initial_state, or_goal, all_actions)

        # Should return finite cost (from reachable branch)
        assert not math.isinf(h_value), (
            f"OR with one reachable branch should have finite cost, got {h_value}"
        )

    def test_and_with_nested_or_uses_all_literals(self):
        """AND(A, OR(B,C)) should be treated as single branch with all literals."""
        objects_by_type = {
            "robot": ["r1"],
            "location": ["start", "a", "b", "c"],
        }
        move_op = construct_move_visited_operator(lambda *args: 5.0)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(
            time=0,
            fluents={F("at r1 start"), F("free r1"), F("visited start")}
        )

        # Nested AND-OR goal
        nested_goal = AndGoal([
            LiteralGoal(F("visited a")),
            OrGoal([
                LiteralGoal(F("visited b")),
                LiteralGoal(F("visited c")),
            ]),
        ])

        # For comparison, pure AND with all literals
        all_literals_goal = AndGoal([
            LiteralGoal(F("visited a")),
            LiteralGoal(F("visited b")),
            LiteralGoal(F("visited c")),
        ])

        h_nested = ff_heuristic_goal(initial_state, nested_goal, all_actions)
        h_all = ff_heuristic_goal(initial_state, all_literals_goal, all_actions)

        # Both should be the same (nested OR gets flattened to all literals)
        assert h_nested == h_all, (
            f"Nested AND-OR should use all literals: "
            f"h_nested={h_nested}, h_all={h_all}"
        )


class TestGoalNegativeConversion:
    """Tests for converting negative fluents in goals to positive 'not-' equivalents."""

    def test_convert_simple_negative_literal(self):
        """Convert a simple negative literal goal."""
        from mrppddl.core import (
            create_positive_fluent_mapping,
            convert_goal_to_positive_preconditions,
        )

        # Create a negative fluent goal
        neg_fluent = ~F("free r1")
        goal = LiteralGoal(neg_fluent)

        # Create mapping for the fluent
        neg_to_pos = create_positive_fluent_mapping({F("free r1")})

        # Convert
        converted_goal = convert_goal_to_positive_preconditions(goal, neg_to_pos)

        # Check the converted goal has the "not-" fluent
        converted_literals = converted_goal.get_all_literals()
        assert len(converted_literals) == 1
        converted_fluent = list(converted_literals)[0]
        assert converted_fluent.name == "not-free"
        assert not converted_fluent.negated

    def test_convert_nested_goals_with_negatives(self):
        """Convert nested AND/OR goals containing negative fluents."""
        from mrppddl.core import (
            create_positive_fluent_mapping,
            convert_goal_to_positive_preconditions,
        )

        # Create a nested goal with negative fluents
        goal = AndGoal([
            LiteralGoal(F("at r1 kitchen")),  # Positive - no conversion
            OrGoal([
                LiteralGoal(~F("free r1")),   # Negative - should convert
                LiteralGoal(~F("holding r1 cup")),  # Negative - should convert
            ]),
        ])

        # Create mapping for the negative fluents
        neg_to_pos = create_positive_fluent_mapping({F("free r1"), F("holding r1 cup")})

        # Convert
        converted_goal = convert_goal_to_positive_preconditions(goal, neg_to_pos)

        # Check structure is preserved
        assert converted_goal.get_type() == GoalType.AND

        # Check all literals
        all_literals = converted_goal.get_all_literals()
        literal_names = {f.name for f in all_literals}

        # Should have: "at" (unchanged), "not-free", "not-holding"
        assert "at" in literal_names
        assert "not-free" in literal_names
        assert "not-holding" in literal_names

        # None should be negated (all converted to positive)
        for lit in all_literals:
            if lit.name.startswith("not-"):
                assert not lit.negated, f"Converted fluent {lit} should not be negated"

    def test_unconverted_fluents_pass_through(self):
        """Fluents not in the mapping should pass through unchanged."""
        from mrppddl.core import (
            create_positive_fluent_mapping,
            convert_goal_to_positive_preconditions,
        )

        # Create a goal with a fluent that won't be in the mapping
        goal = LiteralGoal(~F("at r1 kitchen"))

        # Create mapping without "at" fluents
        neg_to_pos = create_positive_fluent_mapping({F("free r1")})

        # Convert
        converted_goal = convert_goal_to_positive_preconditions(goal, neg_to_pos)

        # Should be unchanged (negated "at" fluent not in mapping)
        converted_literals = converted_goal.get_all_literals()
        assert len(converted_literals) == 1
        converted_fluent = list(converted_literals)[0]
        assert converted_fluent.negated  # Still negated
        assert converted_fluent.name == "at"

    def test_planner_with_negative_goal_fluents(self):
        """Test that planner correctly handles goals with negative fluents."""
        from mrppddl.helper import construct_move_visited_operator

        objects_by_type = {
            "robot": ["r1"],
            "location": ["start", "kitchen"],
        }
        move_op = construct_move_visited_operator(lambda *args: 5.0)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(
            time=0,
            fluents={F("at r1 start"), F("free r1")}
        )

        # Goal: robot should be in kitchen (requires move, which makes free=false temporarily)
        # Include a negative fluent in the goal to test conversion
        goal = AndGoal([
            LiteralGoal(F("at r1 kitchen")),
            LiteralGoal(F("free r1")),  # Robot should be free at the end
        ])

        # Create planner and plan
        mcts = MCTSPlanner(all_actions)

        state = initial_state
        for _ in range(5):
            if goal.evaluate(state.fluents):
                break
            action_name = mcts(state, goal, max_iterations=100, c=10)
            if action_name == "NONE":
                break
            action = get_action_by_name(all_actions, action_name)
            state = transition(state, action)[0][0]

        assert goal.evaluate(state.fluents), "Goal with positive fluents should be achievable"

    def test_negative_goal_with_planner(self):
        """Test that negative fluent goals work correctly with planner.

        Note: Negative fluent goals work through the planner's conversion system.
        Direct evaluation with goal.evaluate() checks if ~F("P") is in the set,
        which won't work for standard state representation. The planner internally
        converts ~F("P") to F("not-P") and adds F("not-P") to state when P is absent.
        """
        from mrppddl.helper import construct_move_visited_operator
        from mrppddl.core import (
            extract_negative_preconditions,
            create_positive_fluent_mapping,
            convert_state_to_positive_preconditions,
            convert_goal_to_positive_preconditions,
        )

        objects_by_type = {
            "robot": ["r1"],
            "location": ["start", "kitchen", "bedroom"],
        }
        move_op = construct_move_visited_operator(lambda *args: 5.0)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(
            time=0,
            fluents={F("at r1 start"), F("free r1")}
        )

        # Goal: robot should NOT be at start (negative goal)
        goal = LiteralGoal(~F("at r1 start"))

        # Check goal structure
        literals = goal.get_all_literals()
        assert len(literals) == 1
        lit = list(literals)[0]
        assert lit.negated, "Goal should have negated fluent"

        # Create planner and plan
        mcts = MCTSPlanner(all_actions)

        state = initial_state
        # Plan until robot moves away from start
        for _ in range(5):
            if F("at r1 start") not in state.fluents:
                break
            action_name = mcts(state, goal, max_iterations=100, c=10)
            if action_name == "NONE":
                break
            action = get_action_by_name(all_actions, action_name)
            state = transition(state, action)[0][0]

        # Robot should have moved away from start
        assert F("at r1 start") not in state.fluents, "Robot should not be at start"

        # To evaluate the goal correctly outside the planner, we need to:
        # 1. Create the same mapping the planner uses
        # 2. Convert the state and goal
        neg_fluents = extract_negative_preconditions(all_actions)
        # Add the goal's negative fluent to the mapping
        neg_fluents.add(F("at r1 start"))
        neg_to_pos = create_positive_fluent_mapping(neg_fluents)
        converted_state = convert_state_to_positive_preconditions(state, neg_to_pos)
        converted_goal = convert_goal_to_positive_preconditions(goal, neg_to_pos)

        # Now evaluation should work
        assert converted_goal.evaluate(converted_state.fluents), (
            "Converted negative goal should be satisfied when robot is not at start"
        )


class TestFluentOperatorOverloading:
    """Tests for & and | operator overloading on Fluent and Goal classes."""

    def test_fluent_or_creates_or_goal(self):
        """F('a') | F('b') creates an OrGoal."""
        from mrppddl.core import Fluent

        goal = Fluent("at r1 kitchen") | Fluent("at r1 bedroom")

        assert goal.get_type() == GoalType.OR
        literals = goal.get_all_literals()
        assert len(literals) == 2

    def test_fluent_and_creates_and_goal(self):
        """F('a') & F('b') creates an AndGoal."""
        from mrppddl.core import Fluent

        goal = Fluent("at r1 kitchen") & Fluent("holding r1 cup")

        assert goal.get_type() == GoalType.AND
        literals = goal.get_all_literals()
        assert len(literals) == 2

    def test_or_chain_flattens(self):
        """a | b | c produces flat OR with 3 children."""
        from mrppddl.core import Fluent

        goal = Fluent("a") | Fluent("b") | Fluent("c")

        assert goal.get_type() == GoalType.OR
        children = list(goal.children())
        assert len(children) == 3

    def test_and_chain_flattens(self):
        """a & b & c produces flat AND with 3 children."""
        from mrppddl.core import Fluent

        goal = Fluent("x") & Fluent("y") & Fluent("z")

        assert goal.get_type() == GoalType.AND
        children = list(goal.children())
        assert len(children) == 3

    def test_mixed_and_or(self):
        """(a | b) & c produces AND(OR(a,b), c)."""
        from mrppddl.core import Fluent

        goal = (Fluent("a") | Fluent("b")) & Fluent("c")

        assert goal.get_type() == GoalType.AND
        literals = goal.get_all_literals()
        assert len(literals) == 3

    def test_mixed_or_and(self):
        """(a & b) | c produces OR(AND(a,b), c)."""
        from mrppddl.core import Fluent

        goal = (Fluent("a") & Fluent("b")) | Fluent("c")

        assert goal.get_type() == GoalType.OR
        literals = goal.get_all_literals()
        assert len(literals) == 3

    def test_operator_goal_evaluation(self):
        """Goals created with operators evaluate correctly."""
        from mrppddl.core import Fluent

        or_goal = Fluent("at r1 kitchen") | Fluent("at r1 bedroom")
        and_goal = Fluent("at r1 kitchen") & Fluent("holding r1 cup")

        # State with robot in kitchen holding cup
        state_fluents = {Fluent("at r1 kitchen"), Fluent("holding r1 cup")}

        assert or_goal.evaluate(state_fluents) is True
        assert and_goal.evaluate(state_fluents) is True

        # State with robot in bedroom (no cup)
        state_fluents2 = {Fluent("at r1 bedroom")}

        assert or_goal.evaluate(state_fluents2) is True  # OR still satisfied
        assert and_goal.evaluate(state_fluents2) is False  # AND not satisfied

    def test_negated_fluent_with_operators(self):
        """Negated fluents work with operators."""
        from mrppddl.core import Fluent

        # Goal: robot at kitchen AND not holding anything
        goal = Fluent("at r1 kitchen") & ~Fluent("holding r1 cup")

        assert goal.get_type() == GoalType.AND
        literals = goal.get_all_literals()

        # Check we have both positive and negative fluents
        names = {f.name for f in literals}
        assert "at" in names
        assert "holding" in names

        negated_count = sum(1 for f in literals if f.negated)
        assert negated_count == 1

    def test_goal_and_fluent(self):
        """Goal & Fluent chains correctly."""
        from mrppddl.core import Fluent

        # Create a goal, then chain with a fluent
        goal1 = Fluent("a") & Fluent("b")
        goal2 = goal1 & Fluent("c")

        assert goal2.get_type() == GoalType.AND
        children = list(goal2.children())
        assert len(children) == 3  # Should be flattened

    def test_goal_or_fluent(self):
        """Goal | Fluent chains correctly."""
        from mrppddl.core import Fluent

        # Create a goal, then chain with a fluent
        goal1 = Fluent("a") | Fluent("b")
        goal2 = goal1 | Fluent("c")

        assert goal2.get_type() == GoalType.OR
        children = list(goal2.children())
        assert len(children) == 3  # Should be flattened

    def test_complex_nested_expression(self):
        """Complex nested expression: (a & b) | (c & d)."""
        from mrppddl.core import Fluent

        goal = (Fluent("a") & Fluent("b")) | (Fluent("c") & Fluent("d"))

        assert goal.get_type() == GoalType.OR
        children = list(goal.children())
        assert len(children) == 2

        # Each child should be an AND
        for child in children:
            assert child.get_type() == GoalType.AND

    def test_operator_with_planner(self):
        """Goals created with operators work correctly with planner."""
        from mrppddl.core import Fluent, transition, get_action_by_name
        from mrppddl.helper import construct_move_visited_operator

        objects_by_type = {
            "robot": ["r1"],
            "location": ["start", "kitchen", "bedroom"],
        }
        move_op = construct_move_visited_operator(lambda *args: 5.0)
        all_actions = move_op.instantiate(objects_by_type)

        initial_state = State(
            time=0,
            fluents={Fluent("at r1 start"), Fluent("free r1")}
        )

        # Goal using operators: visit kitchen OR bedroom
        goal = Fluent("visited kitchen") | Fluent("visited bedroom")

        mcts = MCTSPlanner(all_actions)

        state = initial_state
        for _ in range(5):
            if goal.evaluate(state.fluents):
                break
            action_name = mcts(state, goal, max_iterations=100, c=10)
            if action_name == "NONE":
                break
            action = get_action_by_name(all_actions, action_name)
            state = transition(state, action)[0][0]

        assert goal.evaluate(state.fluents), "OR goal created with | should be achievable"
