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
