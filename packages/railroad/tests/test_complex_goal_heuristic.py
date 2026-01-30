"""
Tests for FF heuristic with negative/negated goal literals.

This module tests the heuristic computation when goals contain negated fluents,
such as "clear the table" scenarios where the goal is NOT(at obj table).
"""

from functools import reduce
from operator import and_

from mrppddl.core import (
    Fluent as F,
    State,
    Operator,
    Effect,
    extract_negative_preconditions,
    create_positive_fluent_mapping,
    convert_goal_to_positive_preconditions,
    convert_state_to_positive_preconditions,
    convert_action_to_positive_preconditions,
    convert_action_effects,
)
from mrppddl._bindings import (
    Goal,
    LiteralGoal,
    GoalType,
)
from mrppddl.core import ff_heuristic
from environments.operators import (
    construct_move_operator_nonblocking,
    construct_pick_operator_nonblocking,
    construct_place_operator_nonblocking,
)


class TestNegativeGoalHeuristic:
    """Tests for heuristic computation with negative goal literals."""

    def test_negative_literal_goal_basic(self):
        """Test that a negative literal goal can be created and evaluated."""
        # Goal: NOT(at Book table)
        goal = LiteralGoal(~F("at Book table"))

        # State where Book IS at table - goal should NOT be satisfied
        state_with_book = State(
            time=0,
            fluents={F("at Book table")}
        )
        assert goal.evaluate(state_with_book.fluents) is False

        # State where Book is NOT at table - goal SHOULD be satisfied
        state_without_book = State(
            time=0,
            fluents={F("at Book shelf")}
        )
        assert goal.evaluate(state_without_book.fluents) is True

    def test_and_of_negative_literals(self):
        """Test AND goal with multiple negative literals."""
        objects = ["Book", "Mug", "Vase"]
        goal = reduce(and_, [~F(f"at {obj} table") for obj in objects])

        # Verify goal structure
        assert isinstance(goal, Goal) and goal.get_type() == GoalType.AND

        # State where all objects are at table - goal NOT satisfied
        state_all_on_table = State(
            time=0,
            fluents={F("at Book table"), F("at Mug table"), F("at Vase table")}
        )
        assert goal.evaluate(state_all_on_table.fluents) is False

        # State where no objects are at table - goal satisfied
        state_cleared = State(
            time=0,
            fluents={F("at Book shelf"), F("at Mug shelf"), F("at Vase shelf")}
        )
        assert goal.evaluate(state_cleared.fluents) is True

        # State where one object still on table - goal NOT satisfied
        state_partial = State(
            time=0,
            fluents={F("at Book shelf"), F("at Mug table"), F("at Vase shelf")}
        )
        assert goal.evaluate(state_partial.fluents) is False

    def test_ff_heuristic_with_negative_goal(self):
        """Test ff_heuristic with negative goals after conversion.

        Negative goals (e.g., NOT at Book table) must be converted to positive
        equivalents before calling the heuristic. The conversion flow is:
        1. Extract negative fluents from the goal
        2. Create a mapping to positive equivalents
        3. Convert actions, state, and goal using the mapping
        4. Call ff_heuristic with converted inputs
        """
        objects_by_type = {
            "robot": ["r1"],
            "location": ["table", "shelf"],
            "object": ["Book"],
        }

        pick_op = construct_pick_operator_nonblocking(1.0)
        move_op = construct_move_operator_nonblocking(1.0)
        all_actions = pick_op.instantiate(objects_by_type) + move_op.instantiate(objects_by_type)

        initial_state = State(
            time=0,
            fluents={
                F("free r1"),
                F("at r1 table"),
                F("at Book table"),
            }
        )

        # Negative goal: Book NOT at table
        goal = LiteralGoal(~F("at Book table"))

        # Extract negative fluents and create mapping
        from mrppddl.core import extract_negative_goal_fluents
        negative_fluents = extract_negative_goal_fluents(goal)

        # Also include negatives from action preconditions
        neg_from_actions = extract_negative_preconditions(all_actions)
        all_negative_fluents = negative_fluents | neg_from_actions

        neg_to_pos_mapping = create_positive_fluent_mapping(all_negative_fluents)

        # Convert actions, state, and goal
        converted_actions = []
        for action in all_actions:
            action_with_preconds = convert_action_to_positive_preconditions(action, neg_to_pos_mapping)
            action_converted = convert_action_effects(action_with_preconds, neg_to_pos_mapping)
            converted_actions.append(action_converted)

        converted_state = convert_state_to_positive_preconditions(initial_state, neg_to_pos_mapping)
        converted_goal = convert_goal_to_positive_preconditions(goal, neg_to_pos_mapping)

        h_value = ff_heuristic(converted_state, converted_goal, converted_actions)

        print(f"Heuristic for negative goal (after conversion): {h_value}")
        # The goal is achievable by picking up the book
        # Heuristic should return a finite positive value
        assert h_value < float('inf'), \
            f"Heuristic should be finite for achievable negative goal, got {h_value}"
        assert h_value > 0, "Should need at least one action (pick) to achieve goal"

class TestMCTSPlannerWithNegativeGoals:
    """Tests for MCTSPlanner with negative goal literals."""

    def test_mcts_planner_with_negative_goal(self):
        """Test that MCTSPlanner works with negative goals."""
        from mrppddl.planner import MCTSPlanner

        objects_by_type = {
            "robot": ["r1"],
            "location": ["table", "shelf"],
            "object": ["Book"],
        }

        pick_op = construct_pick_operator_nonblocking(1.0)
        move_op = construct_move_operator_nonblocking(1.0)
        place_op = construct_place_operator_nonblocking(1.0)
        all_actions = (
            pick_op.instantiate(objects_by_type) +
            move_op.instantiate(objects_by_type) +
            place_op.instantiate(objects_by_type)
        )

        initial_state = State(
            time=0,
            fluents={
                F("free r1"),
                F("at r1 table"),
                F("at Book table"),
            }
        )

        # Negative goal: Book NOT at table
        goal = LiteralGoal(~F("at Book table"))

        # Create planner
        mcts = MCTSPlanner(all_actions)

        # Try to plan
        action_name = mcts(initial_state, goal, max_iterations=1000, c=10)

        print(f"Selected action: {action_name}")

        # Should select pick action to remove Book from table
        assert action_name != 'NONE', "Planner should find an action for achievable goal"
        assert 'pick' in action_name.lower(), f"Expected pick action, got {action_name}"

    def test_mcts_planner_clear_table_multiple_objects(self):
        """Test MCTSPlanner with clear table goal (multiple negative literals)."""
        from mrppddl.planner import MCTSPlanner

        objects_by_type = {
            "robot": ["r1"],
            "location": ["table", "shelf"],
            "object": ["Book", "Mug"],
        }

        pick_op = construct_pick_operator_nonblocking(1.0)
        move_op = construct_move_operator_nonblocking(1.0)
        place_op = construct_place_operator_nonblocking(1.0)
        all_actions = (
            pick_op.instantiate(objects_by_type) +
            move_op.instantiate(objects_by_type) +
            place_op.instantiate(objects_by_type)
        )

        initial_state = State(
            time=0,
            fluents={
                F("free r1"),
                F("at r1 table"),
                F("at Book table"),
                F("at Mug table"),
            }
        )

        # Goal: clear the table
        objects_to_clear = ["Book", "Mug"]
        goal = reduce(and_, [~F(f"at {obj} table") for obj in objects_to_clear])

        # Create planner
        mcts = MCTSPlanner(all_actions)

        # Try to plan
        action_name = mcts(initial_state, goal, max_iterations=2000, c=10)

        print(f"Selected action: {action_name}")

        # Should select a pick action
        assert action_name != 'NONE', "Planner should find an action for achievable goal"


class TestNegativeGoalSolutions:
    """Tests exploring potential solutions for negative goal heuristics."""

    def test_reframe_as_positive_goal(self):
        """Test reframing negative goal as positive goal.

        Instead of NOT(at Book table), use a "cleared" predicate:
        Goal becomes: cleared(Book, table) or at(Book, elsewhere)
        """
        objects_by_type = {
            "robot": ["r1"],
            "location": ["table", "shelf"],
            "object": ["Book"],
        }

        # Modified pick operator that also adds a "not-at" fluent
        # (This must be custom since the standard pick operator doesn't add this fluent)
        pick_op_with_not_at = Operator(
            name="pick",
            parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
            preconditions=[
                F("at ?r ?loc"),
                F("free ?r"),
                F("at ?obj ?loc"),
                ~F("hand-full ?r"),
            ],
            effects=[
                Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?obj ?loc")}),
                Effect(
                    time=1.0,
                    resulting_fluents={
                        F("free ?r"),
                        F("holding ?r ?obj"),
                        F("hand-full ?r"),
                        # Add positive fluent to track "not at location"
                        F("not-at ?obj ?loc"),
                    },
                ),
            ],
        )

        move_op = construct_move_operator_nonblocking(1.0)
        all_actions = pick_op_with_not_at.instantiate(objects_by_type) + move_op.instantiate(objects_by_type)

        initial_state = State(
            time=0,
            fluents={
                F("free r1"),
                F("at r1 table"),
                F("at Book table"),
            }
        )

        # Reframed goal: use positive "not-at" fluent
        goal = LiteralGoal(F("not-at Book table"))

        h_value = ff_heuristic(initial_state, goal, all_actions)
        print(f"\nReframed goal heuristic: {h_value}")

        # This should work because "not-at Book table" is added by pick action
        assert h_value < float('inf'), \
            f"Reframed positive goal should have finite heuristic, got {h_value}"
        assert h_value > 0, "Should need at least one action to clear"
