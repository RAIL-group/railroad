"""Goal formatting and display utilities.

This module provides functions for formatting and analyzing Goal objects.
"""

from typing import Set, Union

from railroad.core import Fluent
from railroad._bindings import GoalType, Goal, AndGoal, LiteralGoal


# =============================================================================
# Goal formatting and display functions
# =============================================================================


def format_goal(goal: Goal, indent: int = 0, compact: bool = False) -> str:
    """Format a goal as a readable string.

    Args:
        goal: A Goal object (LiteralGoal, AndGoal, OrGoal, etc.)
        indent: Current indentation level (for nested goals)
        compact: If True, use single-line format for simple goals

    Returns:
        A formatted string representation of the goal.

    Example output:
        AND(
          (at r1 kitchen)
          OR(
            (holding r1 cup)
            (holding r1 plate)
          )
        )
    """
    goal_type = goal.get_type()
    prefix = "  " * indent

    if goal_type == GoalType.LITERAL:
        assert isinstance(goal, LiteralGoal)
        return f"{prefix}{goal.fluent()}"

    elif goal_type == GoalType.TRUE_GOAL:
        return f"{prefix}TRUE"

    elif goal_type == GoalType.FALSE_GOAL:
        return f"{prefix}FALSE"

    elif goal_type == GoalType.AND:
        children = list(goal.children())
        if compact and len(children) <= 2 and all(
            c.get_type() == GoalType.LITERAL for c in children
        ):
            child_strs = [str(c.fluent()) for c in children if isinstance(c, LiteralGoal)]
            return f"{prefix}AND({', '.join(child_strs)})"
        else:
            lines = [f"{prefix}AND("]
            for child in children:
                lines.append(format_goal(child, indent + 1, compact))
            lines.append(f"{prefix})")
            return "\n".join(lines)

    elif goal_type == GoalType.OR:
        children = list(goal.children())
        if compact and len(children) <= 2 and all(
            c.get_type() == GoalType.LITERAL for c in children
        ):
            child_strs = [str(c.fluent()) for c in children if isinstance(c, LiteralGoal)]
            return f"{prefix}OR({', '.join(child_strs)})"
        else:
            lines = [f"{prefix}OR("]
            for child in children:
                lines.append(format_goal(child, indent + 1, compact))
            lines.append(f"{prefix})")
            return "\n".join(lines)

    else:
        return f"{prefix}<unknown goal type>"


def get_satisfied_branch(goal: Goal, fluents: Set[Fluent]) -> Union[Goal, None]:
    """Find the minimal satisfied branch of a goal.

    For OR goals, returns the first satisfied child.
    For AND goals, returns an AND of all satisfied children's branches.
    For literals, returns the literal if satisfied, None otherwise.

    Args:
        goal: A Goal object
        fluents: Set of fluents representing current state

    Returns:
        A Goal representing the satisfied portion, or None if not satisfied.
    """
    if not goal.evaluate(fluents):
        return None

    goal_type = goal.get_type()

    if goal_type == GoalType.LITERAL:
        return goal

    elif goal_type == GoalType.TRUE_GOAL:
        return goal

    elif goal_type == GoalType.FALSE_GOAL:
        return None

    elif goal_type == GoalType.OR:
        # Return the first satisfied branch
        for child in goal.children():
            if child.evaluate(fluents):
                return get_satisfied_branch(child, fluents)
        return None

    elif goal_type == GoalType.AND:
        # Return AND of all satisfied branches
        satisfied_children = []
        for child in goal.children():
            branch = get_satisfied_branch(child, fluents)
            if branch is not None:
                satisfied_children.append(branch)
        if len(satisfied_children) == 1:
            return satisfied_children[0]
        elif len(satisfied_children) > 1:
            return AndGoal(satisfied_children)
        return None

    return None


def get_best_branch(goal: Goal, fluents: Set[Fluent]) -> Goal:
    """Find the most promising branch of a goal (highest completion ratio).

    For OR goals, returns the child with highest completion ratio.
    For AND goals, returns an AND of the best branches from each child.
    For literals, returns the literal itself.

    Args:
        goal: A Goal object
        fluents: Set of fluents representing current state

    Returns:
        A Goal representing the best branch to pursue.
    """
    goal_type = goal.get_type()

    if goal_type == GoalType.LITERAL:
        return goal

    elif goal_type in (GoalType.TRUE_GOAL, GoalType.FALSE_GOAL):
        return goal

    elif goal_type == GoalType.OR:
        # Find child with best completion ratio
        best_child = None
        best_ratio = -1.0

        for child in goal.children():
            child_literals = child.get_all_literals()
            total = len(child_literals)
            achieved = child.goal_count(fluents)
            ratio = achieved / total if total > 0 else 0.0

            if ratio > best_ratio:
                best_ratio = ratio
                best_child = child

        if best_child is not None:
            return get_best_branch(best_child, fluents)
        return goal

    elif goal_type == GoalType.AND:
        # Return AND of best branches from each child
        best_children = []
        for child in goal.children():
            best_children.append(get_best_branch(child, fluents))

        if len(best_children) == 1:
            return best_children[0]
        elif len(best_children) > 1:
            return AndGoal(best_children)
        return goal

    return goal
