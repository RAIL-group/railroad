from .core import Operator, Fluent, Effect, OptCallable, Num
from typing import Callable, Set, Union
from mrppddl._bindings import GoalType, Goal, AndGoal

F = Fluent


# =============================================================================
# Goal formatting and display functions
# =============================================================================

def format_goal(goal, indent: int = 0, compact: bool = False) -> str:
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
            child_strs = [str(c.fluent()) for c in children]
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
            child_strs = [str(c.fluent()) for c in children]
            return f"{prefix}OR({', '.join(child_strs)})"
        else:
            lines = [f"{prefix}OR("]
            for child in children:
                lines.append(format_goal(child, indent + 1, compact))
            lines.append(f"{prefix})")
            return "\n".join(lines)

    else:
        return f"{prefix}<unknown goal type>"


def get_satisfied_branch(goal, fluents: Set[Fluent]) -> Union[Goal, None]:
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


def get_best_branch(goal, fluents: Set[Fluent]) -> Goal:
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


# =============================================================================
# Operator construction helpers
# =============================================================================


def _make_callable(opt_expr: OptCallable) -> Callable[..., float]:
    if isinstance(opt_expr, Num):
        return lambda *args: opt_expr
    else:
        return lambda *args: opt_expr(*args)


def _invert_prob(opt_expr: OptCallable) -> Callable[..., float]:
    if isinstance(opt_expr, Num):
        return lambda *args: 1 - opt_expr
    else:
        return lambda *args: 1 - opt_expr(*args)


def construct_move_operator(move_time: OptCallable):
    move_time = _make_callable(move_time)
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?r ?from")}),
            Effect(
                time=(move_time, ["?r", "?from", "?to"]),
                resulting_fluents={F("free ?r"), F("at ?r ?to")},
            ),
        ],
    )


def construct_move_visited_operator(move_time: OptCallable):
    move_time = _make_callable(move_time)
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={Fluent("not free ?r")}),
            Effect(
                time=(move_time, ["?r", "?from", "?to"]),
                resulting_fluents={
                    F("free ?r"),
                    F("not at ?r ?from"),
                    F("at ?r ?to"),
                    F("visited ?to"),
                },
            ),
        ],
    )


def construct_search_operator(
    object_find_prob: OptCallable, move_time: OptCallable, pick_time: OptCallable
) -> Operator:
    object_find_prob = _make_callable(object_find_prob)
    inv_object_find_prob = _invert_prob(object_find_prob)
    move_time = _make_callable(move_time)
    pick_time = _make_callable(pick_time)
    return Operator(
        name="search",
        parameters=[
            ("?robot", "robot"),
            ("?loc_from", "location"),
            ("?loc_to", "location"),
            ("?object", "object"),
        ],
        preconditions=[
            Fluent("free ?robot"),
            Fluent("at ?robot ?loc_from"),
            ~Fluent("searched ?loc_to ?object"),
            ~Fluent("found ?object"),
        ],
        effects=[
            Effect(
                time=0,
                resulting_fluents={
                    ~Fluent("free ?robot"),
                    ~Fluent("at ?robot ?loc_from"),
                    Fluent("searched ?loc_to ?object"),
                },
            ),
            Effect(
                time=(move_time, ["?robot", "?loc_from", "?loc_to"]),
                resulting_fluents={Fluent("at ?robot ?loc_to")},
                prob_effects=[
                    (
                        (object_find_prob, ["?robot", "?loc_to", "?object"]),
                        [
                            Effect(time=0, resulting_fluents={Fluent("found ?object")}),
                            Effect(
                                time=(pick_time, ["?robot", "?object"]),
                                resulting_fluents={
                                    Fluent("holding ?robot ?object"),
                                    Fluent("free ?robot"),
                                },
                            ),
                        ],
                    ),
                    (
                        (inv_object_find_prob, ["?robot", "?loc_to", "?object"]),
                        [Effect(time=0, resulting_fluents={Fluent("free ?robot")})],
                    ),
                ],
            ),
        ],
    )


def construct_wait_operator():
    return Operator(
        name="wait",
        parameters=[("?r1", "robot"), ("?r2", "robot")],
        preconditions=[F("free ?r1"), ~F("free ?r2"), ~F("waiting ?r2 ?r1")],
        effects=[Effect(time=0, resulting_fluents={F("not free ?r1"), F("waiting ?r1 ?r2")})],
    )
