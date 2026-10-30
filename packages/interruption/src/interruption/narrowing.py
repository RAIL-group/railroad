"""
Narrowing a search problem before it's searched: to the objects and locations
its relaxed plan uses (relaxed_plan_problem), or to its goal's objects and the
ones a robot holds (goal_objects, held_objects, filter_actions_to_objects).
"""
from dataclasses import replace
from typing import TYPE_CHECKING

from railroad._bindings import relaxed_plan_action_indices as _relaxed_plan_action_indices
from railroad.core import Action, Goal, State

from .constants import LAMBDA_ADD, LAMBDA_FF, LAMBDA_MAX

if TYPE_CHECKING:  # planner imports this module, so only for type checking
    from .planner import InterruptionSearchProblem


def relaxed_plan_problem(
    state: State, problem: "InterruptionSearchProblem", also_keep: frozenset[str] = frozenset()
) -> "InterruptionSearchProblem | None":
    """
    The problem narrowed to what its relaxed plan uses: FF's relaxed plan for the
    goal from state (deletes and negative preconditions ignored, as in the
    heuristic), together with every action directly achieving a goal fluent (so
    every table, say, that would satisfy "on a table" - the relaxed plan picks
    just one), names objects, locations and the like as its actions' arguments;
    only actions whose arguments are all among those, or among what a robot is
    holding, are kept. Held objects are added because the relaxation ignores a
    full hand, so its plan never puts down what the robot holds - which may be
    needed before anything else can be picked up. also_keep adds further names
    (objects, locations, categories) to keep. None when there's nothing to
    narrow: the goal holds, is unreachable, or every action survives.
    """
    actions = problem.actions
    plan = _relaxed_plan_action_indices(
        state, problem.goal, actions, LAMBDA_ADD, LAMBDA_MAX, LAMBDA_FF
    )
    if not plan:
        return None
    keep = {arg for i in plan for arg in actions[i].name.split(" ")[1:]}
    keep |= {
        arg for fluent in state.fluents if fluent.name.startswith("holding") for arg in fluent.args
    }
    keep |= also_keep
    kept = [action for action in actions if keep.issuperset(action.name.split(" ")[1:])]
    if len(kept) == len(actions):
        return None
    return replace(problem, actions=kept)


def goal_objects(goal: Goal, all_objects: set[str]) -> set[str]:
    """The objects among all_objects that any literal of goal names."""
    return {
        arg for literal in goal.get_all_literals() for arg in literal.args if arg in all_objects
    }


def held_objects(state: State, all_objects: set[str]) -> set[str]:
    """
    The objects among all_objects that a robot is holding in state
    (holding-in-left / holding-in-right / holding fluents). A filtered search must
    keep them: with its hands full, a robot that can't put them down can't pick up
    anything else.
    """
    return {
        fluent.args[-1] for fluent in state.fluents
        if fluent.name.startswith("holding") and fluent.args and fluent.args[-1] in all_objects
    }


def filter_actions_to_objects(
    actions: list[Action], keep: set[str], all_objects: set[str]
) -> list[Action]:
    """
    The actions that involve no object of all_objects outside keep; actions
    involving no object at all (moves) are always kept. An action's arguments are
    the words of its name after the operator name.
    """
    excluded = all_objects - keep
    return [action for action in actions if excluded.isdisjoint(action.name.split(" ")[1:])]
