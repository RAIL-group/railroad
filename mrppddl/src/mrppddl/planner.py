from typing import List, Dict, Union, Set
from mrppddl._bindings import astar  # noqa
from mrppddl._bindings import get_usable_actions as _get_usable_actions_cpp
from mrppddl._bindings import MCTSPlanner as _MCTSPlannerCpp
from mrppddl._bindings import Action, State, Fluent
from mrppddl._bindings import GoalType  # noqa: F401
# Import Goal as the base class (it's called "Goal" in bindings, maps to GoalBase in C++)
from mrppddl._bindings import Goal, LiteralGoal


def _normalize_goal(goal: Union[Goal, Fluent]) -> Goal:
    """Normalize goal input to a Goal object.

    If a raw Fluent is passed, automatically wraps it in a LiteralGoal.
    This provides a convenient API where users can pass either:
        - A Goal object: F("a") & F("b"), AndGoal([...]), etc.
        - A single Fluent: F("visited a")

    Args:
        goal: Either a Goal object or a single Fluent

    Returns:
        A Goal object (LiteralGoal if input was a Fluent)
    """
    if isinstance(goal, Fluent):
        return LiteralGoal(goal)
    return goal
from mrppddl.core import (
    extract_negative_preconditions,
    extract_negative_goal_fluents,
    create_positive_fluent_mapping,
    convert_action_to_positive_preconditions,
    convert_action_effects,
    convert_state_to_positive_preconditions,
    convert_goal_to_positive_preconditions,
)


def get_usable_actions(
    state: State,
    goal: Union[Goal, Fluent],
    all_actions: List[Action],
) -> List[Action]:
    """Get actions usable from the given state via forward reachability.

    Args:
        state: The current state
        goal: Goal for the planning problem. Can be:
            - A Goal object: F("a") & F("b"), AndGoal([...]), etc.
            - A single Fluent: F("visited a") (auto-wrapped to LiteralGoal)
        all_actions: List of all available actions

    Returns:
        List of actions that are reachable from the state
    """
    # Normalize goal (wrap Fluent in LiteralGoal if needed)
    goal = _normalize_goal(goal)
    # The goal parameter is kept for API compatibility but the C++ implementation
    # now computes reachability independent of the goal
    return _get_usable_actions_cpp(state, all_actions)


class MCTSPlanner:
    """MCTS Planner with automatic negative precondition preprocessing.

    This wrapper around the C++ MCTSPlanner automatically converts negative
    preconditions and goal fluents to positive equivalents for improved
    planning performance. The preprocessing is transparent to the user.

    When a goal contains negative fluents (e.g., ~F("at Book table")), the
    planner automatically extends the mapping to include these fluents and
    re-converts actions to properly handle them.

    Usage:
        mcts = MCTSPlanner(all_actions)
        goal = F("visited a") & F("visited b")  # Use Goal API
        action_name = mcts(state, goal, max_iterations=1000, c=1.414)
    """

    def __init__(self, actions: List[Action]):
        """Initialize MCTSPlanner with automatic preprocessing.

        Args:
            actions: List of Action objects to use for planning
        """
        # Store original actions for later re-conversion if needed
        self._original_actions = actions

        # Extract negative preconditions from actions (base mapping)
        self._base_negative_fluents: Set[Fluent] = extract_negative_preconditions(actions)

        # Create base mapping from action preconditions only
        self._base_mapping: Dict[Fluent, Fluent] = create_positive_fluent_mapping(
            self._base_negative_fluents
        )

        # Convert actions with base mapping and create initial C++ planner
        self._current_mapping = self._base_mapping
        self._converted_actions = self._convert_actions(actions, self._current_mapping)
        self._cpp_planner = _MCTSPlannerCpp(self._converted_actions)

    def _convert_actions(
        self, actions: List[Action], mapping: Dict[Fluent, Fluent]
    ) -> List[Action]:
        """Convert actions using the given mapping."""
        converted_actions = []
        for action in actions:
            # First convert preconditions
            action_with_preconds = convert_action_to_positive_preconditions(
                action, mapping
            )
            # Then convert effects
            action_with_effects = convert_action_effects(
                action_with_preconds, mapping
            )
            converted_actions.append(action_with_effects)
        return converted_actions

    def _ensure_mapping_includes_goal(self, goal: Goal) -> None:
        """Extend mapping if goal contains new negative fluents.

        If the goal has negative fluents not in the current mapping,
        extends the mapping and re-converts actions.
        """
        # Extract negative fluents from goal
        goal_negative_fluents = extract_negative_goal_fluents(goal)

        # Check if any are new (not in current mapping)
        new_fluents = goal_negative_fluents - set(self._current_mapping.keys())

        if new_fluents:
            # Extend mapping with new fluents
            extended_mapping = dict(self._current_mapping)
            for fluent in new_fluents:
                not_name = f"not-{fluent.name}"
                not_fluent = Fluent(not_name, *fluent.args)
                extended_mapping[fluent] = not_fluent

            # Update current mapping
            self._current_mapping = extended_mapping

            # Re-convert actions with extended mapping
            self._converted_actions = self._convert_actions(
                self._original_actions, self._current_mapping
            )

            # Create new C++ planner with re-converted actions
            self._cpp_planner = _MCTSPlannerCpp(self._converted_actions)

    def __call__(
        self,
        state: State,
        goal: Union[Goal, Fluent],
        max_iterations: int = 1000,
        max_depth: int = 100,
        c: float = 1.414,
        heuristic_multiplier: float = 5.0,
    ) -> str:
        """Run MCTS planning to find the next action.

        Args:
            state: Current state (will be automatically converted)
            goal: Goal to achieve. Can be:
                - A Goal object: F("a") & F("b"), AndGoal([...]), etc.
                - A single Fluent: F("visited a") (auto-wrapped to LiteralGoal)
            max_iterations: Maximum number of MCTS iterations
            max_depth: Maximum depth for rollouts
            c: Exploration constant for UCB1
            heuristic_multiplier: Multiplier for heuristic in reward calculation

        Returns:
            Name of the selected action as a string
        """
        # Normalize goal (wrap Fluent in LiteralGoal if needed)
        goal = _normalize_goal(goal)

        # Ensure mapping includes goal's negative fluents
        self._ensure_mapping_includes_goal(goal)

        # Convert state with (possibly extended) mapping
        converted_state = convert_state_to_positive_preconditions(
            state, self._current_mapping
        )

        # Convert goal with (possibly extended) mapping
        converted_goal = convert_goal_to_positive_preconditions(
            goal, self._current_mapping
        )

        return self._cpp_planner(
            converted_state, converted_goal, max_iterations, max_depth, c,
            heuristic_multiplier
        )

    def get_trace_from_last_mcts_tree(self):
        """Get trace from the last MCTS tree (delegates to C++ planner)."""
        return self._cpp_planner.get_trace_from_last_mcts_tree()


def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        prev, action = came_from[current]
        path.append(action)
        current = prev
    path.reverse()
    return path
