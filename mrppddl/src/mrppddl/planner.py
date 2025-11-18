from typing import List, Set, Dict
from mrppddl._bindings import astar, get_usable_actions  # noqa
from mrppddl._bindings import MCTSPlanner as _MCTSPlannerCpp
from mrppddl._bindings import Action, State, Fluent
from mrppddl.core import (
    extract_negative_preconditions,
    create_positive_fluent_mapping,
    convert_action_to_positive_preconditions,
    convert_action_effects,
    convert_state_to_positive_preconditions,
)


class MCTSPlanner(_MCTSPlannerCpp):
    """MCTS Planner with automatic negative precondition preprocessing.

    This wrapper around the C++ MCTSPlanner automatically converts negative
    preconditions to positive equivalents for improved planning performance.
    The preprocessing is transparent to the user - just use it like the
    original MCTSPlanner.

    Usage:
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(state, goal_fluents, max_iterations=1000, c=1.414)
    """

    def __init__(self, actions: List[Action]):
        """Initialize MCTSPlanner with automatic preprocessing.

        Args:
            actions: List of Action objects to use for planning
        """
        # Step 1: Extract all negative preconditions from actions
        negative_fluents = extract_negative_preconditions(actions)

        # Step 2: Create mapping to positive "not-" versions
        self._neg_to_pos_mapping: Dict[Fluent, Fluent] = create_positive_fluent_mapping(
            negative_fluents
        )

        # Step 3: Convert all actions (preconditions and effects)
        converted_actions = []
        for action in actions:
            # First convert preconditions
            action_with_preconds = convert_action_to_positive_preconditions(
                action, self._neg_to_pos_mapping
            )
            # Then convert effects
            action_with_effects = convert_action_effects(
                action_with_preconds, self._neg_to_pos_mapping
            )
            converted_actions.append(action_with_effects)

        # Initialize parent C++ class with converted actions
        super().__init__(converted_actions)

    def __call__(
        self,
        state: State,
        goal_fluents: Set[Fluent],
        max_iterations: int = 1000,
        max_depth: int = 100,
        c: float = 1.414,
    ) -> str:
        """Run MCTS planning to find the next action.

        Args:
            state: Current state (will be automatically converted)
            goal_fluents: Goal fluents to achieve
            max_iterations: Maximum number of MCTS iterations
            max_depth: Maximum depth for rollouts
            c: Exploration constant for UCB1

        Returns:
            Name of the selected action as a string
        """
        # Convert state to use positive preconditions
        converted_state = convert_state_to_positive_preconditions(
            state, self._neg_to_pos_mapping
        )

        # Call parent's __call__ with converted state
        return super().__call__(
            converted_state, goal_fluents, max_iterations, max_depth, c
        )


def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        prev, action = came_from[current]
        path.append(action)
        current = prev
    path.reverse()
    return path
