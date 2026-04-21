from typing import List, Dict, Union, SupportsFloat, SupportsInt
from collections.abc import Set
from railroad._bindings import get_usable_actions

__all__ = ["MCTSPlanner", "get_usable_actions"]
from railroad._bindings import MCTSPlanner as _MCTSPlannerCpp
from railroad._bindings import Action, State, Fluent
from railroad._bindings import Goal, LiteralGoal
from railroad.core import (
    extract_negative_preconditions,
    extract_negative_goal_fluents,
    create_positive_fluent_mapping,
    convert_action_to_positive_preconditions,
    convert_action_effects,
    convert_state_to_positive_preconditions,
    convert_goal_to_positive_preconditions,
)
from railroad._action_pruning import prune_probabilistic_achievers


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

    def __init__(
        self,
        actions: List[Action],
        enable_action_pruning: bool = True,
        pruning_top_k: int = 3,
    ):
        """Initialize MCTSPlanner with automatic preprocessing.

        Args:
            actions: List of Action objects to use for planning
            enable_action_pruning: If True, prune redundant probabilistic
                achievers before each planning call (keeps top-k by
                probability and top-k by time-to-reach per goal fluent).
            pruning_top_k: Number of achievers to keep per ranking criterion.
        """
        # Store original actions for later re-conversion if needed
        self._original_actions = actions
        self._enable_action_pruning = enable_action_pruning
        self._pruning_top_k = pruning_top_k

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
        # Tracks the C++ planner that actually ran MCTS on the last call
        # (may be a transient pruned-action-set planner). Used by
        # get_trace_from_last_mcts_tree so traces reflect the real search.
        self._last_cpp_planner = self._cpp_planner
        self._last_pruning_stats = (0, 0)

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
        max_iterations: SupportsInt = 1000,
        max_depth: SupportsInt = 100,
        c: SupportsFloat = 1.414,
        heuristic_multiplier: SupportsFloat = 5.0,
        debug_heuristic: bool = False,
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
            debug_heuristic: If True, prints detailed FF heuristic debug report

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

        cpp_planner = self._cpp_planner
        original_count = len(self._converted_actions)
        pruned_count = original_count
        if self._enable_action_pruning:
            pruned_actions = prune_probabilistic_achievers(
                converted_state,
                converted_goal,
                self._converted_actions,
                top_k=self._pruning_top_k,
            )
            pruned_count = len(pruned_actions)
            if pruned_count < original_count:
                cpp_planner = _MCTSPlannerCpp(pruned_actions)

        self._last_cpp_planner = cpp_planner
        self._last_pruning_stats = (original_count, pruned_count)

        if debug_heuristic:
            print(cpp_planner.debug_heuristic(converted_state, converted_goal))

        return cpp_planner(
            converted_state, converted_goal, max_iterations, max_depth, c,
            heuristic_multiplier
        )

    def get_trace_from_last_mcts_tree(self):
        """Get trace from the last MCTS tree (delegates to C++ planner)."""
        return self._last_cpp_planner.get_trace_from_last_mcts_tree()

    def get_last_pruning_stats(self) -> tuple[int, int]:
        """Return ``(original_action_count, action_count_after_pruning)`` from the last call."""
        return self._last_pruning_stats

    def heuristic(self, state: State, goal: Union[Goal, Fluent]) -> float:
        """Compute FF heuristic using converted state/goal/actions.

        This method computes the FF heuristic with proper conversion of
        negative preconditions to positive equivalents, matching the
        internal heuristic used by the MCTS planner.

        Args:
            state: Current state (will be automatically converted)
            goal: Goal to achieve. Can be:
                - A Goal object: F("a") & F("b"), AndGoal([...]), etc.
                - A single Fluent: F("visited a") (auto-wrapped to LiteralGoal)

        Returns:
            Heuristic value (estimated cost to reach goal)
        """
        from railroad._bindings import ff_heuristic as _ff_heuristic_cpp

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

        return _ff_heuristic_cpp(converted_state, converted_goal, self._converted_actions)

    def debug_heuristic(self, state: State, goal: Union[Goal, Fluent]) -> str:
        """Return a detailed FF heuristic debug report for the given state/goal.

        The report includes per-branch achievers considered during backward
        extraction and the probabilistic fluents contributing delta penalties.
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

        cpp_planner = self._cpp_planner
        if self._enable_action_pruning:
            pruned_actions = prune_probabilistic_achievers(
                converted_state,
                converted_goal,
                self._converted_actions,
                top_k=self._pruning_top_k,
            )
            if len(pruned_actions) < len(self._converted_actions):
                cpp_planner = _MCTSPlannerCpp(pruned_actions)

        return cpp_planner.debug_heuristic(converted_state, converted_goal)


def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        prev, action = came_from[current]
        path.append(action)
        current = prev
    path.reverse()
    return path
