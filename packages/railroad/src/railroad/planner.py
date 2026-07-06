from typing import List, Dict, Union, SupportsFloat, SupportsInt
from collections.abc import Set
from railroad._bindings import get_usable_actions
from railroad._bindings import astar as _astar_cpp

__all__ = ["MCTSPlanner", "AStarPlanner", "get_usable_actions"]
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


def _normalize_goal(goal: Union[Goal, Fluent]) -> Goal:
    if isinstance(goal, Fluent):
        return LiteralGoal(goal)
    return goal


class _PlannerBase:
    """Shared negative-precondition preprocessing for all planner backends.

    Converts ~F("x") → F("not-x") transparently so the C++ layer only
    sees positive fluents.  Subclasses implement __call__ with their own
    search algorithm.
    """

    def __init__(self, actions: List[Action], use_det_heuristic: bool = False):
        self._use_det_heuristic = use_det_heuristic
        self._original_actions = actions
        self._base_negative_fluents: Set[Fluent] = extract_negative_preconditions(actions)
        self._base_mapping: Dict[Fluent, Fluent] = create_positive_fluent_mapping(
            self._base_negative_fluents
        )
        self._current_mapping = self._base_mapping
        self._converted_actions = self._convert_actions(actions, self._current_mapping)

    def _convert_actions(
        self, actions: List[Action], mapping: Dict[Fluent, Fluent]
    ) -> List[Action]:
        converted = []
        for action in actions:
            action_with_preconds = convert_action_to_positive_preconditions(action, mapping)
            action_with_effects = convert_action_effects(action_with_preconds, mapping)
            converted.append(action_with_effects)
        return converted

    def _ensure_mapping_includes_goal(self, goal: Goal) -> None:
        """Extend the negative→positive mapping if the goal introduces new negations."""
        goal_negative_fluents = extract_negative_goal_fluents(goal)
        new_fluents = goal_negative_fluents - set(self._current_mapping.keys())
        if new_fluents:
            extended_mapping = dict(self._current_mapping)
            for fluent in new_fluents:
                not_name = f"not-{fluent.name}"
                not_fluent = Fluent(not_name, *fluent.args)
                extended_mapping[fluent] = not_fluent
            self._current_mapping = extended_mapping
            self._converted_actions = self._convert_actions(
                self._original_actions, self._current_mapping
            )
            self._on_mapping_changed()

    def _on_mapping_changed(self) -> None:
        """Hook called after the mapping is extended.  Override in subclasses
        that cache derived objects (e.g. the C++ MCTS planner instance)."""
        pass

    def _prepare(
        self, state: State, goal: Union[Goal, Fluent]
    ) -> tuple[State, Goal]:
        """Normalize goal, extend mapping if needed, convert state and goal."""
        goal = _normalize_goal(goal)
        self._ensure_mapping_includes_goal(goal)
        converted_state = convert_state_to_positive_preconditions(state, self._current_mapping)
        converted_goal = convert_goal_to_positive_preconditions(goal, self._current_mapping)
        return converted_state, converted_goal


class MCTSPlanner(_PlannerBase):
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

    def __init__(self, actions: List[Action], use_det_heuristic: bool = False):
        super().__init__(actions, use_det_heuristic)
        self._cpp_planner = _MCTSPlannerCpp(self._converted_actions, self._use_det_heuristic)

    def _on_mapping_changed(self) -> None:
        self._cpp_planner = _MCTSPlannerCpp(self._converted_actions, self._use_det_heuristic)

    def __call__(
        self,
        state: State,
        goal: Union[Goal, Fluent],
        max_iterations: SupportsInt = 1000,
        max_depth: SupportsInt = 100,
        c: SupportsFloat = 1.414,
        heuristic_multiplier: SupportsFloat = 5.0,
    ) -> str:
        """Run MCTS planning to find the next action.

        Args:
            state: Current state (will be automatically converted)
            goal: Goal to achieve. Can be a Goal object or a single Fluent.
            max_iterations: Maximum number of MCTS iterations
            max_depth: Maximum depth for rollouts
            c: Exploration constant for UCB1
            heuristic_multiplier: Multiplier for heuristic in reward calculation

        Returns:
            Name of the selected action as a string
        """
        converted_state, converted_goal = self._prepare(state, goal)
        return self._cpp_planner(
            converted_state, converted_goal, max_iterations, max_depth, c,
            heuristic_multiplier
        )

    def get_trace_from_last_mcts_tree(self):
        """Get trace from the last MCTS tree (delegates to C++ planner)."""
        return self._cpp_planner.get_trace_from_last_mcts_tree()

    def heuristic(self, state: State, goal: Union[Goal, Fluent]) -> float:
        """Compute FF heuristic using converted state/goal/actions."""
        from railroad._bindings import ff_heuristic as _ff_heuristic_cpp
        from railroad._bindings import det_ff_heuristic as _det_ff_heuristic_cpp

        converted_state, converted_goal = self._prepare(state, goal)
        heuristic_fn = _det_ff_heuristic_cpp if self._use_det_heuristic else _ff_heuristic_cpp
        return heuristic_fn(converted_state, converted_goal, self._converted_actions)


class AStarPlanner(_PlannerBase):
    """A* Planner with the same interface as MCTSPlanner.

    Uses the C++ A* implementation with FF heuristic.  On each call, runs
    A* from the given state to find the optimal plan, then returns the name
    of the first action — matching the one-action-at-a-time interface that
    MCTSPlanner exposes.

    Usage:
        astar = AStarPlanner(all_actions)
        goal  = F("chopped herbs") & ~F("hand-full robot1")
        action_name = astar(state, goal)   # same call signature as MCTSPlanner
    """

    def plan_sequence(self, state: State, goal: Union[Goal, Fluent]) -> List[Action]:
        """Run A* once and return the full plan as a list of Actions."""
        converted_state, converted_goal = self._prepare(state, goal)
        return _astar_cpp(converted_state, self._converted_actions, converted_goal) or []

    def __call__(
        self,
        state: State,
        goal: Union[Goal, Fluent],
        **kwargs,  # absorbs MCTS-specific params (max_iterations, c, …) so it's a drop-in
    ) -> str:
        """Run A* to find the optimal plan and return the first action name.

        Args:
            state: Current state
            goal: Goal to achieve (Goal object or single Fluent)
            **kwargs: Ignored — accepted for drop-in compatibility with MCTSPlanner

        Returns:
            Name of the first action in the optimal plan, or 'NONE' if no
            plan exists or the goal is already satisfied.
        """
        plan = self.plan_sequence(state, goal)
        if not plan:
            return 'NONE'
        return plan[0].name


def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        prev, action = came_from[current]
        path.append(action)
        current = prev
    path.reverse()
    return path
