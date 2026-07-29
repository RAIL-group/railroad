from typing import List, Dict, Union, SupportsFloat, SupportsInt
from collections.abc import Set
from railroad._bindings import get_usable_actions, seed_planner_rng
from railroad._action_pruning import prune_probabilistic_achievers

__all__ = [
    "MCTSPlanner",
    "get_usable_actions",
    "prune_probabilistic_achievers",
    "seed_planner_rng",
]
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
    project_action,
    project_state,
    relevant_predicates,
)


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
        lambda_add: SupportsFloat = 0.5,
        lambda_max: SupportsFloat = 0.0,
        lambda_ff: SupportsFloat = 0.5,
        prune_top_n: int | None = None,
        prune_cheapest_m: int | None = None,
        prune_orphaned_supports: bool = True,
        frontier_objects: set[str] | None = None,
        project_irrelevant: bool = True,
        dead_end_penalty: SupportsFloat | None = None,
    ):
        """Initialize MCTSPlanner with automatic preprocessing.

        Args:
            actions: List of Action objects to use for planning
            lambda_add: weight on the additive (h_add) heuristic component
            lambda_max: weight on the max (h_max) heuristic component
            lambda_ff: weight on the relaxed-plan-cost (h_ff) heuristic component
            prune_top_n: per (fluent, robot), number of achievers to keep by
                success probability. Pruning is **off by default** (``None``);
                set this and/or ``prune_cheapest_m`` to a number to enable it.
                When pruning is on but this is ``None`` it counts as 0 (keep
                none by probability). See ``railroad._action_pruning``.
            prune_cheapest_m: per (fluent, robot), number of achievers to keep
                by time-to-execute. ``None`` (the default) leaves pruning off
                unless ``prune_top_n`` is set; ``None`` while pruning is on
                counts as 0 (keep none by cost).
            prune_orphaned_supports: also drop support actions left unable to
                reach the goal after achiever pruning (relaxed backward closure)
            frontier_objects: names of frontier objects (special, exploration-
                only objects). When given, any frontier whose achievers were all
                pruned and that has nothing located at it is removed entirely --
                including the moves that route to/through it -- which the generic
                closure cannot do in densely connected location graphs.
            project_irrelevant: drop fluents no precondition, branch condition,
                or goal reads from searched states and from action effects
                (``railroad.core.relevant_predicates``). Bisimulation-preserving
                and typically a large win -- every state hash walks the fluent
                set -- so it is on by default; pass False to search over the
                full fluent set when debugging.
            dead_end_penalty: cost charged for a branch the relaxation proves
                cannot reach the goal (h = inf). It is a **flat** cost: the
                time and extra_cost the branch already spent are deliberately
                not added, so failing slowly is not ranked below failing fast
                -- once a branch is dead, how it got there carries no
                information. ``None`` (the default) keeps the legacy behavior
                of clamping h to ``HEURISTIC_CANNOT_FIND_GOAL_PENALTY`` (0),
                which makes dead ends score *better* than reachable states and
                actively draws the search into them. A value that dominates
                typical plan costs (e.g. 1e4) makes MCTS avoid them; it also
                perturbs multi-robot search-ordering ties, which is why it is
                opt-in rather than the default.

        Defaults are an even split between h_add and h_ff (0.5, 0.0, 0.5).
        Weights are free-form (not normalized); the heuristic used during MCTS
        search is `lambda_add * h_add + lambda_max * h_max + lambda_ff * h_ff`
        plus the probabilistic-retry delta.
        """
        # Store original actions for later re-conversion if needed
        self._original_actions = actions

        # Heuristic mixing weights
        self._lambda_add = float(lambda_add)
        self._lambda_max = float(lambda_max)
        self._lambda_ff = float(lambda_ff)
        self._dead_end_penalty = (
            None if dead_end_penalty is None else float(dead_end_penalty)
        )

        # Action-pruning configuration (applied per-call in __call__). Pruning
        # is enabled only when a keep-count is given; both None => off, so
        # planners elsewhere are unaffected unless they opt in.
        self._prune_top_n = prune_top_n
        self._prune_cheapest_m = prune_cheapest_m
        self._prune_achievers = (
            prune_top_n is not None or prune_cheapest_m is not None
        )
        self._prune_orphaned_supports = prune_orphaned_supports
        self._frontier_objects = frontier_objects

        # Extract negative preconditions from actions (base mapping)
        self._base_negative_fluents: Set[Fluent] = extract_negative_preconditions(actions)

        # Create base mapping from action preconditions only
        self._base_mapping: Dict[Fluent, Fluent] = create_positive_fluent_mapping(
            self._base_negative_fluents
        )

        # Convert actions with base mapping and create initial C++ planner
        self._current_mapping = self._base_mapping
        self._converted_actions = self._convert_actions(actions, self._current_mapping)

        # Relevance projection state (see _project_for). Computed lazily on the
        # first call, once the goal and the state's queued effects are known --
        # both are readers, and neither is available here.
        self._project_irrelevant = project_irrelevant
        self._relevant: Set[str] | None = None
        self._search_actions = self._converted_actions

        self._cpp_planner = _MCTSPlannerCpp(
            self._search_actions,
            lambda_add=self._lambda_add,
            lambda_max=self._lambda_max,
            lambda_ff=self._lambda_ff,
            dead_end_penalty=self._dead_end_penalty,
        )

        # Action counts from the most recent search, for introspection/display:
        # how many actions MCTS actually considered vs. the unpruned total.
        self.num_actions_total: int = len(self._converted_actions)
        self.num_actions_considered: int = len(self._converted_actions)

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

            # The projection was derived from the old action set; drop it.
            self._relevant = None
            self._search_actions = self._converted_actions

            # Create new C++ planner with re-converted actions
            self._cpp_planner = _MCTSPlannerCpp(
                self._search_actions,
                lambda_add=self._lambda_add,
                lambda_max=self._lambda_max,
                lambda_ff=self._lambda_ff,
                dead_end_penalty=self._dead_end_penalty,
            )

    def _project_for(self, goal: Goal, state: State) -> State:
        """Project `state`, rebuilding the projected action set if needed.

        The relevance set depends on the goal and on the branch conditions of
        the state's queued effects, so it is computed per call and the action
        projection is rebuilt whenever a new reader shows up. In a normal run
        that happens once: every effect the search can queue comes from an
        action already scanned.
        """
        if not self._project_irrelevant:
            return state

        needed = relevant_predicates(
            self._converted_actions, goal, state.upcoming_effects
        )
        if self._relevant is None or not needed <= self._relevant:
            self._relevant = needed
            self._search_actions = [
                project_action(a, needed) for a in self._converted_actions
            ]
            self._cpp_planner = _MCTSPlannerCpp(
                self._search_actions,
                lambda_add=self._lambda_add,
                lambda_max=self._lambda_max,
                lambda_ff=self._lambda_ff,
                dead_end_penalty=self._dead_end_penalty,
            )
        return project_state(state, self._relevant)

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

        # Drop fluents nothing reads from the searched states and from the
        # actions that would re-add them (may rebuild self._cpp_planner).
        converted_state = self._project_for(converted_goal, converted_state)

        # Optionally prune redundant probabilistic achievers (and the support
        # actions they orphan) before searching. The pruned set depends on the
        # state/goal, so a planner is built per call; assign it to
        # self._cpp_planner so get_trace_from_last_mcts_tree() and other stats
        # readers see the planner that actually ran.
        self.num_actions_total = len(self._converted_actions)
        self.num_actions_considered = self.num_actions_total
        if self._prune_achievers:
            pruned_actions = prune_probabilistic_achievers(
                converted_state,
                converted_goal,
                self._search_actions,
                top_n=self._prune_top_n if self._prune_top_n is not None else 0,
                cheapest_m=(
                    self._prune_cheapest_m
                    if self._prune_cheapest_m is not None
                    else 0
                ),
                prune_orphaned_supports=self._prune_orphaned_supports,
                frontier_objects=self._frontier_objects,
            )
            self.num_actions_considered = len(pruned_actions)
            self._cpp_planner = _MCTSPlannerCpp(
                pruned_actions,
                lambda_add=self._lambda_add,
                lambda_max=self._lambda_max,
                lambda_ff=self._lambda_ff,
                dead_end_penalty=self._dead_end_penalty,
            )

        return self._cpp_planner(
            converted_state, converted_goal, max_iterations, max_depth, c,
            heuristic_multiplier
        )

    def get_trace_from_last_mcts_tree(self):
        """Get trace from the last MCTS tree (delegates to C++ planner)."""
        return self._cpp_planner.get_trace_from_last_mcts_tree()

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

        # Same projection MCTS searches under, so the two agree.
        converted_state = self._project_for(converted_goal, converted_state)

        # No dead_end_penalty here: that shapes the MCTS *reward*, while this
        # reports the raw heuristic, inf included.
        return _ff_heuristic_cpp(
            converted_state, converted_goal, self._search_actions,
            lambda_add=self._lambda_add,
            lambda_max=self._lambda_max,
            lambda_ff=self._lambda_ff,
        )


def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        prev, action = came_from[current]
        path.append(action)
        current = prev
    path.reverse()
    return path
