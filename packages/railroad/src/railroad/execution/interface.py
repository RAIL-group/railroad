"""Execution interface for running PDDL plans.

This module provides a simplified EnvironmentInterface and OngoingAction
for executing PDDL plans. It prioritizes simplicity for testing and examples
while supporting all MCTS planner capabilities including probabilistic effects
with nested timing.
"""

import itertools
from copy import copy
from typing import Collection, Dict, List, Set, Tuple

from railroad.core import Operator, transition
from railroad._bindings import Action, Fluent, State, GroundedEffect

from .environment import Environment

F = Fluent


class OngoingAction:
    """Tracks action execution and schedules effects.

    A unified class that handles all action types (move, search, pick, place, etc.)
    without subclasses. The key difference from railroad.environment.OngoingAction
    is handling of nested effects inside prob_effects - when a probabilistic
    effect's time arrives, it resolves the outcome and schedules the nested
    effects from the chosen branch.

    This fixes Issue #6 where nested effects with timing inside prob_effects
    were not being scheduled.
    """

    def __init__(
        self,
        time: float,
        action: Action,
        environment: Environment,
    ) -> None:
        """Initialize an ongoing action.

        Args:
            time: Start time of the action.
            action: The action being executed.
            environment: Environment for resolving probabilistic effects.
        """
        self.time = float(time)
        self.name = action.name
        self._start_time = float(time)
        self._action = action
        self._environment = environment

        # Schedule only top-level effects initially
        self._upcoming_effects: List[Tuple[float, GroundedEffect]] = sorted(
            [(time + eff.time, eff) for eff in action.effects],
            key=lambda el: el[0],
        )

        # Track applied fluents
        self._applied_fluents: Set[Fluent] = set()

    @property
    def time_to_next_event(self) -> float:
        """Time until the next scheduled effect."""
        if self._upcoming_effects:
            return self._upcoming_effects[0][0]
        return float("inf")

    @property
    def is_done(self) -> bool:
        """Whether all effects have been applied."""
        return not self._upcoming_effects

    @property
    def upcoming_effects(self) -> List[Tuple[float, GroundedEffect]]:
        """Return remaining upcoming effects."""
        return self._upcoming_effects

    def advance(
        self,
        time: float,
        current_fluents: Set[Fluent],
    ) -> Tuple[List[Tuple[float, GroundedEffect]], Set[Fluent]]:
        """Advance the action to a new time, resolving probabilities and scheduling nested effects.

        When a probabilistic effect's time arrives:
        1. Call environment.resolve_probabilistic_effect()
        2. Get the chosen nested effects
        3. Schedule them with times relative to now

        Args:
            time: The new time to advance to.
            current_fluents: Current state fluents for resolution.

        Returns:
            Tuple of (effects_to_apply, immediate_fluents):
            - effects_to_apply: List of (time, effect) tuples for effects due
            - immediate_fluents: Set of fluents to apply immediately
        """
        self.time = time
        effects_to_apply: List[Tuple[float, GroundedEffect]] = []
        immediate_fluents: Set[Fluent] = set()

        # Process effects that are due (with small tolerance for floating point)
        while self._upcoming_effects and self._upcoming_effects[0][0] <= time + 1e-9:
            scheduled_time, effect = self._upcoming_effects.pop(0)

            if effect.is_probabilistic:
                # Resolve which branch happens
                nested_effects, branch_fluents = self._environment.resolve_probabilistic_effect(
                    effect, current_fluents
                )
                immediate_fluents.update(branch_fluents)

                # Schedule nested effects relative to current time
                for nested in nested_effects:
                    nested_time = time + nested.time
                    self._upcoming_effects.append((nested_time, nested))

                # Re-sort after adding nested effects
                self._upcoming_effects.sort(key=lambda el: el[0])

                # Also add the immediate fluents from the effect itself
                immediate_fluents.update(effect.resulting_fluents)
            else:
                effects_to_apply.append((scheduled_time, effect))

        return effects_to_apply, immediate_fluents


class EnvironmentInterface:
    """Executes PDDL actions against an Environment.

    Simpler than railroad.environment.EnvironmentInterface:
    - No OngoingAction subclass dispatch (single unified OngoingAction)
    - No skill execution tracking (IDLE/RUNNING/DONE)
    - No move interruption support
    - Revelation happens via fluent inspection, not explicit _reveal() calls

    This interface is designed for testing and examples where simplicity
    is more important than features like move interruption or real-time
    skill execution tracking.
    """

    def __init__(
        self,
        initial_state: State,
        objects_by_type: Dict[str, Collection[str]],
        operators: List[Operator],
        environment: Environment,
    ) -> None:
        """Initialize the environment interface.

        Args:
            initial_state: Initial planning state.
            objects_by_type: Dictionary mapping type names to object collections.
            operators: List of operators for action instantiation.
            environment: The Environment for probabilistic resolution.
        """
        self._state = initial_state
        self.objects_by_type: Dict[str, Set[str]] = {
            k: set(v) for k, v in objects_by_type.items()
        }
        self.operators = operators
        self.environment = environment

    @property
    def time(self) -> float:
        """Current time in the state."""
        return self._state.time

    @property
    def state(self) -> State:
        """Current state."""
        return self._state

    def get_actions(self) -> List[Action]:
        """Instantiate operators under the current objects_by_type.

        Returns:
            List of grounded actions available in the current state.
        """
        # Add robot locations to location set (for intermediate location support)
        objects_with_rloc: Dict[str, Collection[str]] = {
            k: set(v) for k, v in self.objects_by_type.items()
        }
        robot_locs = set(
            f"{rob}_loc"
            for rob in self.objects_by_type.get("robot", set())
            if F(f"at {rob} {rob}_loc") in self._state.fluents
        )
        if "location" in objects_with_rloc:
            objects_with_rloc["location"] = set(objects_with_rloc["location"]) | robot_locs

        all_actions: List[Action] = list(
            itertools.chain.from_iterable(
                operator.instantiate(objects_with_rloc) for operator in self.operators
            )
        )
        return all_actions

    def advance(self, action: Action) -> State:
        """Execute action and return new state.

        Flow:
        1. Create OngoingAction
        2. Advance time to next effect
        3. If effect is probabilistic, resolve via environment
        4. Apply effects and update state
        5. Schedule any nested effects from chosen branch
        6. Repeat until all effects applied
        7. Update objects_by_type from revealed fluents

        Args:
            action: The action to execute.

        Returns:
            The new state after action execution.

        Raises:
            ValueError: If action preconditions are not satisfied.
        """
        # Check preconditions
        if not self.state.satisfies_precondition(action):
            raise ValueError(
                f"Action preconditions not satisfied: {action.name} in state {self.state}"
            )

        # Create ongoing action
        ongoing = OngoingAction(self._state.time, action, self.environment)

        # Process all effects
        while not ongoing.is_done:
            # Get time of next effect
            next_time = ongoing.time_to_next_event

            # Advance to that time and get effects to apply
            effects_to_apply, immediate_fluents = ongoing.advance(
                next_time, self._state.fluents
            )

            # Build new state with effects
            new_upcoming = list(self._state.upcoming_effects)
            for eff_time, eff in effects_to_apply:
                new_upcoming.append((eff_time, eff))

            new_upcoming.sort(key=lambda el: el[0])

            new_state = State(
                next_time,
                self._state.fluents | immediate_fluents,
                new_upcoming,
            )

            # Apply effects via transition
            states_and_probs = transition(new_state, None)
            self._state = self._get_new_state_by_intersection(states_and_probs)

            # Handle revelation and object updates
            new_fluents, self.objects_by_type = self._reveal()
            self._state.update_fluents(new_fluents)

            # Handle pick/place side effects
            self._handle_pick_place_side_effects(action)

        return self.state

    def _get_new_state_by_intersection(
        self,
        states_and_probs: List[Tuple[State, float]],
    ) -> State:
        """Determine new state by intersecting outcomes.

        When transition returns multiple possible states (from probabilistic
        effects), we take the intersection of fluents that are common to all.

        Args:
            states_and_probs: List of (state, probability) tuples from transition.

        Returns:
            State with fluents common to all outcomes.
        """
        states = [s for s, _ in states_and_probs]
        base = states[0]
        new_fluents: Set[Fluent] = {
            fl for fl in base.fluents if all(fl in s.fluents for s in states[1:])
        }
        new_upcoming = [
            ue for ue in base.upcoming_effects
            if all(ue in s.upcoming_effects for s in states[1:])
        ]
        return State(states[0].time, new_fluents, new_upcoming)

    def _reveal(self) -> Tuple[Set[Fluent], Dict[str, Set[str]]]:
        """Handle revelation of objects at searched locations.

        When a location is searched (has 'searched' fluent), reveal objects
        that are there according to ground truth.

        Returns:
            Tuple of (new_fluents, updated_objects_by_type).
        """
        new_fluents: Set[Fluent] = set(self._state.fluents)

        def _is_location_searched(location: str, fluents: Set[Fluent]) -> bool:
            return any(
                f.name == "searched" and f.args[0] == location
                for f in fluents
            )

        # Find newly revealed locations
        newly_revealed_locations = [
            loc
            for loc in self.objects_by_type.get("location", set())
            if _is_location_searched(loc, new_fluents)
            and F(f"revealed {loc}") not in new_fluents
        ]

        updated_objects_by_type = copy(self.objects_by_type)
        for loc in newly_revealed_locations:
            new_fluents.add(F(f"revealed {loc}"))

            # Add objects at this location to type universe
            objects_at_loc = self.environment.get_objects_at_location(loc)
            for obj in objects_at_loc:
                updated_objects_by_type.setdefault("object", set()).add(obj)
                new_fluents.add(F(f"found {obj}"))
                new_fluents.add(F(f"at {obj} {loc}"))

        return new_fluents, updated_objects_by_type

    def _handle_pick_place_side_effects(self, action: Action) -> None:
        """Handle side effects of pick and place actions.

        Updates environment's object locations when objects are picked or placed.

        Args:
            action: The action that was executed.
        """
        parts = action.name.split()
        action_type = parts[0]

        if action_type == "pick" and len(parts) >= 4:
            _, _, loc, obj = parts[:4]
            self.environment.remove_object_from_location(obj, loc)
        elif action_type == "place" and len(parts) >= 4:
            _, _, loc, obj = parts[:4]
            self.environment.add_object_at_location(obj, loc)

    def is_goal_reached(self, goal_fluents: Collection[Fluent]) -> bool:
        """Check if all goal fluents are satisfied.

        Args:
            goal_fluents: Collection of fluents that must be true.

        Returns:
            True if all goal fluents are in the current state.
        """
        return all(fluent in self.state.fluents for fluent in goal_fluents)
