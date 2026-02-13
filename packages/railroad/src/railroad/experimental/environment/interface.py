"""Environment interface for executing PDDL plans.

This module provides the EnvironmentInterface class that bridges PDDL planning
with environment execution, and OngoingAction classes that track action execution.
"""

import itertools
import math
from copy import copy
from typing import (
    Callable,
    Collection,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from railroad.core import Operator, transition
from railroad._bindings import Action, Fluent, State, GroundedEffect

from .base import AbstractEnvironment, SkillStatus

F = Fluent


class EnvironmentInterface:
    """Interface between PDDL planning and environment execution.

    Manages state transitions, action execution, and coordination
    with the underlying environment.
    """

    def __init__(
        self,
        initial_state: State,
        objects_by_type: Dict[str, Collection[str]],
        operators: List[Operator],
        environment: AbstractEnvironment,
    ) -> None:
        """Initialize the environment interface.

        Args:
            initial_state: Initial planning state.
            objects_by_type: Dictionary mapping type names to object collections.
            operators: List of operators for action instantiation.
            environment: The underlying environment implementation.
        """
        self._state = initial_state
        self.objects_by_type: Dict[str, Set[str]] = {k: set(v) for k, v in objects_by_type.items()}
        self.operators = operators
        self.ongoing_actions: List[OngoingAction] = []
        self.environment = environment

    @property
    def time(self) -> float:
        """Current time in the state."""
        return self._state.time

    @property
    def state(self) -> State:
        """The state with upcoming effects from ongoing actions."""
        effects: List[Tuple[float, GroundedEffect]] = []
        for act in self.ongoing_actions:
            effects += act.upcoming_effects
        self.ongoing_actions = [act for act in self.ongoing_actions if not act.is_done]
        return State(
            self._state.time,
            self._state.fluents,
            sorted(self._state.upcoming_effects + effects, key=lambda el: el[0]),
        )

    def get_actions(self) -> List[Action]:
        """Instantiate operators under the current objects_by_type.

        Returns:
            List of grounded actions available in the current state.
        """
        objects_with_rloc: Dict[str, Collection[str]] = {k: set(v) for k, v in self.objects_by_type.items()}
        # Add robot locations to location set
        robot_locs = set(
            f"{rob}_loc"
            for rob in self.objects_by_type["robot"]
            if F(f"at {rob} {rob}_loc") in self._state.fluents
        )
        objects_with_rloc["location"] = set(objects_with_rloc["location"]) | robot_locs
        all_actions: List[Action] = list(
            itertools.chain.from_iterable(operator.instantiate(objects_with_rloc) for operator in self.operators)
        )

        def filter_intermediate_locations_as_destination(action: Action) -> bool:
            ans = action.name.split()
            if ans[0] == "move" and "_loc" in ans[3]:
                # (move robot1 robot1_loc other_loc)
                return False
            if ans[0] == "place" and "_loc" in ans[2]:
                # (place robot1 robot1_loc Pillow)
                return False
            if ans[0] == "search" and "_loc" in ans[2]:
                # (search robot1 robot1_loc Pillow)
                return False
            return True

        def filter_infinite_effect_time(action: Action) -> bool:
            # Infinite time appears when the skill time function returns infinity
            # This indicates that the skill is not available for the robot
            for eff in action.effects:
                if math.isinf(eff.time):
                    return False
            return True

        # Filter out actions that should not be allowed.
        filtered_actions = [a for a in all_actions if filter_infinite_effect_time(a)]
        filtered_actions = [a for a in filtered_actions if filter_intermediate_locations_as_destination(a)]
        return filtered_actions

    def _any_free_robots(self) -> bool:
        return any(f.name == "free" for f in self._state.fluents)

    def advance(
        self,
        action: Action,
        do_interrupt: bool = True,
        loop_callback_fn: Optional[Callable[[], None]] = None,
    ) -> State:
        """Advance the state by executing an action.

        Args:
            action: The action to execute.
            do_interrupt: Whether to interrupt ongoing actions when a robot becomes free.
            loop_callback_fn: Optional callback called each iteration.

        Returns:
            The new state after action execution.

        Raises:
            ValueError: If action preconditions are not satisfied.
        """
        # Check preconditions for safety (helps in debugging)
        if not self.state.satisfies_precondition(action):
            raise ValueError(f"Action preconditions not satisfied: {action.name} in state {self.state}")

        new_act = self._get_ongoing_action(action)
        self.ongoing_actions.append(new_act)

        robot_free = False
        while not robot_free and self.ongoing_actions:
            # if every action is done, break
            if all(act.is_done for act in self.ongoing_actions):
                break

            adv_time, completed_actions = self._get_advance_time_and_completed_actions()

            # stop robots for completed actions
            for act in completed_actions:
                self.environment.stop_robot(act.robot)

            new_effects = list(
                itertools.chain.from_iterable([act.advance(adv_time) for act in self.ongoing_actions])
            )
            new_state = State(
                adv_time,
                self._state.fluents,
                sorted(self._state.upcoming_effects + new_effects, key=lambda el: el[0]),
            )
            # Add new effects to state
            self._state = self._get_new_state_by_intersection(transition(new_state, None))

            # Reveal and get new fluents and objects_by_type
            new_fluents, self.objects_by_type = self._reveal()
            self._state.update_fluents(new_fluents)

            # Remove any actions that are now done
            self.ongoing_actions = [act for act in self.ongoing_actions if not act.is_done]

            # Update environment time (used ONLY in simulators to measure robot's action progress)
            self.environment.time = self._state.time

            robot_free = self._any_free_robots()

            if loop_callback_fn is not None:
                loop_callback_fn()

        # interrupt ongoing actions if needed
        if do_interrupt:
            for act in self.ongoing_actions:
                new_fluents = act.interrupt()
                self._state.update_fluents(new_fluents)

        self.ongoing_actions = [act for act in self.ongoing_actions if act.upcoming_effects]
        return self.state

    def _get_new_state_by_intersection(
        self, states_and_probs: List[Tuple[State, float]]
    ) -> State:
        """Determine new state by intersecting outcomes."""
        states = [s for s, _ in states_and_probs]
        base = states[0]
        new_fluents: Set[Fluent] = {fl for fl in base.fluents if all(fl in s.fluents for s in states[1:])}
        new_upcoming = [ue for ue in base.upcoming_effects if all(ue in s.upcoming_effects for s in states[1:])]
        new_state = State(states[0].time, new_fluents, new_upcoming)
        return new_state

    def _get_ongoing_action(self, action: Action) -> "OngoingAction":
        action_name = action.name.split()[0]
        if action_name not in {"move", "pick", "place", "search", "no_op"}:
            raise ValueError(f"Action {action.name} not supported in simulator.")
        if action_name == "move":
            return OngoingMoveAction(self._state.time, action, self.environment)
        elif action_name == "search":
            return OngoingSearchAction(self._state.time, action, self.environment)
        elif action_name == "pick":
            return OngoingPickAction(self._state.time, action, self.environment)
        elif action_name == "place":
            return OngoingPlaceAction(self._state.time, action, self.environment)
        elif action_name == "no_op":
            return OngoingNoOpAction(self._state.time, action, self.environment)
        # This should never be reached due to the check above
        raise ValueError(f"Action {action.name} not supported in simulator.")

    def _get_advance_time_and_completed_actions(self) -> Tuple[float, List["OngoingAction"]]:
        time_to_next_event = self.time
        completed_actions: List[OngoingAction] = []

        for act in self.ongoing_actions:
            if act.is_action_complete:
                completed_actions.append(act)
                act.do_add_all_remaining_effects = True
                time_to_next_event = act.time_to_next_event

        return time_to_next_event, completed_actions

    def is_goal_reached(self, goal_fluents: Collection[Fluent]) -> bool:
        """Check if all goal fluents are satisfied.

        Args:
            goal_fluents: Collection of fluents that must be true.

        Returns:
            True if all goal fluents are in the current state.
        """
        return all(fluent in self.state.fluents for fluent in goal_fluents)

    def _reveal(self) -> Tuple[Set[Fluent], Dict[str, Set[str]]]:
        new_fluents: Set[Fluent] = {fl for fl in self._state.fluents}

        def _is_location_searched(location: str, fluents: Set[Fluent]) -> bool:
            # Fluent 'searched' looks like: F("searched <loc> <obj>")
            return any(f.name == "searched" and f.args[0] == location for f in fluents)

        # Locations that have been searched but not yet marked revealed
        newly_revealed_locations = [
            loc
            for loc in self.objects_by_type.get("location", set())
            if _is_location_searched(loc, new_fluents) and F(f"revealed {loc}") not in new_fluents
        ]

        updated_objects_by_type = copy(self.objects_by_type)
        for loc in newly_revealed_locations:
            new_fluents.add(F(f"revealed {loc}"))

            # Add every object present at this location to type universe and mark as found/located
            for obj_type, objs in self.environment.get_objects_at_location(loc).items():
                updated_objects_by_type.setdefault(obj_type, set()).update(objs)
                for obj in objs:
                    new_fluents.add(F(f"found {obj}"))
                    new_fluents.add(F(f"at {obj} {loc}"))

        return new_fluents, updated_objects_by_type


class OngoingAction:
    """Base class for tracking ongoing action execution.

    Manages the lifecycle of an action from start to completion,
    including effect scheduling and environment interaction.
    """

    def __init__(
        self,
        time: Union[int, float],
        action: Action,
        environment: Optional[AbstractEnvironment] = None,
    ) -> None:
        """Initialize an ongoing action.

        Args:
            time: Start time of the action.
            action: The action being executed.
            environment: Optional environment for skill execution.
        """
        self.time = float(time)
        self.name = action.name
        self._start_time = float(time)
        self._action = action
        self._upcoming_effects: List[Tuple[float, GroundedEffect]] = sorted(
            [(time + eff.time, eff) for eff in action.effects], key=lambda el: el[0]
        )
        self.environment = environment
        self.is_action_called = False
        self.do_add_all_remaining_effects = False
        self.robot = self.name.split()[1]  # (e.g., move r1 locA locB)

    @property
    def time_to_next_event(self) -> float:
        """Time until the next scheduled effect."""
        if self._upcoming_effects:
            return self._upcoming_effects[0][0]
        else:
            return float("inf")

    @property
    def is_done(self) -> bool:
        """Whether all effects have been applied."""
        return not self.upcoming_effects

    @property
    def is_action_complete(self) -> bool:
        """Whether the environment reports the action as complete."""
        if self.is_action_called:
            assert self.environment is not None
            action_name = self.name.split()[0]
            action_status = self.environment.get_executed_skill_status(self.robot, action_name)
            if action_status == SkillStatus.DONE:
                return True
        return False

    @property
    def upcoming_effects(self) -> List[Tuple[float, GroundedEffect]]:
        """Return remaining upcoming effects."""
        return self._upcoming_effects

    def advance(self, time: float) -> List[Tuple[float, GroundedEffect]]:
        """Advance the action to a new time.

        Args:
            time: The new time to advance to.

        Returns:
            List of effects that occurred during the advance.
        """
        # Update the internal time
        self.time = time

        # Pop and return all effects scheduled at or before the new time
        new_effects = [effect for effect in self._upcoming_effects if effect[0] <= time + 1e-9]
        # Remove the new_effects from upcoming_effects (effects are sorted)
        self._upcoming_effects = self._upcoming_effects[len(new_effects) :]

        # If the action is complete, add all the remaining effects immediately
        # This is so that we can add fluents like just-picked, just-moved, etc. after delta_time
        # of action completion. These just-picked, just-moved were added so that it can be used in
        # preconditions to prevent the same action being taken again immediately for planning efficiency.

        if self.do_add_all_remaining_effects:
            new_effects += self._upcoming_effects
            self._upcoming_effects = []

        return new_effects

    def interrupt(self) -> Set[Fluent]:
        """Interrupt this action.

        Returns:
            Set of fluents that result from the interruption.
        """
        return set()

    def __str__(self) -> str:
        return f"OngoingAction<{self.name}, {self.time}, {self.upcoming_effects}>"


class OngoingSearchAction(OngoingAction):
    """Ongoing action for search operations."""

    def advance(self, time: float) -> List[Tuple[float, GroundedEffect]]:
        assert self.environment is not None
        _, _, loc, obj = self.name.split()  # (e.g., search r1 locA objA)
        if not self.is_action_called:
            self.environment.execute_skill(self.robot, "search", loc, obj)
            self.is_action_called = True
        return super().advance(time)


class OngoingPickAction(OngoingAction):
    """Ongoing action for pick operations."""

    def advance(self, time: float) -> List[Tuple[float, GroundedEffect]]:
        assert self.environment is not None
        _, _, loc, obj = self.name.split()  # (e.g., pick r1 locA objA)
        if not self.is_action_called:
            self.environment.execute_skill(self.robot, "pick", loc, obj)
            self.is_action_called = True

        new_effects = super().advance(time)
        if self.is_done:
            # remove the object from the location
            self.environment.remove_object_from_location(obj, loc)
        return new_effects


class OngoingPlaceAction(OngoingAction):
    """Ongoing action for place operations."""

    def advance(self, time: float) -> List[Tuple[float, GroundedEffect]]:
        assert self.environment is not None
        _, _, loc, obj = self.name.split()  # (e.g., place r1 locA objA)
        if not self.is_action_called:
            self.environment.execute_skill(self.robot, "place", loc, obj)
            self.is_action_called = True

        new_effects = super().advance(time)
        if self.is_done:
            # add the object to the location
            self.environment.add_object_at_location(obj, loc)
        return new_effects


class OngoingMoveAction(OngoingAction):
    """Ongoing action for move operations with interruption support."""

    def __init__(
        self,
        time: Union[int, float],
        action: Action,
        environment: Optional[AbstractEnvironment] = None,
    ) -> None:
        super().__init__(time, action, environment)
        # Keep track of initial start and end locations
        _, self.robot, self.start, self.end = self.name.split()  # (e.g., move r1 locA locB)

    def advance(self, time: float) -> List[Tuple[float, GroundedEffect]]:
        assert self.environment is not None
        if not self.is_action_called:
            self.environment.execute_skill(self.robot, "move", self.start, self.end)
            self.is_action_called = True
        return super().advance(time)

    def interrupt(self) -> Set[Fluent]:
        """Interrupt a move and update to intermediate location.

        Returns:
            Set of fluents for the intermediate location.

        Raises:
            ValueError: If attempting to interrupt probabilistic effects.
        """
        assert self.environment is not None

        if self.time <= self._start_time:
            return set()  # Cannot interrupt before start time

        # This action is done. Treat this as having "reached" the destination
        # but where the destination is robot_loc, which means we must replace
        # all the old "target location" with "robot_loc". While this may seem
        # like a fair bit of needless complexity, it means that we don't need to
        # have a custom function for each new move action: all it's
        # post-conditions are added automatically.

        # stop robot
        self.environment.stop_robot(self.robot)
        robot = self.robot
        old_target = self.end
        new_target = f"{robot}_loc"
        new_fluents: Set[Fluent] = set()

        for _, eff in self._upcoming_effects:
            if eff.is_probabilistic:
                raise ValueError("Probabilistic effects cannot be interrupted.")
            sorted_fluents = sorted(eff.resulting_fluents, key=lambda f: f.negated, reverse=True)
            for fluent in sorted_fluents:
                if (~fluent) in new_fluents:
                    new_fluents.remove(~fluent)
                new_fluents.add(
                    F(
                        " ".join([fluent.name] + [fa if fa != old_target else new_target for fa in fluent.args]),
                        negated=fluent.negated,
                    )
                )

        self._upcoming_effects = []
        return new_fluents


class OngoingNoOpAction(OngoingAction):
    """Ongoing action for no-op (wait) operations."""

    def advance(self, time: float) -> List[Tuple[float, GroundedEffect]]:
        assert self.environment is not None
        if not self.is_action_called:
            self.environment.execute_skill(self.robot, "no_op")
            self.is_action_called = True
        return super().advance(time)
