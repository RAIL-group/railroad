"""Base Environment class for planning environments."""

import itertools
import math
from abc import ABC, abstractmethod
from typing import Callable, Collection, Dict, List, Optional, Set, Tuple

from railroad._bindings import Action, Fluent, GroundedEffect, State
from railroad.core import Operator

from .skill import ActiveSkill


class Environment(ABC):
    """Base class for planning environments.

    Provides concrete implementations for:
    - Active skill tracking and time management
    - State assembly (fluents + upcoming effects)
    - Action instantiation from operators
    - The act() loop (execute until a robot is free)
    - Common effect application and probabilistic resolution
    """

    def __init__(
        self,
        state: State,
        operators: List[Operator] | None = None,
    ) -> None:
        self._operators = operators or []
        self._time: float = state.time
        self._active_skills: List[ActiveSkill] = []

        # Convert initial upcoming effects to a skill
        if state.upcoming_effects:
            initial_skill = self._create_initial_effects_skill(
                state.time, list(state.upcoming_effects)
            )
            self._active_skills.append(initial_skill)

    @property
    def operators(self) -> List[Operator]:
        return self._operators

    @operators.setter
    def operators(self, value: List[Operator]) -> None:
        self._operators = value

    @abstractmethod
    def _create_initial_effects_skill(
        self,
        start_time: float,
        upcoming_effects: List[Tuple[float, GroundedEffect]],
    ) -> ActiveSkill:
        """Create an ActiveSkill from initial upcoming effects."""
        ...

    @property
    def time(self) -> float:
        return self._time

    @property
    @abstractmethod
    def fluents(self) -> Set[Fluent]:
        """Current ground truth fluents."""
        ...

    @property
    @abstractmethod
    def objects_by_type(self) -> Dict[str, Set[str]]:
        """All known objects, organized by type."""
        ...

    @abstractmethod
    def create_skill(self, action: Action, time: float) -> ActiveSkill:
        """Create an ActiveSkill for this action."""
        ...

    def apply_effect(
        self, effect: GroundedEffect
    ) -> List[Tuple[float, GroundedEffect]]:
        """Apply an effect to the environment.

        Returns:
            List of (relative_time, effect) tuples for effects that should be
            scheduled later (e.g., nested effects inside prob_effects with time > 0).
            The caller is responsible for scheduling these at current_time + relative_time.
        """
        delayed_effects: List[Tuple[float, GroundedEffect]] = []

        # Apply deterministic resulting_fluents
        for fluent in effect.resulting_fluents:
            if fluent.negated:
                self.fluents.discard(~fluent)
            else:
                self.fluents.add(fluent)

        # Handle probabilistic branches if present
        if effect.is_probabilistic:
            nested_effects, _ = self.resolve_probabilistic_effect(
                effect, self.fluents
            )
            for nested in nested_effects:
                if nested.time > 1e-9:
                    # Schedule for later - time is relative to when parent fired
                    delayed_effects.append((nested.time, nested))
                else:
                    # Apply immediately and collect any further delayed effects
                    delayed_effects.extend(self.apply_effect(nested))

        # Handle revelation (objects discovered when locations searched)
        self._handle_revelation()

        return delayed_effects

    def _handle_revelation(self) -> None:
        """Reveal objects when locations are searched."""
        for fluent in list(self.fluents):
            if fluent.name == "searched":
                location = fluent.args[0]
                revealed_fluent = Fluent("revealed", location)

                if revealed_fluent not in self.fluents:
                    self.fluents.add(revealed_fluent)

                    # Reveal objects at this location
                    objs_at_loc = self.get_objects_at_location(location)
                    for obj_type, objs in objs_at_loc.items():
                        for obj in objs:
                            self.fluents.add(Fluent("found", obj))
                            self.fluents.add(Fluent("at", obj, location))
                            self.objects_by_type.setdefault(obj_type, set()).add(obj)

    def get_objects_at_location(self, location: str) -> Dict[str, Set[str]]:
        """Get ground truth objects at a location (perception).

        Subclasses should override this if they have a perception system.
        """
        return {}

    def resolve_probabilistic_effect(
        self,
        effect: GroundedEffect,
        current_fluents: Set[Fluent],
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        """Resolve which branch of a probabilistic effect occurs."""
        if not effect.is_probabilistic:
            return [effect], current_fluents

        branches = effect.prob_effects
        if not branches:
            return [], current_fluents

        # Find success branch (contains positive "found" fluent)
        success_branch = None
        target_object = None
        location = None

        for branch in branches:
            _, branch_effects = branch
            for eff in branch_effects:
                for fluent in eff.resulting_fluents:
                    if fluent.name == "found" and not fluent.negated:
                        success_branch = branch
                        target_object = fluent.args[0]
                        location = self._find_search_location(eff, target_object)

        # If we can resolve from ground truth, do so deterministically
        if success_branch and target_object and location:
            objs_at_loc = self.get_objects_at_location(location)
            # Find if object is in any type category at this location
            is_present = any(target_object in objs for objs in objs_at_loc.values())

            if is_present:
                _, effects = success_branch
                return list(effects), current_fluents

            # Object not at location - return failure branch deterministically
            other_branches = [b for b in branches if b is not success_branch]
            if other_branches:
                _, effects = other_branches[0]
                return list(effects), current_fluents

        # Can't determine from ground truth or not a search - sample from distribution
        import random
        probs = [p for p, _ in branches]
        _, effects = random.choices(branches, weights=probs, k=1)[0]
        return list(effects), current_fluents

    def _find_search_location(
        self, effect: GroundedEffect, target_object: str
    ) -> str | None:
        """Find the location from 'at object location' fluent in a branch."""
        for fluent in effect.resulting_fluents:
            if fluent.name == "at" and len(fluent.args) >= 2:
                if fluent.args[0] == target_object:
                    return fluent.args[1]
        return None

    @property
    def state(self) -> State:
        """Assemble state from fluents + upcoming effects from active skills."""
        self._update_skills()

        effects: List[Tuple[float, GroundedEffect]] = []
        for skill in self._active_skills:
            effects.extend(skill.upcoming_effects)

        return State(
            self._time,
            self.fluents,
            sorted(effects, key=lambda el: el[0]),
        )

    def _update_skills(self) -> None:
        """Advance skills to current time and remove completed ones."""
        for skill in self._active_skills:
            skill.advance(self._time, self)
        self._active_skills = [s for s in self._active_skills if not s.is_done]

    def get_actions(self) -> List[Action]:
        """Instantiate available actions from operators."""
        objects_by_type = self.objects_by_type

        # Add robot intermediate locations (robot_loc) to location set
        robot_locs = {
            f"{rob}_loc"
            for rob in objects_by_type.get("robot", set())
            if Fluent("at", rob, f"{rob}_loc") in self.fluents
        }
        objects_with_rloc: Dict[str, Collection[str]] = {
            k: set(v) for k, v in objects_by_type.items()
        }
        objects_with_rloc["location"] = (
            set(objects_with_rloc.get("location", set())) | robot_locs
        )

        all_actions: List[Action] = list(
            itertools.chain.from_iterable(
                op.instantiate(objects_with_rloc) for op in self._operators
            )
        )

        return [a for a in all_actions if self._is_valid_action(a)]

    def _is_valid_action(self, action: Action) -> bool:
        """Filter actions with infinite effects or invalid destinations."""
        if any(math.isinf(eff.time) for eff in action.effects):
            return False
        parts = action.name.split()
        if parts[0] == "move" and len(parts) > 3 and "_loc" in parts[3]:
            return False
        if parts[0] == "place" and len(parts) > 2 and "_loc" in parts[2]:
            return False
        if parts[0] == "search" and len(parts) > 2 and "_loc" in parts[2]:
            return False
        return True

    def is_goal_reached(self, goal_fluents: Collection[Fluent]) -> bool:
        """Check if all goal fluents are satisfied."""
        return all(f in self.state.fluents for f in goal_fluents)

    def _any_robot_free(self) -> bool:
        """Check if any robot is free."""
        return any(f.name == "free" for f in self.fluents)

    def act(
        self,
        action: Action,
        loop_callback_fn: Optional[Callable[[], None]] = None,
        do_interrupt: bool = True,
    ) -> State:
        """Execute action, return state when a robot is free for new dispatch.

        Args:
            action: The action to execute.
            loop_callback_fn: Optional callback called each iteration.
            do_interrupt: Whether to interrupt ongoing skills when returning.

        Returns:
            The new state after execution.

        Raises:
            ValueError: If action preconditions are not satisfied.
        """
        if not self.state.satisfies_precondition(action):
            raise ValueError(
                f"Action preconditions not satisfied: {action.name}"
            )

        skill = self.create_skill(action, self._time)
        self._active_skills.append(skill)

        # Apply immediate effects at current time
        for s in self._active_skills:
            s.advance(self._time, self)
        self._active_skills = [s for s in self._active_skills if not s.is_done]

        # Continue until any robot becomes free
        while not self._any_robot_free():
            if all(s.is_done for s in self._active_skills):
                break

            skill_times = [s.time_to_next_event for s in self._active_skills] or [float("inf")]
            next_time = min(skill_times)
            if next_time == float("inf"):
                break

            # Advance all skills to next event time
            for s in self._active_skills:
                s.advance(next_time, self)

            dt = next_time - self._time
            self._time = next_time
            self._on_act_loop_iteration(dt)

            # Identify skills that just finished
            newly_done = [s for s in self._active_skills if s.is_done]
            for skill in newly_done:
                self._on_skill_completed(skill)

            self._active_skills = [s for s in self._active_skills if not s.is_done]

            if loop_callback_fn is not None:
                loop_callback_fn()

        # Interrupt interruptible skills if requested
        if do_interrupt:
            for skill in self._active_skills:
                if skill.is_interruptible and not skill.is_done:
                    skill.interrupt(self)

        self._active_skills = [s for s in self._active_skills if not s.is_done]
        return self.state

    def _on_skill_completed(self, skill: ActiveSkill) -> None:
        """Hook called when a skill completes during act()."""
        pass

    def _on_act_loop_iteration(self, dt: float) -> None:
        """Hook called on each iteration of the act() loop.

        Args:
            dt: Time advanced in this iteration.
        """
        pass
