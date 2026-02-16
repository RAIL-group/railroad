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

    Subclasses implement environment-specific behavior:
    - How fluents are stored/retrieved
    - How skills are created for actions
    - How effects are applied
    - How probabilistic effects are resolved
    """

    def __init__(
        self,
        state: State,
        operators: List[Operator],
    ) -> None:
        self._operators = operators
        self._time: float = state.time
        self._active_skills: List[ActiveSkill] = []

        # Convert initial upcoming effects to a skill
        if state.upcoming_effects:
            initial_skill = self._create_initial_effects_skill(
                state.time, list(state.upcoming_effects)
            )
            self._active_skills.append(initial_skill)

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

    @abstractmethod
    def apply_effect(
        self, effect: GroundedEffect
    ) -> List[Tuple[float, GroundedEffect]]:
        """Apply an effect to the environment.

        Returns:
            List of (relative_time, effect) tuples for effects that should be
            scheduled later (e.g., nested effects inside prob_effects with time > 0).
            The caller is responsible for scheduling these at current_time + relative_time.
        """
        ...

    @abstractmethod
    def resolve_probabilistic_effect(
        self,
        effect: GroundedEffect,
        current_fluents: Set[Fluent],
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        """Resolve which branch of a probabilistic effect occurs."""
        ...

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
        if any(math.isinf(eff.time) or math.isnan(eff.time) for eff in action.effects):
            return False

        parts = action.name.split()
        if not parts:
            return False

        if parts[0] == "move":
            if len(parts) > 3 and parts[2] == parts[3]:
                return False
            if action.effects and all(eff.time <= 1e-9 for eff in action.effects):
                return False
            if len(parts) > 3 and parts[3].endswith("_loc"):
                return False

        if parts[0] == "place" and len(parts) > 2 and parts[2].endswith("_loc"):
            return False
        if parts[0] == "search" and len(parts) > 2 and parts[2].endswith("_loc"):
            return False

        return True

    def is_goal_reached(self, goal_fluents: Collection[Fluent]) -> bool:
        """Check if all goal fluents are satisfied."""
        return all(f in self.state.fluents for f in goal_fluents)

    def _any_robot_free(self) -> bool:
        """Check if any robot is free."""
        return any(f.name == "free" for f in self.fluents)

    def interrupt_skills(self) -> None:
        """Interrupt active interruptible skills according to env policy.

        Args:
            force: If True, interrupt even when no robot is currently free.

        Returns:
            True if at least one skill was interrupted.
        """
        for skill in self._active_skills:
            skill.interrupt(self)

    def _should_interrupt_skills(self) -> bool:
        """Internal predicate hook for early loop interruption."""
        return False

    def _cap_next_advance_time(self, proposed_next_time: float) -> float:
        """Optional hook to constrain the scheduler's next advance time."""
        return proposed_next_time

    def _after_skills_advanced(self, advanced_to_time: float) -> None:
        """Optional hook invoked after skills advance and cleanup."""
        del advanced_to_time

    def set_robot_pose(self, robot: str, pose: object) -> None:
        """Optional hook for environments that track continuous robot pose."""
        del robot, pose

    def act(
        self,
        action: Action,
        loop_callback_fn: Optional[Callable[[], None]] = None,
    ) -> State:
        """Execute action, return state when a robot is free for new dispatch.

        Args:
            action: The action to execute.
            loop_callback_fn: Optional callback called each iteration.

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

        # Continue until any robot becomes free or interrupt requested
        while not self._any_robot_free():
            if all(s.is_done for s in self._active_skills):
                break

            if self._should_interrupt_skills():
                break

            skill_times = [s.time_to_next_event for s in self._active_skills] or [float("inf")]
            next_time = min(skill_times)
            if next_time == float("inf"):
                break
            next_time = self._cap_next_advance_time(next_time)
            if next_time == float("inf"):
                break
            if next_time <= self._time + 1e-12:
                next_time = min(
                    (t for t in skill_times if t > self._time + 1e-12),
                    default=float("inf"),
                )
                if next_time == float("inf"):
                    break

            # Advance all skills to next event time
            for s in self._active_skills:
                s.advance(next_time, self)

            self._time = next_time
            self._active_skills = [s for s in self._active_skills if not s.is_done]
            self._after_skills_advanced(next_time)

            if loop_callback_fn is not None:
                loop_callback_fn()

        # Interrupt once after loop exit. This handles both explicit interrupt
        # requests and the case where a non-interruptible action freed a robot
        # while interruptible skills remain active.
        self.interrupt_skills()

        self._active_skills = [s for s in self._active_skills if not s.is_done]
        return self.state
