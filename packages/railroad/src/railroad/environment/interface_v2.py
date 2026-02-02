"""New EnvironmentInterface using Environment/ActiveSkill architecture."""

import itertools
import math
from typing import Callable, Collection, Dict, List, Optional, Set

from railroad._bindings import Action, Fluent, State, GroundedEffect
from railroad.core import Operator

from .skill import ActiveSkill, Environment

F = Fluent


class EnvironmentInterfaceV2:
    """Interface between PDDL planning and environment execution (v2).

    Coordinates execution without owning world state:
    - Tracks the list of active skills
    - Maintains current time
    - Holds operators for action instantiation
    - Assembles State for planner (fluents from Environment + upcoming effects from skills)
    - Runs the advance loop until a robot becomes free
    """

    def __init__(
        self,
        environment: Environment,
        operators: List[Operator],
    ) -> None:
        self._environment = environment
        self._operators = operators
        self._active_skills: List[ActiveSkill] = []

        # Initialize time and upcoming effects from environment's initial_state if available
        initial_state = getattr(environment, "initial_state", None)
        if initial_state is not None:
            self._time: float = initial_state.time
            self._initial_upcoming_effects: List[tuple[float, GroundedEffect]] = list(
                initial_state.upcoming_effects
            )
        else:
            self._time = 0.0
            self._initial_upcoming_effects = []

    @property
    def time(self) -> float:
        return self._time

    def _update_skills(self) -> None:
        """Update all active skills to check for completion and apply effects.

        This ensures fluent state stays synchronized with physical execution.
        """
        for skill in self._active_skills:
            skill.advance(self._time, self._environment)
        self._active_skills = [s for s in self._active_skills if not s.is_done]

    def _apply_initial_effects(self, time: float) -> None:
        """Apply initial upcoming effects that are due at or before the given time."""
        due_effects = [(t, e) for t, e in self._initial_upcoming_effects if t <= time + 1e-9]
        for _, effect in due_effects:
            self._environment.apply_effect(effect)
        self._initial_upcoming_effects = [
            (t, e) for t, e in self._initial_upcoming_effects if t > time + 1e-9
        ]

    def _next_initial_effect_time(self) -> float:
        """Return the time of the next initial upcoming effect, or inf if none."""
        if not self._initial_upcoming_effects:
            return float("inf")
        return min(t for t, _ in self._initial_upcoming_effects)

    @property
    def state(self) -> State:
        """Assemble state from Environment fluents + ActiveSkill upcoming effects."""
        # Update skills to apply any completed effects before assembling state
        self._update_skills()

        # Combine initial upcoming effects with active skill effects
        effects: List[tuple[float, GroundedEffect]] = []
        effects.extend(
            (t, e) for t, e in self._initial_upcoming_effects if t > self._time
        )
        for skill in self._active_skills:
            effects.extend(skill.upcoming_effects)

        return State(
            self._time,
            self._environment.fluents,
            sorted(effects, key=lambda el: el[0]),
        )

    def get_actions(self) -> List[Action]:
        """Instantiate available actions from operators and env.objects_by_type."""
        objects_by_type = self._environment.objects_by_type

        # Add robot locations to location set
        robot_locs = set(
            f"{rob}_loc"
            for rob in objects_by_type.get("robot", set())
            if F("at", rob, f"{rob}_loc") in self._environment.fluents
        )
        objects_with_rloc: Dict[str, Collection[str]] = {
            k: set(v) for k, v in objects_by_type.items()
        }
        objects_with_rloc["location"] = set(objects_with_rloc.get("location", set())) | robot_locs

        all_actions: List[Action] = list(
            itertools.chain.from_iterable(
                op.instantiate(objects_with_rloc) for op in self._operators
            )
        )

        def filter_intermediate_locations_as_destination(action: Action) -> bool:
            parts = action.name.split()
            if parts[0] == "move" and len(parts) > 3 and "_loc" in parts[3]:
                return False
            if parts[0] == "place" and len(parts) > 2 and "_loc" in parts[2]:
                return False
            if parts[0] == "search" and len(parts) > 2 and "_loc" in parts[2]:
                return False
            return True

        def filter_infinite_effect_time(action: Action) -> bool:
            for eff in action.effects:
                if math.isinf(eff.time):
                    return False
            return True

        filtered = [a for a in all_actions if filter_infinite_effect_time(a)]
        filtered = [a for a in filtered if filter_intermediate_locations_as_destination(a)]
        return filtered

    def advance(
        self,
        action: Action,
        do_interrupt: bool = True,
        loop_callback_fn: Optional[Callable[[], None]] = None,
    ) -> State:
        """Execute action, return state when a robot is free for new dispatch."""
        if not self.state.satisfies_precondition(action):
            raise ValueError(
                f"Action preconditions not satisfied: {action.name} in state {self.state}"
            )

        skill = self._environment.create_skill(action, self._time)
        self._active_skills.append(skill)

        # Apply any immediate effects at the current time
        self._apply_initial_effects(self._time)
        for s in self._active_skills:
            s.advance(self._time, self._environment)
        self._active_skills = [s for s in self._active_skills if not s.is_done]

        # Continue until any robot becomes free (enables concurrent dispatch)
        while not self._any_robot_free():
            if all(s.is_done for s in self._active_skills) and not self._initial_upcoming_effects:
                break

            # Consider both skill effects and initial effects for next time
            skill_times = [s.time_to_next_event for s in self._active_skills] or [float("inf")]
            next_time = min(min(skill_times), self._next_initial_effect_time())
            if next_time == float("inf"):
                break

            # Apply effects at next_time
            self._apply_initial_effects(next_time)
            for s in self._active_skills:
                s.advance(next_time, self._environment)

            self._time = next_time
            self._active_skills = [s for s in self._active_skills if not s.is_done]

            if loop_callback_fn is not None:
                loop_callback_fn()

        if do_interrupt:
            for skill in self._active_skills:
                if skill.is_interruptible and not skill.is_done:
                    skill.interrupt(self._environment)

        self._active_skills = [s for s in self._active_skills if not s.is_done]
        return self.state

    def _any_robot_free(self) -> bool:
        return any(f.name == "free" for f in self._environment.fluents)

    def _is_robot_free(self, robot: Optional[str]) -> bool:
        """Check if a specific robot is free."""
        if robot is None:
            return self._any_robot_free()
        return F("free", robot) in self._environment.fluents

    def is_goal_reached(self, goal_fluents: Collection[Fluent]) -> bool:
        return all(f in self.state.fluents for f in goal_fluents)
