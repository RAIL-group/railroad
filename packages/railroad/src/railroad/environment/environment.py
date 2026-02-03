"""Base Environment class for planning environments."""

import itertools
import math
from abc import ABC, abstractmethod
from typing import Callable, Collection, Dict, List, Optional, Set, Tuple

from railroad._bindings import Action, Fluent, GroundedEffect, State
from railroad.core import Operator

from .skill import ActiveSkill, SymbolicSkill


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

    def _create_initial_effects_skill(
        self,
        start_time: float,
        upcoming_effects: List[Tuple[float, GroundedEffect]],
    ) -> SymbolicSkill:
        """Create a SymbolicSkill from initial upcoming effects."""
        relative_effects = [
            GroundedEffect(abs_time - start_time, effect.resulting_fluents)
            for abs_time, effect in upcoming_effects
        ]
        action = Action([], relative_effects, name="_initial_effects")
        return SymbolicSkill(action=action, start_time=start_time)

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
    def apply_effect(self, effect: GroundedEffect) -> None:
        """Apply an effect to the environment."""
        ...

    @abstractmethod
    def resolve_probabilistic_effect(
        self,
        effect: GroundedEffect,
        current_fluents: Set[Fluent],
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        """Resolve which branch of a probabilistic effect occurs."""
        ...
