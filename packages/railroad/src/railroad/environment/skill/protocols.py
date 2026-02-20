"""Skill-related protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Protocol, Tuple, runtime_checkable

import numpy as np

from railroad._bindings import GroundedEffect

if TYPE_CHECKING:
    from ..environment import Environment


@runtime_checkable
class ActiveSkill(Protocol):
    """Protocol for tracking execution of a single action."""

    @property
    def is_done(self) -> bool:
        """Whether the skill has completed."""
        ...

    @property
    def is_interruptible(self) -> bool:
        """Whether this skill can be interrupted mid-execution."""
        ...

    @property
    def upcoming_effects(self) -> List[Tuple[float, GroundedEffect]]:
        """Effects still to occur, with expected times."""
        ...

    @property
    def time_to_next_event(self) -> float:
        """Time until next effect. May block in physical mode."""
        ...

    def advance(self, time: float, env: "Environment") -> None:
        """Advance to given time, apply due effects to environment."""
        ...

    def interrupt(self, env: "Environment") -> None:
        """Interrupt this skill, applying partial effects to environment."""
        ...


@runtime_checkable
class MotionSkill(Protocol):
    """Optional protocol for skills that provide continuous robot motion."""

    @property
    def controlled_robot(self) -> str:
        """Robot name controlled by this motion skill."""
        ...

    def is_motion_active_at(self, time: float) -> bool:
        """Whether the robot is actively moving at the given absolute time."""
        ...


@runtime_checkable
class SupportsMovePathEnvironment(Protocol):
    """Environment contract required by NavigationMoveSkill construction."""

    def compute_move_path(
        self,
        loc_from: str,
        loc_to: str,
        robot: str | None = None,
    ) -> np.ndarray:
        """Compute 2xN path between symbolic locations."""
        ...
