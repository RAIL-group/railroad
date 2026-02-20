"""Skill-related protocol definitions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


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
