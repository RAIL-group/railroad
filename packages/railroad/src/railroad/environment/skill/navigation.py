"""Navigation move skill with path following and symbolic effect execution."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np

from railroad._bindings import Action, Fluent

from ..navigation.pathing import get_coordinates_at_distance, path_total_length
from ..symbolic import SymbolicSkill
from ..types import Pose
from .protocols import MotionSkill, SupportsMovePathEnvironment

if TYPE_CHECKING:
    from ..environment import Environment


class NavigationMoveSkill(SymbolicSkill, MotionSkill):
    """Move skill with path-following and effect scheduling."""

    def __init__(
        self,
        action: Action,
        start_time: float,
        env: SupportsMovePathEnvironment,
    ) -> None:
        super().__init__(action=action, start_time=start_time)

        # Parse "move robot from to"
        parts = action.name.split()
        self._robot = parts[1]
        self._move_origin = parts[2]
        self._move_destination = parts[3]

        # Keep a runtime guard for dynamic callers that bypass static typing.
        compute_move_path = getattr(env, "compute_move_path", None)
        if not callable(compute_move_path):
            raise TypeError(
                "NavigationMoveSkill requires env.compute_move_path(loc_from, loc_to)"
            )
        compute_move_path_fn: Callable[[str, str], np.ndarray] = compute_move_path
        self._path = compute_move_path_fn(
            self._move_origin,
            self._move_destination,
        )
        if self._path.size == 0 or self._path.shape[0] != 2:
            raise ValueError(
                f"No traversable path for action: {action.name}"
            )
        self._path_length = path_total_length(self._path)

        # Use the action's "free robot" effect as the move completion time.
        free_effect_times = [
            float(eff.time)
            for eff in action.effects
            if Fluent("free", self._robot) in eff.resulting_fluents
        ]
        planned_duration = min(free_effect_times) if free_effect_times else 0.0
        self._end_time = start_time + planned_duration

        # Effective speed along the realized trajectory to match planned timing.
        self._speed = (
            (self._path_length / planned_duration)
            if self._path_length > 0 and planned_duration > 0.0
            else 0.0
        )

    @property
    def controlled_robot(self) -> str:
        return self._robot

    def is_motion_active_at(self, time: float) -> bool:
        if self.is_done:
            return False
        return self._start_time <= time < self._end_time - 1e-9

    def _interpolate_pose(self, time: float) -> Pose:
        """Compute robot pose at given time by interpolating along path."""
        elapsed = time - self._start_time
        distance = elapsed * self._speed
        coords = get_coordinates_at_distance(self._path, distance)

        # Compute yaw from path direction
        look_ahead = min(distance + 1.0, self._path_length)
        ahead_coords = get_coordinates_at_distance(self._path, look_ahead)
        dx = ahead_coords[0] - coords[0]
        dy = ahead_coords[1] - coords[1]
        yaw = math.atan2(dy, dx) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else 0.0

        return Pose(x=float(coords[0]), y=float(coords[1]), yaw=yaw)

    def advance(self, time: float, env: Environment) -> None:
        """Advance to given time: update pose and apply due effects."""
        self._current_time = time

        # Update robot pose along path
        pose = self._interpolate_pose(time)
        env.set_robot_pose(self._robot, pose)

        # Delegate symbolic effect execution to shared base logic.
        super().advance(time, env)

    def _destination_is_stale(self, env: Environment) -> bool:
        """Whether this move's destination is no longer planner-usable."""
        # Prefer symbolic location typing when available.
        locations = set(env.objects_by_type.get("location", set()))
        if locations and self._move_destination not in locations:
            return True

        # Also respect registry removals when environments prune coordinates.
        registry = getattr(env, "location_registry", None)
        if registry is not None and self._move_destination not in registry:
            return True

        return False

    def _interrupt_with_destination_rewrite(
        self,
        env: Environment,
    ) -> None:
        """Interrupt move and rewrite pending effects to a robot-local anchor."""
        if self._current_time <= self._start_time:
            return
        if self.is_done:
            return
        if self._move_destination is None or self._robot is None:
            return

        # Get current interpolated pose
        pose = self._interpolate_pose(self._current_time)
        new_target = f"{self._robot}_loc"
        old_target = self._move_destination

        # Register intermediate location in location_registry
        registry = getattr(env, "location_registry", None)
        if registry is not None:
            registry.register(new_target, np.array([pose.x, pose.y]))

        # Keep continuous pose state in sync at the interrupt instant.
        env.set_robot_pose(self._robot, pose)

        # Collect and rewrite fluents from remaining effects
        new_fluents: set[Fluent] = set()
        for _, eff in self._upcoming_effects:
            if eff.is_probabilistic:
                # Drop probabilistic effects on interrupt per spec
                continue
            for fluent in eff.resulting_fluents:
                if (~fluent) in new_fluents:
                    new_fluents.remove(~fluent)
                new_args = [
                    arg if arg != old_target else new_target
                    for arg in fluent.args
                ]
                new_fluents.add(
                    Fluent(" ".join([fluent.name] + new_args), negated=fluent.negated)
                )

        # Apply rewritten fluents to environment
        for fluent in new_fluents:
            if fluent.negated:
                env.fluents.discard(~fluent)
            else:
                env.fluents.add(fluent)

        # Clear remaining effects
        self._upcoming_effects = []

    def interrupt(self, env: Environment) -> None:
        """Interrupt only when destination became stale."""
        if not self._destination_is_stale(env):
            return
        self._interrupt_with_destination_rewrite(env)


class InterruptibleNavigationMoveSkill(NavigationMoveSkill):
    """Navigation move skill that supports unconditional interruption."""

    def __init__(
        self,
        action: Action,
        start_time: float,
        env: SupportsMovePathEnvironment,
    ) -> None:
        super().__init__(action=action, start_time=start_time, env=env)
        self._is_interruptible = True

    def interrupt(self, env: Environment) -> None:
        """Interrupt move immediately when requested by the environment."""
        self._interrupt_with_destination_rewrite(env)
