"""Navigation move skill with path following and symbolic effect execution."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np

from railroad._bindings import Action, Fluent

from ..skill import MotionSkill
from ..symbolic import SymbolicSkill
from . import pathing
from .types import Pose

if TYPE_CHECKING:
    from ..environment import Environment


class NavigationMoveSkill(SymbolicSkill, MotionSkill):
    """Move skill with path-following and effect scheduling.

    The skill:
    1. Builds a path at creation time using the observed grid.
    2. Advances the robot pose along the path over time.
    3. Applies symbolic effects at their scheduled times.
    """

    def __init__(
        self,
        action: Action,
        start_time: float,
        env: Environment,
    ) -> None:
        super().__init__(action=action, start_time=start_time)

        # Parse "move robot from to"
        parts = action.name.split()
        self._robot = parts[1]
        self._move_origin = parts[2]
        self._move_destination = parts[3]

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
        self._path_length = pathing.path_total_length(self._path)

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
        coords = pathing.get_coordinates_at_distance(self._path, distance)

        # Compute yaw from path direction
        look_ahead = min(distance + 1.0, self._path_length)
        ahead_coords = pathing.get_coordinates_at_distance(self._path, look_ahead)
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


class InterruptibleNavigationMoveSkill(NavigationMoveSkill):
    """Navigation move skill that supports mid-execution interruption."""

    def __init__(
        self,
        action: Action,
        start_time: float,
        env: Environment,
    ) -> None:
        super().__init__(action=action, start_time=start_time, env=env)
        self._is_interruptible = True

    def interrupt(self, env: Environment) -> None:
        """Interrupt move: place robot at current pose, rewrite effects."""
        if self._current_time <= self._start_time:
            return
        if self.is_done:
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
