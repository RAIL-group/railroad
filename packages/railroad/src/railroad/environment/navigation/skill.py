"""Navigation move skill with path following and symbolic effect execution."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from railroad._bindings import Action, Fluent

from ..skill import MotionSkill
from ..symbolic import SymbolicSkill
from . import pathing
from .types import Pose

if TYPE_CHECKING:
    from ..environment import Environment
    from .environment import UnknownSpaceEnvironment


class NavigationMoveSkill(SymbolicSkill, MotionSkill):
    """Interruptible move skill with path-following and effect scheduling.

    The skill:
    1. Builds a path at creation time using the observed grid.
    2. Advances the robot pose along the path over time.
    3. Applies symbolic effects at their scheduled times.
    4. On interrupt, places the robot at its current interpolated pose
       and rewrites pending effects to a transient location.
    """

    def __init__(
        self,
        action: Action,
        start_time: float,
        env: UnknownSpaceEnvironment,
    ) -> None:
        super().__init__(action=action, start_time=start_time)
        self._is_interruptible = env.config.move_execution_interruptible

        # Parse "move robot from to"
        parts = action.name.split()
        self._robot = parts[1]
        self._move_origin = parts[2]
        self._move_destination = parts[3]

        # Build path from observed grid. Prefer Theta* when configured, but
        # fall back to Dijkstra if any-angle planning fails from rounded
        # intermediate (robot_loc) coordinates.
        use_theta = env.config.move_execution_use_theta_star
        self._path = env.compute_move_path(
            self._move_origin,
            self._move_destination,
            use_theta=use_theta,
        )
        if use_theta and (self._path.size == 0 or self._path.shape[0] != 2):
            self._path = env.compute_move_path(
                self._move_origin,
                self._move_destination,
                use_theta=False,
            )
        if self._path.size == 0 or self._path.shape[0] != 2:
            raise ValueError(
                f"No traversable path for action: {action.name}"
            )
        self._path_length = pathing.path_total_length(self._path)

        # Preserve operator timeline as-is. Scale interpolation speed so the
        # trajectory reaches destination at the move's planned effect time.
        delayed_effect_times = [eff.time for eff in action.effects if eff.time > 1e-9]
        self._move_duration = min(delayed_effect_times) if delayed_effect_times else 0.01
        self._move_duration = max(0.01, float(self._move_duration))
        self._end_time = start_time + self._move_duration

        # Effective speed along the realized trajectory to match planned timing.
        self._speed = (
            (self._path_length / self._move_duration)
            if self._path_length > 0
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

    def interrupt(self, env: Environment) -> None:
        """Interrupt move: place robot at current pose, rewrite effects."""
        if not self._is_interruptible:
            return
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

        # Update robot pose
        # FIXME: I _think_ this is optional.
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
