"""Navigation move skill with path following and laser sensing."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from railroad._bindings import Action, Fluent, GroundedEffect

from ..skill import ActiveSkill
from . import pathing
from .types import Pose

if TYPE_CHECKING:
    from ..environment import Environment
    from .environment import UnknownSpaceEnvironment


class NavigationMoveSkill:
    """Interruptible move skill with continuous sensing along a grid path.

    The skill:
    1. Builds a path at creation time using the observed grid.
    2. Advances the robot pose along the path over time.
    3. Performs laser observations at regular intervals (sensor_dt).
    4. Applies symbolic effects at their scheduled times.
    5. On interrupt, places the robot at its current interpolated pose
       and rewrites pending effects to a transient location.
    """

    def __init__(
        self,
        action: Action,
        start_time: float,
        env: UnknownSpaceEnvironment,
    ) -> None:
        self._action = action
        self._start_time = start_time
        self._current_time = start_time

        # Parse "move robot from to"
        parts = action.name.split()
        self._robot = parts[1]
        self._move_origin = parts[2]
        self._move_destination = parts[3]

        # Build path from observed grid
        self._path = env.compute_move_path(self._move_origin, self._move_destination)
        self._path_length = pathing.path_total_length(self._path)

        # Compute move duration from path
        speed = env.config.speed_cells_per_sec
        self._move_duration = max(0.01, self._path_length / speed) if self._path_length > 0 else 0.01
        self._end_time = start_time + self._move_duration

        # Build effect timeline — rewrite move_time effects to use actual path duration
        self._upcoming_effects: List[Tuple[float, GroundedEffect]] = []
        for eff in action.effects:
            if eff.time <= 1e-9:
                # Immediate effect at start time
                self._upcoming_effects.append((start_time, eff))
            else:
                # Delayed effect — use our computed end time
                self._upcoming_effects.append((self._end_time, eff))
        self._upcoming_effects.sort(key=lambda el: el[0])

        # Sensing state
        self._sensor_dt = env.config.sensor_dt
        self._last_sense_time = start_time
        self._distance_traveled = 0.0

        # Track starting pose for yaw computation
        self._env_ref = env

        # Perform initial observation at the start pose
        robot_pose = env.robot_poses.get(self._robot)
        if robot_pose is not None:
            env.observe_from_pose(self._robot, robot_pose, start_time)

    @property
    def is_done(self) -> bool:
        return len(self._upcoming_effects) == 0

    @property
    def is_interruptible(self) -> bool:
        return True

    @property
    def upcoming_effects(self) -> List[Tuple[float, GroundedEffect]]:
        return self._upcoming_effects

    @property
    def time_to_next_event(self) -> float:
        times = []
        if self._upcoming_effects:
            times.append(self._upcoming_effects[0][0])
        # Next sensing time
        next_sense = self._last_sense_time + self._sensor_dt
        if next_sense < self._end_time + 1e-9:
            times.append(next_sense)
        return min(times) if times else float("inf")

    def _interpolate_pose(self, time: float) -> Pose:
        """Compute robot pose at given time by interpolating along path."""
        elapsed = time - self._start_time
        distance = elapsed * self._env_ref.config.speed_cells_per_sec
        coords = pathing.get_coordinates_at_distance(self._path, distance)

        # Compute yaw from path direction
        look_ahead = min(distance + 1.0, self._path_length)
        ahead_coords = pathing.get_coordinates_at_distance(self._path, look_ahead)
        dx = ahead_coords[0] - coords[0]
        dy = ahead_coords[1] - coords[1]
        yaw = math.atan2(dy, dx) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else 0.0

        return Pose(x=float(coords[0]), y=float(coords[1]), yaw=yaw)

    def advance(self, time: float, env: Environment) -> None:
        """Advance to given time: update pose, sense, apply due effects."""
        self._current_time = time

        # Update robot pose along path
        pose = self._interpolate_pose(time)
        nav_env: UnknownSpaceEnvironment = env  # type: ignore[assignment]
        nav_env.robot_poses[self._robot] = pose

        # Perform sensing at regular intervals
        while self._last_sense_time + self._sensor_dt <= time + 1e-9:
            self._last_sense_time += self._sensor_dt
            nav_env.observe_from_pose(self._robot, pose, self._last_sense_time)

        # Apply due effects
        due_effects = [
            (t, eff) for t, eff in self._upcoming_effects
            if t <= time + 1e-9
        ]
        self._upcoming_effects = self._upcoming_effects[len(due_effects):]

        for effect_time, effect in due_effects:
            delayed = env.apply_effect(effect)
            for relative_time, delayed_effect in delayed:
                abs_time = effect_time + relative_time
                self._upcoming_effects.append((abs_time, delayed_effect))

        if due_effects:
            self._upcoming_effects.sort(key=lambda el: el[0])

    def interrupt(self, env: Environment) -> None:
        """Interrupt move: place robot at current pose, rewrite effects."""
        if self._current_time <= self._start_time:
            return
        if self.is_done:
            return

        nav_env: UnknownSpaceEnvironment = env  # type: ignore[assignment]

        # Get current interpolated pose
        pose = self._interpolate_pose(self._current_time)
        new_target = f"{self._robot}_loc"
        old_target = self._move_destination

        # Register intermediate location in location_registry
        registry = nav_env.location_registry
        if registry is not None:
            registry.register(new_target, np.array([pose.x, pose.y]))

        # Update robot pose
        nav_env.robot_poses[self._robot] = pose

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

        # Clear claim on old target if it was set
        claim_fluent = Fluent("claimed", old_target)
        env.fluents.discard(claim_fluent)  # type: ignore[union-attr]

        # Clear remaining effects
        self._upcoming_effects = []

        # Final observation at interrupt point
        nav_env.observe_from_pose(self._robot, pose, self._current_time,
                                  allow_interrupt=False)
