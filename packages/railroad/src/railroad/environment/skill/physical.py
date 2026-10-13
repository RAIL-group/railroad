"""ActiveSkill that runs one action on a PhysicalEnvironment's robots."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from railroad._bindings import Action, Fluent, GroundedEffect

from ..environment import ActiveSkill
from ..types import Pose
from .navigation import destination_is_stale, retarget_remaining_effects
from .protocols import MotionSkill

if TYPE_CHECKING:
    from ..environment import Environment
    from ..physical import PhysicalEnvironment

_EPS = 1e-9


class SkillStatus(IntEnum):
    """What ``PhysicalEnvironment.get_skill_status`` reports for a robot's skill.

    IDLE is a robot's initial state, when nothing has been assigned to it or
    is running; a skill that sees it keeps waiting (the command has not been
    picked up yet). RUNNING keeps it waiting too; DONE and FAILED end it.
    """

    IDLE = -1
    RUNNING = 0
    DONE = 1
    FAILED = 2


@dataclass(frozen=True)
class SkillCommand:
    """One grounded action as dispatched to a robot.

    Attributes:
        robot: The robot executing the skill (the one the action frees).
        skill: The skill name -- the first token of the action name.
        args: The remaining action-name tokens, with ``robot`` removed.
        action: The grounded action.
        planned_duration: When the planner expects the robot to be free again.
        is_motion: Whether the skill moves the robot (destination = last arg).
        interruptible: Whether it stops when another robot frees.
    """

    robot: str
    skill: str
    args: tuple[str, ...]
    action: Action
    planned_duration: float
    is_motion: bool = False
    interruptible: bool = False

    @property
    def destination(self) -> str | None:
        if self.is_motion and self.args:
            return self.args[-1]
        return None


class Phase(Enum):
    """Lifecycle of an :class:`PhysicalSkill`."""

    PENDING = "pending"  # not started yet
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"  # the robot reported failure
    CANCELED = "canceled"  # interrupted by the environment

    @property
    def is_terminal(self) -> bool:
        return self not in (Phase.PENDING, Phase.RUNNING)


class PhysicalSkill(ActiveSkill, MotionSkill):
    """Execute one action on a :class:`~railroad.environment.PhysicalEnvironment`'s robots.

    The skill is started (``robots.execute_skill(robot, skill, *args)``) lazily, on the
    first :meth:`advance`, so constructing it never moves a robot. Then:

    - *Progress* effects -- those before the planned completion, when the
      action frees its robot -- apply on schedule while the robot works, e.g.
      ``not at r1 kitchen`` at t=0.
    - *Completion* effects are held until ``get_skill_status`` reports DONE, then
      apply at the actual time (later ones keep their offsets). Status is
      polled every ``poll_period``, and whenever the act() loop wakes, e.g.
      for a progress effect or a :meth:`PhysicalEnvironment.notify`. First,
      ``robots.observe(...)`` goes to :meth:`PhysicalEnvironment.record_observations`,
      so search outcomes follow what the robot saw.
    - On FAILED (or an exception from the robot) the completion effects are
      dropped and :meth:`PhysicalEnvironment.on_skill_failed` repairs the state --
      by default rolling back the progress effects -- so the planner replans.
    - :meth:`interrupt` calls ``stop_robot`` for interruptible moves (or moves
      whose destination disappeared) and leaves the robot at ``{robot}_loc``.
    """

    def __init__(
        self,
        action: Action,
        start_time: float,
        command: SkillCommand,
        robots: PhysicalEnvironment,
        poll_period: float = 1.0,
    ) -> None:
        if poll_period <= 0.0:
            raise ValueError(f"poll_period must be positive, got {poll_period}")
        self._action = action
        self._command = command
        self._robots = robots
        self._poll_period = poll_period
        self._start_time = start_time
        self._current_time = start_time
        self._planned_completion = start_time + command.planned_duration

        effects = sorted(
            ((start_time + eff.time, eff) for eff in action.effects),
            key=lambda el: el[0],
        )
        cutoff = self._planned_completion - _EPS
        # Absolute times.
        self._progress: List[Tuple[float, GroundedEffect]] = [
            (t, eff) for t, eff in effects if t < cutoff
        ]
        # Offsets from completion.
        self._held: List[Tuple[float, GroundedEffect]] = [
            (t - self._planned_completion, eff) for t, eff in effects if t >= cutoff
        ]
        # Absolute times; filled once the robot reports DONE.
        self._scheduled: List[Tuple[float, GroundedEffect]] = []
        # Fluents the progress effects changed -> whether each held before.
        self._progress_changes: Dict[Fluent, bool] = {}

        self._phase = Phase.PENDING
        self._message = ""
        self._started_wall = 0.0
        self._seen_running = False
        self._pose: Pose | None = None

    @property
    def command(self) -> SkillCommand:
        return self._command

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def message(self) -> str:
        """Why the skill failed, if it did."""
        return self._message

    @property
    def is_done(self) -> bool:
        return self._phase.is_terminal and not self._scheduled

    @property
    def is_interruptible(self) -> bool:
        return self._command.interruptible

    @property
    def controlled_robot(self) -> str:
        return self._command.robot

    def is_motion_active_at(self, time: float) -> bool:
        return (
            self._command.is_motion
            and self._phase == Phase.RUNNING
            and self._start_time <= time
        )

    @property
    def upcoming_effects(self) -> List[Tuple[float, GroundedEffect]]:
        if self._phase.is_terminal:
            return list(self._scheduled)
        # A robot running late is still running: never complete in the past.
        completion = max(self._planned_completion, self._current_time)
        effects = self._progress + [(completion + off, eff) for off, eff in self._held]
        return sorted(effects, key=lambda el: el[0])

    @property
    def time_to_next_event(self) -> float:
        if self._phase.is_terminal:
            return self._scheduled[0][0] if self._scheduled else float("inf")
        if self._phase == Phase.PENDING:
            return self._start_time
        next_time = self._current_time + self._poll_period
        if self._progress:
            next_time = min(next_time, self._progress[0][0])
        return next_time

    def advance(self, time: float, env: Environment) -> None:
        self._current_time = max(self._current_time, time)

        if self._phase.is_terminal:
            self._apply_due(self._scheduled, time, env)
            return

        if self._phase == Phase.PENDING:
            self._start()
            if self._phase.is_terminal:  # the robot refused
                self._fail(env)
                return

        before = set(env.fluents)
        self._apply_due(self._progress, time, env)
        for fluent in before.symmetric_difference(env.fluents):
            self._progress_changes.setdefault(fluent, fluent in before)

        status = self._poll()
        pose = self._call_quietly(self._robots.get_pose, self._command.robot)
        if pose is not None:
            self._pose = pose
            env.set_robot_pose(self._command.robot, pose)

        if status == SkillStatus.DONE:
            self._succeed(time, env)
        elif status == SkillStatus.FAILED:
            self._phase = Phase.FAILED
            self._fail(env)

    def _start(self) -> None:
        command = self._command
        self._started_wall = time.monotonic()
        self._phase = Phase.RUNNING
        try:
            self._robots.execute_skill(command.robot, command.skill, *command.args)
        except Exception as exc:  # report as a failure, don't crash act()
            self._phase = Phase.FAILED
            self._message = f"{command.skill} raised {type(exc).__name__}: {exc}"

    def _poll(self) -> SkillStatus:
        command = self._command
        try:
            status = SkillStatus(self._robots.get_skill_status(command.robot, command.skill))
        except Exception as exc:
            status = SkillStatus.FAILED
            self._message = f"get_skill_status raised {type(exc).__name__}: {exc}"
        if status == SkillStatus.RUNNING:
            self._seen_running = True
        elif (
            not self._seen_running
            and time.monotonic() - self._started_wall < self._robots.stale_status_grace
        ):
            # Probably still the previous command's status.
            status = SkillStatus.RUNNING
        return status

    @staticmethod
    def _call_quietly(fn, *args):
        try:
            return fn(*args)
        except Exception:
            return None

    def _succeed(self, time: float, env: Environment) -> None:
        self._phase = Phase.DONE
        command = self._command
        seen = self._call_quietly(self._robots.observe, command.robot, command.skill, *command.args)
        if seen is not None:
            self._robots.record_observations(command, seen)
        # Stable sort: leftover progress effects precede completion effects
        # that share the timestamp.
        self._scheduled = sorted(
            [(time, eff) for _, eff in self._progress]
            + [(time + off, eff) for off, eff in self._held],
            key=lambda el: el[0],
        )
        self._progress = []
        self._held = []
        self._apply_due(self._scheduled, time, env)

    def _fail(self, env: Environment) -> None:
        self._robots.on_skill_failed(self)
        self._progress = []
        self._held = []

    @staticmethod
    def _apply_due(
        queue: List[Tuple[float, GroundedEffect]],
        time: float,
        env: Environment,
    ) -> None:
        """Apply (in place) every effect in ``queue`` due by ``time``,
        scheduling delayed sub-effects relative to their parent."""
        while queue and queue[0][0] <= time + _EPS:
            effect_time, effect = queue.pop(0)
            for relative_time, delayed in env.apply_effect(effect):
                queue.append((effect_time + relative_time, delayed))
            queue.sort(key=lambda el: el[0])

    def interrupt(self, env: Environment) -> None:
        """Stop a running move and leave the robot at ``{robot}_loc``.

        Interruptible moves always stop; others only once their destination
        is no longer a planner location (e.g. a frontier that closed).
        Non-motion skills are never interrupted.
        """
        if not self._command.is_motion or self._phase != Phase.RUNNING:
            return
        if self._current_time <= self._start_time:
            return
        destination = self._command.destination
        if destination is None:
            return
        if not self._command.interruptible and not destination_is_stale(env, destination):
            return

        self._robots.stop_robot(self._command.robot)
        self._phase = Phase.CANCELED
        self.stop_at_robot_loc(env)

    def roll_back_progress(self, env: Environment) -> None:
        """Undo the fluent changes made by progress effects applied so far,
        e.g. releasing a ``lock-search`` claimed at t=0."""
        for fluent, held_before in self._progress_changes.items():
            if held_before:
                env.fluents.add(fluent)
            else:
                env.fluents.discard(fluent)
        self._progress_changes = {}

    def current_pose(self) -> Pose | None:
        """The robot's pose: fresh from the robots if they report one, else
        the last one seen."""
        pose = self._call_quietly(self._robots.get_pose, self._command.robot)
        return pose if pose is not None else self._pose

    def place_robot_at_robot_loc(self, env: Environment) -> None:
        """Put the robot at ``{robot}_loc`` -- registered at its current pose
        when known -- in place of wherever the state had it. Applies no other
        effect."""
        robot = self._command.robot
        anchor = f"{robot}_loc"
        pose = self.current_pose()
        if pose is not None:
            registry = getattr(env, "location_registry", None)
            if registry is not None:
                registry.register(anchor, np.array([pose.x, pose.y]))
            env.set_robot_pose(robot, pose)
        for fluent in [
            f for f in env.fluents if f.name == "at" and f.args and f.args[0] == robot
        ]:
            env.fluents.discard(fluent)
        env.fluents.add(Fluent("at", robot, anchor))

    def stop_at_robot_loc(self, env: Environment) -> None:
        """Apply the remaining effects with the destination retargeted to
        ``{robot}_loc`` at the robot's current pose, then drop them (the
        interrupt semantics shared with NavigationMoveSkill)."""
        destination = self._command.destination
        if destination is None:
            raise ValueError(
                f"stop_at_robot_loc needs a motion command: {self._action.name}"
            )
        remaining = self._progress + [
            (self._planned_completion + off, eff) for off, eff in self._held
        ]
        retarget_remaining_effects(
            env,
            robot=self._command.robot,
            old_target=destination,
            pose=self.current_pose(),
            remaining_effects=remaining,
        )
        self._progress = []
        self._held = []
