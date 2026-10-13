"""PhysicalEnvironment: execute planned actions on real robots or a simulator.

Subclass :class:`PhysicalEnvironment` the way the example scripts subclass
``ObjectSearchEnvironment``: add ``define_operators()`` plus
:meth:`~PhysicalEnvironment.execute_skill`,
:meth:`~PhysicalEnvironment.get_skill_status` and :meth:`~PhysicalEnvironment.stop_robot`::

    class MyPhysicalEnvironment(PhysicalEnvironment):
        def define_operators(self):
            return [move_op]

        def execute_skill(self, robot, skill, *args):   # start the skill; return at once
            if skill == "move":
                loc_from, loc_to = args
                ...

        def get_skill_status(self, robot, skill):       # SkillStatus.RUNNING / DONE / FAILED
            ...

        def stop_robot(self, robot):
            ...

    env = MyPhysicalEnvironment(state=..., objects_by_type=...)

Then plan and ``env.act(action)`` as usual. Skills listed in
:attr:`~PhysicalEnvironment.symbolic_skills` run symbolically instead.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from time import monotonic
from typing import Collection, Dict, List, Mapping, Set, Type

from railroad._bindings import Action, Fluent, State
from railroad.core import Operator

from .environment import ActiveSkill
from .object_search import ObjectSearchEnvironment
from .skill.physical import PhysicalSkill, SkillCommand, SkillStatus
from .symbolic import LocationRegistry
from .types import Pose

__all__ = ["PhysicalEnvironment", "SkillStatus"]


class PhysicalEnvironment(ObjectSearchEnvironment, ABC):
    """Environment whose actions run on real robots or an external simulator.

    Subclasses implement:

    - ``define_operators()``, as for every environment;
    - :meth:`execute_skill`, :meth:`get_skill_status` and :meth:`stop_robot`;
    - optionally :meth:`get_pose`, :meth:`observe`, :attr:`symbolic_skills`,
      :attr:`stale_status_grace`.

    The environment then runs in real time (scaled by ``realtime_factor``),
    polls each running skill, applies the
    action's completion effects when the robot reports DONE, and on FAILED
    rolls back what the action started so the planner replans. A failed or
    interrupted move leaves the robot at ``{robot}_loc``.
    ``skill_overrides={"move": InterruptibleNavigationMoveSkill}`` makes moves
    stop whenever another robot becomes free, as in simulation.

    Args:
        state, objects_by_type, operators, true_object_locations,
        skill_overrides, location_registry, seed:
            As for :class:`ObjectSearchEnvironment`. ``true_object_locations``
            is only a prior belief: what robots :meth:`observe` replaces it.
        realtime_factor: Planner time units per wall-clock second: 1.0 when
            durations are seconds and the robots are real; the speed-up for
            a simulator running faster than real time.
        poll_period: Longest interval between status polls of a running
            skill, in planner time units. A robot finishing is seen on the
            next poll, or at once if the subclass calls :meth:`notify`.
    """

    #: Seconds after a skill starts during which a DONE/FAILED status is not
    #: trusted unless RUNNING was seen first: a per-robot status service may
    #: still describe the previous command. Set 0 for robots that track each
    #: command themselves.
    stale_status_grace: float = 0.5

    #: Skills (operator names) that run symbolically, on their planned
    #: timing, instead of on the robots -- e.g. ``{"wait"}``.
    symbolic_skills: Collection[str] = frozenset()

    def __init__(
        self,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        operators: List[Operator] | None = None,
        true_object_locations: Dict[str, Set[str]] | None = None,
        skill_overrides: Dict[str, Type[ActiveSkill]] | None = None,
        location_registry: LocationRegistry | None = None,
        seed: int | None = None,
        *,
        realtime_factor: float = 1.0,
        poll_period: float = 1.0,
    ) -> None:
        if poll_period <= 0.0:
            raise ValueError(f"poll_period must be positive, got {poll_period}")
        if realtime_factor <= 0.0:
            raise ValueError(f"realtime_factor must be positive, got {realtime_factor}")
        self._realtime_factor = realtime_factor
        self._start_wall = monotonic()
        self._start_time = state.time
        self._poll_period = poll_period
        # The last state handed out; act() checks actions against it.
        self._observed_state: State | None = None
        self._wakeup = threading.Event()
        super().__init__(
            state=state,
            objects_by_type=objects_by_type,
            operators=operators,
            true_object_locations=true_object_locations,
            skill_overrides=skill_overrides,
            location_registry=location_registry,
            seed=seed,
        )

    # --- implemented by subclasses ------------------------------------------ #

    @abstractmethod
    def define_operators(self) -> List[Operator]:
        """Build operators for this environment instance."""
        ...

    @abstractmethod
    def execute_skill(self, robot: str, skill: str, *args: str) -> None:
        """Start ``skill`` on ``robot`` and return without waiting for it.

        ``skill`` is the operator name and ``args`` the action's other
        arguments, in operator order -- e.g. ``("move", "kitchen",
        "bedroom")`` for ``move robot1 kitchen bedroom``. Raise if the robot
        cannot run it; that counts as a failure."""
        ...

    @abstractmethod
    def get_skill_status(self, robot: str, skill: str) -> SkillStatus:
        """Status of the skill ``robot`` was last asked to run."""
        ...

    @abstractmethod
    def stop_robot(self, robot: str) -> None:
        """Stop whatever ``robot`` is doing."""
        ...

    def get_pose(self, robot: str) -> Pose | None:
        """The robot's current pose, if known. Lets an interrupted or failed
        move leave the robot at a real ``{robot}_loc``."""
        return None

    def observe(
        self, robot: str, skill: str, *args: str
    ) -> Mapping[str, Collection[str]] | None:
        """What the robot perceived during a skill that just finished:
        location -> the objects seen there (a complete account of that
        location). Search outcomes follow it."""
        return None

    # --- for subclasses to call ---------------------------------------------- #

    def notify(self) -> None:
        """Wake the act() loop now -- e.g. from a ROS result callback --
        instead of on its next poll."""
        self._wakeup.set()

    # --- machinery ------------------------------------------------------------ #

    @property
    def realtime_factor(self) -> float:
        return self._realtime_factor

    def now(self) -> float:
        """The current time in planner units: real time since construction,
        scaled by ``realtime_factor``."""
        return self._start_time + (monotonic() - self._start_wall) * self._realtime_factor

    @property
    def state(self) -> State:
        state = super().state
        self._observed_state = state
        return state

    def _update_skills(self) -> None:
        # Real time keeps passing between act() calls (e.g. while planning).
        self._time = max(self._time, self.now())
        super()._update_skills()

    def _check_preconditions(self, action: Action) -> None:
        """Accept actions valid in the last state handed out.

        The robots keep moving while the planner thinks, so the chosen action
        may have gone stale (another robot finished something). Running it
        anyway is fine: if it fails on the robot, it is rolled back and the
        planner replans.
        """
        planned_from = self._observed_state
        if planned_from is not None and planned_from.satisfies_precondition(action):
            self._update_skills()
            return
        super()._check_preconditions(action)

    def _wait_until(self, time: float) -> float:
        """Sleep until ``time`` arrives, or until :meth:`notify` is called."""
        remaining = (time - self.now()) / self._realtime_factor
        if remaining > 0.0:
            self._wakeup.wait(remaining)
        # Cleared before skills poll: a later notify() is either seen by that
        # poll or wakes the next wait.
        self._wakeup.clear()
        return max(self.now(), self._time)

    def create_skill(self, action: Action, time: float) -> ActiveSkill:
        """Actions run on the robots, except :attr:`symbolic_skills` and
        actions that occupy no robot, which run as usual."""
        command = self.make_skill_command(action)
        if command is None or command.skill in self.symbolic_skills:
            return super().create_skill(action, time)
        for skill in self._active_skills:
            if (
                isinstance(skill, PhysicalSkill)
                and skill.command.robot == command.robot
                and not skill.phase.is_terminal
            ):
                raise RuntimeError(
                    f"{command.robot} is still running {skill.command.skill!r}; "
                    "a robot runs one skill at a time (operators should "
                    "remove `free ?robot` at t=0)."
                )
        return PhysicalSkill(
            action=action,
            start_time=time,
            command=command,
            robots=self,
            poll_period=self._poll_period,
        )

    def make_skill_command(self, action: Action) -> SkillCommand | None:
        """Describe an action as a command for a robot.

        The robot is the one whose ``free`` fluent the action adds (earliest,
        at positive time), and the planned duration is when it does so.
        Actions that free no robot later -- bookkeeping, instantaneous
        actions -- return None and run symbolically. ``move`` (and variants
        such as ``move-to-goal``) is motion; it is interruptible when
        ``skill_overrides`` maps it to an interruptible skill.
        """
        from .skill.navigation import InterruptibleNavigationMoveSkill

        parts = action.name.split()
        if not parts:
            return None
        frees = [
            (float(eff.time), fluent.args[0])
            for eff in action.effects
            if eff.time > 1e-9
            for fluent in eff.resulting_fluents
            if fluent.name == "free" and not fluent.negated and fluent.args
        ]
        if not frees:
            return None
        planned_duration, robot = min(frees)
        args = list(parts[1:])
        if robot in args:
            args.remove(robot)
        override = self._skill_overrides.get(parts[0])
        return SkillCommand(
            robot=robot,
            skill=parts[0],
            args=tuple(args),
            action=action,
            planned_duration=planned_duration,
            is_motion=parts[0].startswith("move"),
            interruptible=isinstance(override, type)
            and issubclass(override, InterruptibleNavigationMoveSkill),
        )

    def record_observations(
        self, command: SkillCommand, objects_at: Mapping[str, Collection[str]]
    ) -> None:
        """Perceived objects replace the believed contents of each location.

        Search resolution and revelation read the same table, so a search
        that completes after this succeeds or fails on what the robot saw.
        """
        del command
        for location, objects in objects_at.items():
            self._objects_at_locations[location] = set(objects)

    def on_skill_failed(self, skill: PhysicalSkill) -> None:
        """A robot reported FAILED: undo what the action started (e.g. an
        unfinished search stays unsearched and its location unlocked) and
        free the robot so the planner replans. A failed move leaves the robot
        at ``{robot}_loc``, its current pose -- none of the move's completion
        effects (arriving, ``visited``, ...) happened."""
        skill.roll_back_progress(self)
        self.fluents.add(Fluent("free", skill.command.robot))
        if skill.command.is_motion:
            skill.place_robot_at_robot_loc(self)
