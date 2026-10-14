"""PyRoboSimEnvironment: a PhysicalEnvironment on the PyRoboSim robots.

Time: PyRoboSim robots move in meters at ``linear_velocity`` m/s, so planner
time is seconds. ``realtime_factor`` speeds the simulation up (robots move,
and timed skills wait, ``realtime_factor`` times faster than real time) and
the environment's real time is scaled to match, so planner time stays in
simulated seconds.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Set, Tuple, Type

from pyrobosim.core.locations import ObjectSpawn
from pyrobosim.core.robot import Robot
from pyrobosim.planning.actions import ExecutionResult, ExecutionStatus
from pyrobosim.utils.pose import Pose as SimPose

from railroad._bindings import State
from ..environment import ActiveSkill
from ..physical import PhysicalEnvironment
from ..skill import SkillStatus
from ..types import Pose

from .canvas import MatplotlibWorldCanvas
from .scene import PyRoboSimScene


def _to_pose(pose: SimPose) -> Pose:
    return Pose(x=float(pose.x), y=float(pose.y), yaw=float(pose.get_yaw()))


@dataclass
class _Run:
    skill: str
    args: Tuple[str, ...]
    cancel: threading.Event = field(default_factory=threading.Event)
    status: SkillStatus = SkillStatus.RUNNING
    message: str = ""
    observations: Dict[str, Set[str]] | None = None
    thread: threading.Thread | None = None


class PyRoboSimEnvironment(PhysicalEnvironment):
    """Object-search environment whose actions run on the PyRoboSim robots.

    Subclasses provide ``define_operators()`` (``self.scene.move_time`` is the
    trajectory-based move duration), as in the example scripts. Search
    outcomes come from what the robots detect: the environment starts with no
    belief about where objects are unless ``prior_object_locations`` is given.

    :meth:`execute_skill` starts a worker thread for the skill and returns:

    - ``move (loc_from, loc_to)``: plan with the robot's path planner and
      follow the path; the pose is live, so an interrupted move stops where
      the robot actually is.
    - ``search (loc, obj)``: open the location if it is closed, detect
      objects there (fails if the robot is not there or it is locked), and
      report everything on its spawns through :meth:`observe`.
    - ``pick (loc, obj)`` / ``place (loc, obj)`` / ``no_op ()``.

    Any other skill fails, except ``wait``, which runs symbolically.

    PyRoboSim manipulation and detection are instantaneous, so non-move
    skills then wait out ``skill_durations[skill]`` -- pass the durations the
    operators use, so execution matches the plan.

    Args:
        scene: The PyRoboSim scene.
        state: Initial state; defaults to ``scene.initial_state()``.
        objects_by_type: Defaults to ``scene.objects_by_type()``.
        realtime_factor: Simulation speed-up; the environment's real time is
            scaled to match, so planner time stays in simulated seconds.
        skill_durations: Seconds (simulated) for non-move skills.
        skill_overrides: As for any environment; e.g.
            ``{"move": InterruptibleNavigationMoveSkill}`` makes moves stop
            whenever another robot frees.
        prior_object_locations: Initial belief about object locations, used
            until a search reports otherwise.
        show_plot: Show the simulator view live while acting.
        save_video: Record the simulator view; saved here by :meth:`close`.
        poll_period: Longest gap between status polls (simulated seconds).
            The simulator view redraws on each act() loop step, so the
            default is finer (0.1) while drawing than otherwise (0.5).

    Call :meth:`close` when done: it lets running skills finish, saves the
    simulator video, and keeps a shown window open until it is closed.
    """

    stale_status_grace = 0.0  # status is tracked per skill run, never stale
    symbolic_skills = frozenset({"wait"})

    def __init__(
        self,
        scene: PyRoboSimScene,
        state: State | None = None,
        objects_by_type: Dict[str, Set[str]] | None = None,
        *,
        realtime_factor: float = 1.0,
        skill_durations: Mapping[str, float] | None = None,
        skill_overrides: Dict[str, Type[ActiveSkill]] | None = None,
        prior_object_locations: Dict[str, Set[str]] | None = None,
        show_plot: bool = False,
        save_video: str | None = None,
        poll_period: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.scene = scene
        self._skill_durations = dict(skill_durations or {})
        self._runs: Dict[str, _Run] = {}
        drawing = show_plot or bool(save_video)
        self._save_video = save_video
        self.canvas = (
            MatplotlibWorldCanvas(scene.world, show_plot, record_video=bool(save_video))
            if drawing
            else None
        )
        if poll_period is None:
            poll_period = 0.1 if drawing else 0.5
        super().__init__(
            state=state if state is not None else scene.initial_state(),
            objects_by_type=(
                objects_by_type if objects_by_type is not None else scene.objects_by_type()
            ),
            true_object_locations=prior_object_locations,
            skill_overrides=skill_overrides,
            location_registry=scene.location_registry,
            seed=seed,
            realtime_factor=realtime_factor,
            poll_period=poll_period,
        )

    # --- skills ---------------------------------------------------------- #

    # --- robot interface -------------------------------------------------- #

    def execute_skill(self, robot: str, skill: str, *args: str) -> None:
        work = {
            "move": self._do_move,
            "search": self._do_search,
            "pick": self._do_pick,
            "place": self._do_place,
            "no_op": self._do_no_op,
        }.get(skill)
        if work is None:
            raise ValueError(f"PyRoboSim robots cannot run {skill!r}")
        self._launch(robot, skill, args, work)

    def get_skill_status(self, robot: str, skill: str) -> SkillStatus:
        return self._runs[robot].status

    def stop_robot(self, robot: str) -> None:
        run = self._runs.get(robot)
        if run is None:
            return
        run.cancel.set()
        sim_robot = self.scene.robots[robot]
        if sim_robot.is_busy():
            sim_robot.cancel_actions()  # blocks until the robot has stopped
        if run.thread is not None:
            run.thread.join(timeout=5.0)

    def get_pose(self, robot: str) -> Pose | None:
        sim_robot = self.scene.robots.get(robot)
        return None if sim_robot is None else _to_pose(sim_robot.get_pose())

    def observe(self, robot: str, skill: str, *args: str) -> Dict[str, Set[str]] | None:
        run = self._runs.get(robot)
        return run.observations if run is not None and run.skill == skill else None

    def wait_idle(self, timeout: float | None = None) -> None:
        """Join all skill threads (e.g. before tearing the world down)."""
        for run in list(self._runs.values()):
            if run.thread is not None:
                run.thread.join(timeout)

    def close(self) -> None:
        """Finish up: let running skills end, save the simulator video (if
        ``save_video`` was given), and keep a shown window open until closed."""
        self.wait_idle(timeout=5.0)
        if self.canvas is None:
            return
        if self._save_video:
            self.canvas.save_video(self._save_video)
        self.canvas.wait_for_close()

    def _after_skills_advanced(self, advanced_to_time: float) -> None:
        super()._after_skills_advanced(advanced_to_time)
        if self.canvas is not None:
            self.canvas.update()

    # --- worker side ------------------------------------------------------ #

    def _launch(
        self,
        robot: str,
        skill: str,
        args: Tuple[str, ...],
        work: Callable[[str, _Run], Tuple[SkillStatus, str]],
    ) -> None:
        run = _Run(skill=skill, args=args)
        self._runs[robot] = run

        def target() -> None:
            try:
                status, message = work(robot, run)
            except Exception as exc:  # report, never kill the act() loop
                status, message = SkillStatus.FAILED, f"{type(exc).__name__}: {exc}"
            run.message = message
            run.status = status
            self.notify()

        run.thread = threading.Thread(target=target, name=f"pyrobosim-{robot}-{skill}", daemon=True)
        run.thread.start()

    def _result(self, result: ExecutionResult) -> Tuple[SkillStatus, str]:
        status = SkillStatus.DONE if result.status == ExecutionStatus.SUCCESS else SkillStatus.FAILED
        return status, result.message or result.status.name

    def _navigate(self, robot: str, run: _Run, goal: SimPose) -> Tuple[SkillStatus, str]:
        sim_robot = self.scene.robots[robot]
        planner = sim_robot.path_planner
        if planner is None:
            return SkillStatus.FAILED, "robot has no path planner"
        with self.scene.planner_lock(robot):
            path = planner.plan(sim_robot.get_pose(), goal)
        if path is None or path.num_poses == 0:
            return SkillStatus.FAILED, "no path"
        if run.cancel.is_set():
            return SkillStatus.FAILED, "stopped before moving"
        return self._result(sim_robot.follow_path(path, realtime_factor=self.realtime_factor))

    def _wait_out(self, run: _Run, start: float) -> bool:
        """Sleep out the skill's duration; False if stopped meanwhile."""
        deadline = start + self._skill_durations.get(run.skill, 0.0) / self.realtime_factor
        return not run.cancel.wait(max(deadline - time.monotonic(), 0.0))

    def _spawn_at(self, sim_robot: Robot, location: str) -> ObjectSpawn | None:
        spawn = sim_robot.location
        if isinstance(spawn, ObjectSpawn) and spawn.parent is not None and spawn.parent.name == location:
            return spawn
        return None

    def _do_move(self, robot: str, run: _Run) -> Tuple[SkillStatus, str]:
        return self._navigate(robot, run, self.scene.nav_pose(robot, run.args[-1]))

    def _do_search(self, robot: str, run: _Run) -> Tuple[SkillStatus, str]:
        start = time.monotonic()
        location = run.args[0]
        sim_robot = self.scene.robots[robot]
        if self._spawn_at(sim_robot, location) is None:
            return SkillStatus.FAILED, f"{robot} is not at {location}"
        sim_location = self.scene.world.get_location_by_name(location)
        assert sim_location is not None
        if not sim_location.is_open:
            result = sim_robot.open_location()
            if not result.is_success():
                return self._result(result)
        result = sim_robot.detect_objects()
        if not result.is_success():
            return self._result(result)
        seen = {obj.name for spawn in sim_location.children for obj in spawn.children}
        if not self._wait_out(run, start):
            return SkillStatus.FAILED, "stopped"
        run.observations = {location: seen}
        return SkillStatus.DONE, ""

    def _do_pick(self, robot: str, run: _Run) -> Tuple[SkillStatus, str]:
        start = time.monotonic()
        location, obj_name = run.args[0], run.args[-1]
        sim_robot = self.scene.robots[robot]
        obj = self.scene.world.get_object_by_name(obj_name)
        if obj is None:
            return SkillStatus.FAILED, f"no object {obj_name}"
        if self._spawn_at(sim_robot, location) is None:
            return SkillStatus.FAILED, f"{robot} is not at {location}"
        # A location may have several spawns; step over to the object's.
        spawn = obj.parent
        if isinstance(spawn, ObjectSpawn) and spawn is not sim_robot.location:
            goal = self.scene._nav_pose_for_entity(sim_robot, spawn.name)
            if goal is None:
                return SkillStatus.FAILED, f"cannot reach {spawn.name}"
            status, message = self._navigate(robot, run, goal)
            if status != SkillStatus.DONE:
                return status, message
        status, message = self._result(sim_robot.pick_object(obj_name))
        if status != SkillStatus.DONE:
            return status, message
        if not self._wait_out(run, start):
            return SkillStatus.FAILED, "stopped"
        return SkillStatus.DONE, ""

    def _do_place(self, robot: str, run: _Run) -> Tuple[SkillStatus, str]:
        start = time.monotonic()
        location, obj_name = run.args[0], run.args[-1]
        sim_robot = self.scene.robots[robot]
        held = sim_robot.manipulated_object
        if held is None or held.name != obj_name:
            return SkillStatus.FAILED, f"{robot} is not holding {obj_name}"
        if self._spawn_at(sim_robot, location) is None:
            return SkillStatus.FAILED, f"{robot} is not at {location}"
        status, message = self._result(sim_robot.place_object())
        if status != SkillStatus.DONE:
            return status, message
        if not self._wait_out(run, start):
            return SkillStatus.FAILED, "stopped"
        return SkillStatus.DONE, ""

    def _do_no_op(self, robot: str, run: _Run) -> Tuple[SkillStatus, str]:
        if not self._wait_out(run, time.monotonic()):
            return SkillStatus.FAILED, "stopped"
        return SkillStatus.DONE, ""
