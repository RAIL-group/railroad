"""Tests for PhysicalEnvironment: executing actions on robots (PhysicalSkill)."""

import math
import threading
import time
from abc import ABC
from dataclasses import dataclass
from typing import Callable, Collection, Dict, List, Mapping, Set, Tuple

import numpy as np
import pytest

from railroad import operators
from railroad._bindings import Fluent as F, State
from railroad.core import get_action_by_name
from railroad.environment import (
    InterruptibleNavigationMoveSkill,
    LocationRegistry,
    ObjectSearchEnvironment,
    PhysicalEnvironment,
    PhysicalSkill,
    Pose,
    SkillStatus,
)
from railroad.environment.skill.physical import Phase

from env_helpers import env_with_operators

_EPS = 1e-9

@dataclass
class _Run:
    skill: str
    args: Tuple[str, ...]
    start: float
    duration: float
    fail_at: float | None
    stopped_at: float | None = None


class SimulatedPhysicalEnvironment(PhysicalEnvironment, ABC):
    """A PhysicalEnvironment with scripted, in-process robots, for testing
    PhysicalEnvironment itself.

    Skills complete in simulated time -- waiting is instant, so tests are
    fast and exact -- and can be scripted to fail or to report observations.
    Like a robot calling :meth:`notify`, a finishing skill ends the act()
    loop's wait, so completion is seen at its exact time.

    Subclasses provide ``define_operators()``.

    Args:
        state, objects_by_type, operators: As for any environment.
        duration: ``duration(robot, skill, *args)``: how long a skill takes.
        fail_at: ``fail_at(robot, skill, *args)``: elapsed time at which the
            skill fails, or None if it succeeds.
        observe: ``observe(robot, skill, *args)``: what a finished skill saw
            (location -> objects), or None.
        symbolic_skills: Skills that run symbolically instead.
        **kwargs: Passed to :class:`PhysicalEnvironment` (``location_registry``
            is also used to interpolate ``move`` poses).
    """

    stale_status_grace = 0.0  # status is tracked per skill run, never stale

    def __init__(
        self,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        operators=None,
        *,
        duration: Callable[..., float],
        fail_at: Callable[..., float | None] | None = None,
        observe: Callable[..., Mapping[str, Collection[str]] | None] | None = None,
        symbolic_skills: Collection[str] = (),
        **kwargs,
    ) -> None:
        self._sim_now = state.time
        self._duration = duration
        self._fail_at = fail_at
        self._observe_fn = observe
        self.symbolic_skills = frozenset(symbolic_skills)
        self._runs: Dict[str, _Run] = {}
        self.started: List[Tuple[str, str, Tuple[str, ...]]] = []
        self.stopped: List[str] = []
        super().__init__(state, objects_by_type, operators, **kwargs)

    # --- simulated time ------------------------------------------------- #

    def now(self) -> float:
        return self._sim_now

    def _wait_until(self, time: float) -> float:
        # A robot reports its own completion (as if calling notify()): the
        # wait ends early when a running skill finishes or fails.
        ends = [
            end for end in map(self._end_time, self._runs.values())
            if end is not None and self._sim_now + _EPS < end < time
        ]
        self._sim_now = max(self._sim_now, min(ends, default=time))
        return self._sim_now

    @staticmethod
    def _end_time(run: _Run) -> float | None:
        if run.stopped_at is not None:
            return None
        if run.fail_at is not None:
            return run.start + min(run.fail_at, run.duration)
        return run.start + run.duration

    def advance_time(self, time: float) -> None:
        """Let simulated time pass outside act() -- e.g. while planning."""
        self._sim_now = max(self._sim_now, time)

    # --- skills ---------------------------------------------------------- #

    def execute_skill(self, robot: str, skill: str, *args: str) -> None:
        self._runs[robot] = _Run(
            skill=skill,
            args=args,
            start=self.now(),
            duration=self._duration(robot, skill, *args),
            fail_at=self._fail_at(robot, skill, *args) if self._fail_at else None,
        )
        self.started.append((robot, skill, args))

    # --- robot interface ---------------------------------------------------- #

    def get_skill_status(self, robot: str, skill: str) -> SkillStatus:
        run = self._runs[robot]
        if run.stopped_at is not None:
            return SkillStatus.FAILED
        elapsed = self.now() - run.start
        if run.fail_at is not None and elapsed >= run.fail_at - _EPS:
            return SkillStatus.FAILED
        if elapsed < run.duration - _EPS:
            return SkillStatus.RUNNING
        return SkillStatus.DONE

    def stop_robot(self, robot: str) -> None:
        run = self._runs.get(robot)
        if run is not None and run.stopped_at is None:
            run.stopped_at = self.now()
            self.stopped.append(robot)

    def observe(self, robot: str, skill: str, *args: str):
        return self._observe_fn(robot, skill, *args) if self._observe_fn else None

    def get_pose(self, robot: str) -> Pose | None:
        """Linear interpolation along a move, frozen where it stopped/failed."""
        run = self._runs.get(robot)
        registry = self.location_registry
        if run is None or registry is None or run.skill != "move" or len(run.args) < 2:
            return None
        start = registry.get(run.args[-2])
        end = registry.get(run.args[-1])
        if start is None or end is None:
            return None
        start = np.asarray(start, dtype=float).reshape(-1)
        end = np.asarray(end, dtype=float).reshape(-1)
        now = self.now()
        for limit in (run.stopped_at, None if run.fail_at is None else run.start + run.fail_at):
            if limit is not None:
                now = min(now, limit)
        frac = 1.0 if run.duration <= 0 else min(max((now - run.start) / run.duration, 0.0), 1.0)
        xy = start[:2] + frac * (end[:2] - start[:2])
        yaw = math.atan2(end[1] - start[1], end[0] - start[0])
        return Pose(x=float(xy[0]), y=float(xy[1]), yaw=yaw)


LOCATIONS = {
    "kitchen": np.array([0.0, 0.0]),
    "bedroom": np.array([10.0, 0.0]),
    "hall": np.array([0.0, 4.0]),
}
SEARCH_TIME = 3.0


def on_plan(registry, scale=1.0, scale_robots=None):
    """Skill durations matching the operators' (optionally scaled)."""
    move_time = registry.move_time_fn(1.0)

    def duration(robot, skill, *args):
        planned = move_time(robot, *args) if skill == "move" else SEARCH_TIME
        if scale_robots is None or robot in scale_robots:
            return planned * scale
        return planned

    return duration


def make_env(
    robot_names=("r1",),
    scale=1.0,
    scale_robots=None,
    true_object_locations=None,
    env_cls=SimulatedPhysicalEnvironment,
    move_op=None,
    **env_kwargs,
):
    """Object-search problem (move + search) with r1 in the kitchen and r2 in
    the hall. ``env_cls`` executes it: SimulatedPhysicalEnvironment (default),
    ObjectSearchEnvironment for the symbolic reference, or a test robot env.
    Returns (env, registry)."""
    registry = LocationRegistry(dict(LOCATIONS))
    starts = {"r1": "kitchen", "r2": "hall"}
    fluents = set()
    for robot in robot_names:
        fluents |= {F("at", robot, starts[robot]), F("free", robot)}
    if env_cls is SimulatedPhysicalEnvironment:
        env_kwargs.setdefault("duration", on_plan(registry, scale, scale_robots))
    env = env_with_operators(
        env_cls,
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": set(robot_names),
            "location": set(LOCATIONS),
            "object": {"Knife"},
        },
        operators=[
            move_op or operators.construct_move_operator_blocking(registry.move_time_fn(1.0)),
            operators.construct_search_operator(0.5, SEARCH_TIME),
        ],
        true_object_locations=(
            {"bedroom": {"Knife"}}
            if true_object_locations is None
            else true_object_locations
        ),
        location_registry=registry,
        seed=0,
        **env_kwargs,
    )
    return env, registry


def act(env, name):
    return env.act(get_action_by_name(env.get_actions(), name))


def test_matches_symbolic_execution_when_on_plan():
    """Robots that take exactly the planned time reproduce the symbolic run:
    same times, fluents and pending effects."""
    symbolic, _ = make_env(env_cls=ObjectSearchEnvironment)
    physical, _ = make_env()

    for name in ["move r1 kitchen bedroom", "search r1 bedroom Knife"]:
        expected = act(symbolic, name)
        actual = act(physical, name)
        assert actual.time == pytest.approx(expected.time)
        assert actual.fluents == expected.fluents
        assert [t for t, _ in actual.upcoming_effects] == pytest.approx(
            [t for t, _ in expected.upcoming_effects]
        )

    assert physical.started == [
        ("r1", "move", ("kitchen", "bedroom")),
        ("r1", "search", ("bedroom", "Knife")),
    ]
    assert F("found Knife") in physical.state.fluents


@pytest.mark.parametrize("scale", [0.5, 1.5])
def test_completion_follows_robot_not_plan(scale):
    env, _ = make_env(scale=scale)

    state = act(env, "move r1 kitchen bedroom")

    assert state.time == pytest.approx(10.0 * scale)
    assert F("at r1 bedroom") in state.fluents
    assert F("free r1") in state.fluents
    # Effects after completion keep their offset from the actual completion.
    assert [t for t, _ in state.upcoming_effects] == pytest.approx([10.0 * scale + 0.1])


def test_execute_skill_gets_action_arguments_and_starts_lazily():
    env, _ = make_env()
    action = get_action_by_name(env.get_actions(), "move r1 kitchen bedroom")

    skill = env.create_skill(action, env.time)

    assert isinstance(skill, PhysicalSkill)
    assert env.started == []  # constructing never moves a robot
    skill.advance(env.time, env)
    assert env.started == [("r1", "move", ("kitchen", "bedroom"))]
    assert skill.command.planned_duration == pytest.approx(10.0)
    assert skill.command.is_motion


def test_concurrent_robots_and_planner_view_of_late_robot():
    """Two robots run at once; the planner never sees a late robot's
    completion in the past."""
    env, _ = make_env(robot_names=("r1", "r2"), scale=2.0, scale_robots={"r1"})

    act(env, "move r1 kitchen bedroom")  # planned 10, takes 20
    state = act(env, "search r2 hall Knife")

    assert state.time == pytest.approx(SEARCH_TIME)
    assert F("free r1") not in state.fluents
    assert len(env.started) == 2

    def r1_free_times(state):
        return [t for t, eff in state.upcoming_effects if F("free r1") in eff.resulting_fluents]

    assert r1_free_times(state) == pytest.approx([10.0])  # as planned

    act(env, "move r2 hall kitchen")  # r2 frees at 7
    state = act(env, "search r2 kitchen Knife")  # r2 frees at 10, r1 still driving
    assert state.time == pytest.approx(10.0)
    assert all(t >= state.time for t in r1_free_times(state))

    state = act(env, "move r2 kitchen hall")  # r2 frees at 14
    assert F("free r1") not in state.fluents
    assert all(t >= state.time for t in r1_free_times(state))


def test_failed_move_stops_at_robot_loc():
    env, registry = make_env(fail_at=lambda robot, skill, *args: 4.0 if skill == "move" else None)

    state = act(env, "move r1 kitchen bedroom")

    assert state.time == pytest.approx(4.0)
    assert F("at r1 r1_loc") in state.fluents
    assert F("free r1") in state.fluents
    assert F("at r1 bedroom") not in state.fluents
    assert F("at r1 kitchen") not in state.fluents
    np.testing.assert_allclose(registry.get("r1_loc"), [4.0, 0.0])
    assert not state.upcoming_effects
    # The planner can move on from the stopping point.
    assert any(a.name.startswith("move r1 r1_loc") for a in env.get_actions())


def test_failed_move_applies_no_completion_effects():
    """A move that never arrives must not count as having arrived: here the
    operator also marks the destination ``visited`` on completion."""
    registry = LocationRegistry(dict(LOCATIONS))
    env, registry = make_env(
        move_op=operators.construct_move_visited_operator(registry.move_time_fn(1.0)),
        fail_at=lambda *args: 4.0,
    )

    state = act(env, "move r1 kitchen bedroom")

    assert F("visited bedroom") not in state.fluents
    assert F("at r1 r1_loc") in state.fluents
    assert F("at r1 kitchen") not in state.fluents
    assert F("free r1") in state.fluents
    coords = registry.get("r1_loc")
    assert coords is not None
    np.testing.assert_allclose(coords, [4.0, 0.0])


def test_failed_search_rolls_back_progress():
    env, _ = make_env(
        true_object_locations={"kitchen": {"Knife"}},
        fail_at=lambda robot, skill, *args: 1.0 if skill == "search" else None,
    )

    state = act(env, "search r1 kitchen Knife")

    assert state.time == pytest.approx(1.0)
    assert F("free r1") in state.fluents
    assert F("lock-search kitchen") not in state.fluents
    assert F("searched kitchen Knife") not in state.fluents
    assert F("found Knife") not in state.fluents
    # Nothing was learned, so searching again is still possible.
    assert "search r1 kitchen Knife" in {a.name for a in env.get_actions()}


@pytest.mark.parametrize(
    "seen, expect_found",
    [({"kitchen": {"Knife"}}, True), ({"kitchen": set()}, False)],
)
def test_search_outcome_follows_observations(seen, expect_found):
    """What the robot reports seeing overrides the environment's prior
    belief (here: Knife believed nowhere, or in the kitchen)."""
    believed = {} if expect_found else {"kitchen": {"Knife"}}
    env, _ = make_env(
        true_object_locations=believed,
        observe=lambda robot, skill, *args: seen if skill == "search" else None,
    )

    state = act(env, "search r1 kitchen Knife")

    assert (F("found Knife") in state.fluents) == expect_found
    assert (F("at Knife kitchen") in state.fluents) == expect_found
    assert F("revealed kitchen") in state.fluents


def test_skill_overrides_make_moves_interruptible():
    """The examples' switch for interruptible moves applies to robots too."""
    env, registry = make_env(
        robot_names=("r1", "r2"), skill_overrides={"move": InterruptibleNavigationMoveSkill}
    )

    act(env, "move r1 kitchen bedroom")
    state = act(env, "search r2 hall Knife")  # r2 frees at t=3 -> stop r1

    assert state.time == pytest.approx(SEARCH_TIME)
    assert env.stopped == ["r1"]
    assert F("at r1 r1_loc") in state.fluents
    assert F("free r1") in state.fluents
    np.testing.assert_allclose(registry.get("r1_loc"), [SEARCH_TIME, 0.0])


def test_moves_are_not_interrupted_by_default():
    env, _ = make_env(robot_names=("r1", "r2"))

    act(env, "move r1 kitchen bedroom")
    state = act(env, "search r2 hall Knife")

    assert env.stopped == []
    assert F("at r1 r1_loc") not in state.fluents
    state = act(env, "move r2 hall kitchen")  # r2 frees at t=7, r1 at t=10
    state = act(env, "search r2 kitchen Knife")
    assert state.time == pytest.approx(10.0)
    assert F("at r1 bedroom") in state.fluents


def test_symbolic_skills_run_symbolically():
    env, _ = make_env(symbolic_skills={"search"})

    act(env, "search r1 kitchen Knife")
    act(env, "move r1 kitchen bedroom")

    assert [skill for _, skill, _ in env.started] == ["move"]


class ScriptedEnvironment(SimulatedPhysicalEnvironment):
    """A physical environment whose status replies are scripted per poll."""

    def __init__(self, replies, raise_on_move=False, **kwargs):
        self.replies = list(replies)
        self.raise_on_move = raise_on_move
        super().__init__(duration=lambda *args: 0.0, **kwargs)

    def execute_skill(self, robot, skill, *args):
        if self.raise_on_move and skill == "move":
            raise ConnectionError("rosbridge down")

    def get_skill_status(self, robot, skill):
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    def stop_robot(self, robot):
        pass

    stale_status_grace = 0.5  # as for a real per-robot status service


class WallTimedEnvironment(PhysicalEnvironment):
    """Moves finish after ``seconds`` of real time, announced by notify()."""

    stale_status_grace = 0.0

    def __init__(self, seconds, **kwargs):
        self.seconds = seconds
        self.done = {}
        super().__init__(**kwargs)

    def execute_skill(self, robot, skill, *args):
        self.done[robot] = False

        def finish():
            self.done[robot] = True
            self.notify()

        threading.Timer(self.seconds, finish).start()

    def get_skill_status(self, robot, skill):
        return SkillStatus.DONE if self.done[robot] else SkillStatus.RUNNING

    def stop_robot(self, robot):
        pass


def test_physical_environments_run_in_real_time():
    env, _ = make_env(env_cls=WallTimedEnvironment, seconds=0.1, realtime_factor=200.0)
    before = env.now()
    time.sleep(0.05)
    assert env.now() - before == pytest.approx(0.05 * 200.0, rel=0.5)


def test_real_time_execution_wakes_on_notify():
    """A move taking 0.1 s of real time at 200x is ~20 planner units. The
    poll period is huge, so only notify() can end the wait promptly."""
    env, _ = make_env(
        env_cls=WallTimedEnvironment, seconds=0.1, realtime_factor=200.0, poll_period=1e6
    )
    start = time.monotonic()

    state = act(env, "move r1 kitchen bedroom")

    assert time.monotonic() - start < 1.0
    assert 15.0 < state.time < 60.0
    assert F("at r1 bedroom") in state.fluents
    assert F("free r1") in state.fluents


def test_stale_status_is_ignored_until_running_is_seen():
    """A per-robot status service may still say DONE from the previous move."""
    env, _ = make_env(
        env_cls=ScriptedEnvironment,
        replies=[SkillStatus.DONE, SkillStatus.RUNNING, SkillStatus.RUNNING, SkillStatus.DONE],
    )

    state = act(env, "move r1 kitchen bedroom")

    # Polls at t=0 (stale DONE, ignored), 1, 2 (RUNNING), 3 (DONE).
    assert state.time == pytest.approx(3.0)
    assert F("at r1 bedroom") in state.fluents


def test_idle_keeps_the_skill_waiting():
    """IDLE -- the robot's initial state, nothing assigned or running -- is
    not a failure: the command has not been picked up yet."""
    env, _ = make_env(
        env_cls=ScriptedEnvironment,
        replies=[SkillStatus.IDLE, SkillStatus.IDLE, SkillStatus.RUNNING, SkillStatus.DONE],
    )
    env.stale_status_grace = 0.0

    state = act(env, "move r1 kitchen bedroom")

    # Polls at t=0, 1 (IDLE), 2 (RUNNING), 3 (DONE).
    assert state.time == pytest.approx(3.0)
    assert F("at r1 bedroom") in state.fluents
    assert env.stopped == []


def test_without_grace_an_immediate_done_is_trusted():
    env, _ = make_env(env_cls=ScriptedEnvironment, replies=[SkillStatus.DONE])
    env.stale_status_grace = 0.0

    state = act(env, "move r1 kitchen bedroom")

    assert state.time == pytest.approx(0.0)
    assert F("at r1 bedroom") in state.fluents


def test_robot_errors_become_failures():
    env, _ = make_env(
        env_cls=ScriptedEnvironment, replies=[SkillStatus.RUNNING], raise_on_move=True,
    )

    state = act(env, "move r1 kitchen bedroom")

    assert F("free r1") in state.fluents
    assert F("at r1 bedroom") not in state.fluents


def test_act_accepts_actions_valid_when_planned():
    """The world moves while the planner thinks: an action valid in the state
    the planner saw still runs (and may fail on the robot)."""
    env, _ = make_env(robot_names=("r1", "r2"), true_object_locations={"kitchen": {"Knife"}})
    act(env, "search r1 kitchen Knife")  # r2 is free, so this returns at t=0

    # The loop: the planner reads env.state and picks an action from it...
    stale = get_action_by_name(env.get_actions(), "search r2 hall Knife")
    assert env.state.satisfies_precondition(stale)
    env.advance_time(5.0)  # ... while it thought, r1's search finished.

    env.act(stale)  # no error, although `not found Knife` no longer holds

    assert ("r2", "search", ("hall", "Knife")) in env.started
    assert F("found Knife") in env.state.fluents

    never_valid = get_action_by_name(env.get_actions(), "move r1 kitchen bedroom")
    env.fluents.discard(F("at r1 kitchen"))
    env.state  # the planner's view now excludes it, too
    with pytest.raises(ValueError):
        env.act(never_valid)


def test_symbolic_environments_keep_strict_preconditions():
    env, _ = make_env(env_cls=ObjectSearchEnvironment)
    action = get_action_by_name(env.get_actions(), "move r1 kitchen bedroom")
    env.state
    env.fluents.discard(F("at r1 kitchen"))
    with pytest.raises(ValueError):
        env.act(action)


def test_physical_skill_reports_phase():
    env, _ = make_env(robot_names=("r1", "r2"), scale=2.0)

    act(env, "move r1 kitchen bedroom")
    (skill,) = [s for s in env._active_skills if isinstance(s, PhysicalSkill)]
    assert skill.phase == Phase.RUNNING
    assert skill.is_motion_active_at(env.time)
    assert skill.controlled_robot == "r1"
