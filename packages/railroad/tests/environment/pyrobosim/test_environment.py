"""PyRoboSimEnvironment: railroad actions on the PyRoboSim robots, end to end.

These run the real simulator on a wall clock, sped up by REALTIME_FACTOR, so
times are checked with tolerances.
"""

import logging

import numpy as np
import pytest
from pyrobosim.core.locations import ObjectSpawn

from railroad import operators
from railroad.core import Fluent as F, get_action_by_name
from railroad.environment import InterruptibleNavigationMoveSkill, PhysicalSkill, SkillStatus
from railroad.environment.pyrobosim import PyRoboSimEnvironment, PyRoboSimScene
from railroad.environment.skill.physical import Phase

REALTIME_FACTOR = 10.0
SKILL_TIME = 1.0
SKILL_DURATIONS = {"search": SKILL_TIME, "pick": SKILL_TIME, "place": SKILL_TIME}
# PyRoboSim's path follower settles for 0.1 s of wall time after each path.
SETTLE = 0.1 * REALTIME_FACTOR

logging.disable(logging.INFO)


class Env(PyRoboSimEnvironment):
    def define_operators(self):
        return [
            operators.construct_move_operator_blocking(self.scene.move_time),
            operators.construct_search_operator(0.5, SKILL_TIME),
            operators.construct_pick_operator_blocking(SKILL_TIME),
            operators.construct_place_operator_blocking(SKILL_TIME),
        ]


@pytest.fixture
def scene():
    # Some random layouts (e.g. seed 0's) leave too little room to put
    # anything down on counter0: fix a roomy one.
    return PyRoboSimScene(seed=1)


@pytest.fixture
def make_env(scene):
    envs = []

    def make(robots=("robot1",), **kwargs):
        # act() returns as soon as *any* robot is free, so tests about one
        # robot plan with that robot alone.
        env = Env(
            scene,
            state=scene.initial_state(robots),
            objects_by_type=scene.objects_by_type(robots),
            realtime_factor=REALTIME_FACTOR,
            skill_durations=SKILL_DURATIONS,
            **kwargs,
        )
        envs.append(env)
        return env

    yield make
    for env in envs:
        env.wait_idle(timeout=10.0)


def act(env, name):
    return env.act(get_action_by_name(env.get_actions(), name))


def sim_location(scene, robot):
    """The location (not spawn) the simulated robot is at, if any."""
    loc = scene.robots[robot].location
    return loc.parent.name if isinstance(loc, ObjectSpawn) else None


def test_scene_describes_world(scene):
    assert {"robot1", "robot2"} == set(scene.robots)
    assert {"table0", "my_desk", "counter0", "trash"} == set(scene.locations)
    assert scene.object_locations["my_desk"] == {"apple0", "water1", "soda"}
    assert "robot1_loc" in scene.location_registry
    t = scene.move_time("robot1", "robot1_loc", "my_desk")
    assert 3.0 < t < 10.0
    assert scene.move_time("robot1", "robot1_loc", "my_desk") == t  # cached
    assert scene.move_time("robot1", "robot1_loc", "nowhere") == float("inf")
    assert F("at robot1 robot1_loc") in scene.initial_state().fluents


def test_move_runs_in_simulator(make_env, scene):
    env = make_env()
    planned = scene.move_time("robot1", "robot1_loc", "my_desk")

    action = get_action_by_name(env.get_actions(), "move robot1 robot1_loc my_desk")
    start = env.state.time  # wall time passes while grounding, too
    state = env.act(action)

    assert F("at robot1 my_desk") in state.fluents
    assert F("free robot1") in state.fluents
    assert sim_location(scene, "robot1") == "my_desk"
    # Wall-clock jitter is magnified REALTIME_FACTOR times, so under load
    # (e.g. parallel tests) this is a sanity bound, not a precision check.
    assert planned - 0.5 <= state.time - start <= planned + SETTLE + 4.0


def test_search_uses_detections_not_ground_truth(make_env, scene):
    env = make_env()  # no prior belief about object locations
    act(env, "move robot1 robot1_loc my_desk")

    state = act(env, "search robot1 my_desk apple0")

    assert F("found apple0") in state.fluents
    assert F("at apple0 my_desk") in state.fluents
    assert F("revealed my_desk") in state.fluents
    # Revelation reports everything the robot saw there.
    assert F("at soda my_desk") in state.fluents
    assert F("at banana0 my_desk") not in state.fluents


def test_fetch_object_end_to_end(make_env, scene):
    env = make_env()
    for name in [
        "move robot1 robot1_loc table0",
        "search robot1 table0 banana0",
        "pick robot1 table0 banana0",
        "move robot1 table0 counter0",
    ]:
        state = act(env, name)
        assert F("free robot1") in state.fluents, name

    state = act(env, "place robot1 counter0 banana0")

    assert F("at banana0 counter0") in state.fluents
    banana = scene.world.get_object_by_name("banana0")
    assert banana.parent.parent.name == "counter0"
    assert scene.robots["robot1"].manipulated_object is None


def test_search_opens_a_closed_location(make_env, scene):
    env = make_env()
    assert not scene.world.get_location_by_name("trash").is_open
    act(env, "move robot1 robot1_loc trash")

    state = act(env, "search robot1 trash fuji")

    assert F("found fuji") in state.fluents
    assert scene.world.get_location_by_name("trash").is_open


def test_search_of_locked_location_fails_and_rolls_back(make_env, scene):
    trash = scene.world.get_location_by_name("trash")
    trash.is_locked = True  # closed and locked: nothing to see
    env = make_env()
    act(env, "move robot1 robot1_loc trash")

    state = act(env, "search robot1 trash fuji")

    assert F("free robot1") in state.fluents
    assert F("lock-search trash") not in state.fluents
    assert F("searched trash fuji") not in state.fluents
    assert F("found fuji") not in state.fluents
    assert F("at robot1 trash") in state.fluents


def test_two_robots_and_interrupted_move_stops_at_real_pose(make_env, scene):
    env = make_env(
        robots=("robot1", "robot2"), skill_overrides={"move": InterruptibleNavigationMoveSkill}
    )

    act(env, "move robot1 robot1_loc my_desk")  # ~5.5 s
    (move,) = [s for s in env._active_skills if isinstance(s, PhysicalSkill)]
    state = act(env, "move robot2 robot2_loc table0")  # ~1 s; frees robot2 first

    assert move.phase == Phase.CANCELED
    assert F("at robot2 table0") in state.fluents
    assert F("at robot1 robot1_loc") in state.fluents
    assert F("free robot1") in state.fluents
    assert not scene.robots["robot1"].is_busy()

    # robot1_loc now names where robot1 actually stopped: away from its
    # start, short of the desk.
    pose = scene.robots["robot1"].get_pose()
    np.testing.assert_allclose(scene.location_registry.get("robot1_loc"), [pose.x, pose.y])
    assert np.hypot(pose.x, pose.y) > 0.1
    assert sim_location(scene, "robot1") != "my_desk"

    # ...and the planner can continue from there. robot2 is free, so act()
    # returns at once; reading the state later catches up with the robot.
    state = act(env, "move robot1 robot1_loc my_desk")
    assert F("free robot1") not in state.fluents
    env.wait_idle(timeout=10.0)
    state = env.state
    assert F("at robot1 my_desk") in state.fluents
    assert F("free robot1") in state.fluents
    assert sim_location(scene, "robot1") == "my_desk"


def test_skill_failures_are_reported(make_env):
    env = make_env()
    assert "wait" in env.symbolic_skills
    with pytest.raises(ValueError):
        env.execute_skill("robot1", "close")  # not a PyRoboSim skill

    env.execute_skill("robot1", "pick", "my_desk", "apple0")  # robot1 is not at the desk
    env.wait_idle(timeout=10.0)

    assert env.get_skill_status("robot1", "pick") == SkillStatus.FAILED
    assert env._wakeup.is_set()  # the act() loop is woken, not left to poll
