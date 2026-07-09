"""Multi-robot offline replay.

The replay arenas span every robot in the log: ``build_replay_env`` emits
per-robot ``at``/``free`` fluents and poses, and :func:`goal_fluent` is a
disjunction (*any* robot reaching the goal succeeds). Construction is checked
deterministically; the end-to-end drive uses the real MCTS selector (the greedy
``explore_first`` test helper is single-robot-oriented and strands robots that
pile onto one frontier — the production planner does not).
"""

from __future__ import annotations

import numpy as np

from railroad.core import Fluent
from railroad.lsp.frontier_statistics import FixedPriorFrontierStatistics
from railroad.replay import CandidatePolicy, replay
from railroad.replay.replay_env import build_replay_env, goal_fluent
from railroad.replay.types import RolloutLog, SubgoalRecord

from .conftest import REPLAY_TEST_CONFIG, parse_ascii_grid

# Goal reachable through free space; two unobserved pockets -> two frontiers.
# ``1``/``2`` mark the two robot starts, ``G`` the shared goal.
MAP_TWO_ROBOTS = """
##########
#1......G#
#..?..?..#
#2.......#
##########
"""


def _two_robot_log() -> RolloutLog:
    grid, markers = parse_ascii_grid(MAP_TWO_ROBOTS)
    s1, s2, goal = markers["1"][0], markers["2"][0], markers["G"][0]
    return RolloutLog(
        recorded_grid=grid,
        goal_cell=goal,
        robot_starts={
            "robot1": (float(s1[0]), float(s1[1]), 0.0),
            "robot2": (float(s2[0]), float(s2[1]), 0.0),
        },
        config=dict(REPLAY_TEST_CONFIG),
    )


def _estimator() -> FixedPriorFrontierStatistics:
    return FixedPriorFrontierStatistics(prob_feasible=0.8)


def test_arena_spans_all_robots() -> None:
    """Both robots get initial ``at``/``free`` fluents and are grounded objects."""
    log = _two_robot_log()
    assert log.robots == ["robot1", "robot2"]
    env = build_replay_env(log, _estimator())

    assert env.objects_by_type["robot"] == {"robot1", "robot2"}
    for robot in log.robots:
        assert Fluent(f"free {robot}") in env.state.fluents
        assert any(
            f.name == "at" and not f.negated and f.args[0] == robot
            for f in env.state.fluents
        ), f"{robot} must have an initial location fluent"


def test_goal_fluent_is_disjunction_over_robots() -> None:
    """The point-goal goal is satisfied by *any* robot reaching ``goal``."""
    goal = goal_fluent(["robot1", "robot2"])
    assert goal.evaluate({Fluent("at robot1 goal")})
    assert goal.evaluate({Fluent("at robot2 goal")})
    assert not goal.evaluate({Fluent("at robot1 start_loc")})


def test_multi_robot_replay_reaches_goal_with_sound_bounds() -> None:
    """The real planner drives a two-robot replay to the goal; bounds are sound.

    Invariants asserted (robust to MCTS stochasticity): the goal is reached by
    some robot, the bounds are finite, and the optimistic lower bound does not
    exceed the realized cost.
    """
    log = _two_robot_log()
    policy = CandidatePolicy(name="prior-0.8", frontier_statistics=_estimator())
    result = replay(log, policy, max_planning_iterations=200)

    assert result.goal_reached, "two robots must be able to reach the goal"
    assert result.termination == "goal_reached"
    assert np.isfinite(result.bounds.optimistic_lb)
    assert np.isfinite(result.bounds.simply_connected_lb)
    assert result.bounds.optimistic_lb <= result.total_cost + 1e-6


# Enclosed room (no frontiers), two robot starts (``1``/``2``), three containers;
# only ``E`` holds the target, so the search always resolves there.
SEARCH_ROOM_TWO_ROBOTS = """
##########
#1.......#
#..A...B.#
#..2.E...#
##########
"""


def test_multi_robot_object_search_finds_target() -> None:
    """A two-robot object-search replay resolves the target with sound bounds."""
    grid, markers = parse_ascii_grid(SEARCH_ROOM_TWO_ROBOTS)
    s1, s2 = markers["1"][0], markers["2"][0]

    def _container(name: str, cell, contents: tuple) -> SubgoalRecord:
        return SubgoalRecord(
            signature=name,
            centroid=(int(cell[0]), int(cell[1])),
            cells=np.array([[int(cell[0])], [int(cell[1])]], dtype=int),
            contents=contents,
            searched=True,
        )

    log = RolloutLog(
        recorded_grid=grid,
        goal_cell=(int(s1[0]), int(s1[1])),
        robot_starts={
            "robot1": (float(s1[0]), float(s1[1]), 0.0),
            "robot2": (float(s2[0]), float(s2[1]), 0.0),
        },
        problem_class="object-search",
        target_object="obj",
        subgoals=[
            _container("box_A", markers["A"][0], ()),
            _container("box_E", markers["E"][0], ("obj",)),
            _container("box_B", markers["B"][0], ()),
        ],
        config=dict(REPLAY_TEST_CONFIG),
    )
    assert log.robots == ["robot1", "robot2"]

    policy = CandidatePolicy(
        name="search",
        frontier_find_prob=lambda r, f, o: 0.5,
        container_find_prob=lambda r, l, o: 0.8,
    )
    result = replay(log, policy, max_planning_iterations=60)

    assert result.goal_reached
    assert any(loc == "box_E" and found for loc, _, found in result.search_log)
    assert result.bounds.optimistic_lb <= result.total_cost + 1e-6
