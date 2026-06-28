"""Tests for object-search replay (SearchReplayEnvironment + from_log/run).

GL-free: containers + contents come from the log's subgoals; outcomes resolve
from that recorded ground truth. No panos needed (the served-vantage learned
path shares LearnedFrontierStatistics, covered by test_learned_replay).
"""

from __future__ import annotations

import math

import numpy as np

from railroad.replay.search_replay_env import (
    SearchReplayEnvironment,
    run_search_replay,
)
from railroad.replay.types import RolloutLog, SubgoalRecord

from .conftest import REPLAY_TEST_CONFIG, parse_ascii_grid

# Open room: the robot senses it, containers are revealed, it searches them.
ROOM = """
########
#S.....#
#..C...#
#....D.#
########
"""


def _container_subgoal(name: str, cell, contents) -> SubgoalRecord:
    return SubgoalRecord(
        signature=name,
        centroid=(int(cell[0]), int(cell[1])),
        cells=np.array([[int(cell[0])], [int(cell[1])]], dtype=int),
        contents=tuple(contents),
    )


def _log(grid, start, subgoals) -> RolloutLog:
    return RolloutLog(
        recorded_grid=grid,
        goal_cell=(int(start[0]), int(start[1])),
        robot_starts={"robot1": (float(start[0]), float(start[1]), 0.0)},
        problem_class="object-search",
        subgoals=subgoals,
        config=dict(REPLAY_TEST_CONFIG),
    )


def test_search_replay_finds_object_at_true_container() -> None:
    grid, markers = parse_ascii_grid(ROOM)
    start = markers["S"][0]
    log = _log(grid, start, [_container_subgoal("box_1", markers["C"][0], ("obj",))])

    arena = SearchReplayEnvironment.from_log(log, target_object="obj")
    result = run_search_replay(
        arena,
        frontier_find_prob=lambda r, f, o: 0.5,
        container_find_prob=lambda r, l, o: 0.8,
        max_planning_iterations=40,
        mcts_iterations=1000,
    )

    assert result.goal_reached is True
    assert any(loc == "box_1" and found for loc, _, found in result.search_log)
    assert math.isfinite(result.bounds.optimistic_lb)
    assert result.bounds.simply_connected_lb >= 0.0


def test_search_replay_resolves_empty_container_as_not_found() -> None:
    grid, markers = parse_ascii_grid(ROOM)
    start = markers["S"][0]
    log = _log(
        grid,
        start,
        [
            _container_subgoal("box_1", markers["C"][0], ()),       # empty
            _container_subgoal("box_2", markers["D"][0], ("obj",)),  # has it
        ],
    )

    arena = SearchReplayEnvironment.from_log(log, target_object="obj")
    result = run_search_replay(
        arena,
        frontier_find_prob=lambda r, f, o: 0.5,
        container_find_prob=lambda r, l, o: 0.8,
        max_planning_iterations=60,
        mcts_iterations=1500,
    )

    assert result.goal_reached is True
    found_by_loc = {loc: found for loc, _, found in result.search_log}
    assert found_by_loc.get("box_2") is True
    assert found_by_loc.get("box_1", False) is False


def test_arena_is_reusable_across_candidates() -> None:
    """from_log builds a policy-agnostic arena; run_search_replay is one-shot
    per call, so the same arena replays multiple candidates."""
    grid, markers = parse_ascii_grid(ROOM)
    start = markers["S"][0]
    log = _log(grid, start, [_container_subgoal("box_1", markers["C"][0], ("obj",))])
    arena = SearchReplayEnvironment.from_log(log, target_object="obj")

    r1 = run_search_replay(
        arena, frontier_find_prob=lambda r, f, o: 0.5,
        container_find_prob=lambda r, l, o: 0.8, max_planning_iterations=40,
        mcts_iterations=800,
    )
    r2 = run_search_replay(
        arena, frontier_find_prob=lambda r, f, o: 0.5,
        container_find_prob=lambda r, l, o: 0.8, max_planning_iterations=40,
        mcts_iterations=800,
    )
    assert r1.goal_reached and r2.goal_reached
