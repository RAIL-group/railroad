"""Tests for object-search replay (SearchReplayEnvironment + from_log/run).

GL-free: containers + contents come from the log's subgoals; outcomes resolve
from that recorded ground truth. No panos needed (the served-vantage learned
path shares LearnedFrontierStatistics, covered by test_learned_replay).
"""

from __future__ import annotations

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

# Three containers (A, B, E) in one open room: enough to contrast a
# searched-empty container (no commit) against an unsearched one (commit).
ROOM3 = """
##########
#S.......#
#..A...B.#
#....E...#
##########
"""


def _container_subgoal(name: str, cell, contents, searched: bool = True) -> SubgoalRecord:
    # A container with recorded contents was searched by the deployment; an
    # unsearched container carries no contents (searched=False).
    return SubgoalRecord(
        signature=name,
        centroid=(int(cell[0]), int(cell[1])),
        cells=np.array([[int(cell[0])], [int(cell[1])]], dtype=int),
        contents=tuple(contents),
        searched=searched,
    )


def _assert_commit_based_bound(result, log) -> None:
    """optimistic_lb is the commit-based bound, and commits are logged only at
    subgoals the deployment did NOT search (searched ones replay exactly)."""
    searched = {s.signature for s in log.subgoals if s.searched}
    for c in result.commits:
        assert c.frontier_signature not in searched, (
            f"commit logged at searched subgoal {c.frontier_signature}"
        )
    if result.commits:
        expected = min(c.cost_accrued + c.optimistic_to_goal for c in result.commits)
    else:
        expected = result.total_cost  # exact replay when nothing was unverified
    assert result.bounds.optimistic_lb == expected
    assert result.bounds.optimistic_lb <= result.total_cost + 1e-6
    assert result.bounds.simply_connected_lb == result.total_cost


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
    # box_1 was searched by the deployment and holds the object → found exactly,
    # no commit; the enclosed room has no frontiers → no commits → exact replay.
    _assert_commit_based_bound(result, log)


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


def test_commit_only_at_unsearched_containers() -> None:
    """A searched-empty container (deployment verified it empty) resolves
    not-found with NO optimistic commit; a revealed-but-unsearched container is
    an unverified subgoal → not-found + a commit. We do not assume one container
    per object, so finding the object in E does not exempt the unsearched box."""
    grid, markers = parse_ascii_grid(ROOM3)
    start = markers["S"][0]
    log = _log(
        grid,
        start,
        [
            _container_subgoal("searched_empty", markers["A"][0], (), searched=True),
            _container_subgoal("unsearched", markers["B"][0], (), searched=False),
            _container_subgoal("target_box", markers["E"][0], ("obj",), searched=True),
        ],
    )
    arena = SearchReplayEnvironment.from_log(log, target_object="obj")
    result = run_search_replay(
        arena,
        frontier_find_prob=lambda r, f, o: 0.5,
        container_find_prob=lambda r, l, o: 0.8,
        max_planning_iterations=80,
        mcts_iterations=2000,
    )
    assert result.goal_reached is True
    _assert_commit_based_bound(result, log)
    # The searched-empty container never commits, even if the candidate searches
    # it; only the unsearched one can.
    assert all(c.frontier_signature != "searched_empty" for c in result.commits)
    searched_by_candidate = {loc for loc, _, found in result.search_log if not found}
    if "unsearched" in searched_by_candidate:
        assert any(c.frontier_signature == "unsearched" for c in result.commits)


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
