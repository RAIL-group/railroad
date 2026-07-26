"""Tests for object-search replay (ReplayUnknownSearchEnvironment).

GL-free: containers + contents come from the log's subgoals; outcomes resolve
from that recorded ground truth. No panos needed (the served-vantage learned
path shares LearnedFrontierStatistics, covered by test_learned_replay). The arena
is built policy-agnostic with ``build_replay_env`` and a candidate applied by
``run_replay``.
"""

from __future__ import annotations

import numpy as np

from railroad.core import Fluent as F
from railroad.experimental.unknown_search import CallableObjectFind
from railroad.replay import (
    MctsConfig,
    build_replay_env,
    run_replay,
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


def _policy(container_prob=0.8) -> CallableObjectFind:
    return CallableObjectFind(lambda r, loc, o: container_prob)


def _search_mcts(iterations: int) -> MctsConfig:
    return MctsConfig(iterations=iterations, c=300.0, max_depth=20, heuristic_multiplier=2.0)


def _search_all_select(env, actions, goal) -> str:
    """Deterministic: search the current site, else move to an unsearched one.

    Removes MCTS stochasticity so the compound-goal test is reproducible.
    """
    applicable = [a for a in actions if env.state.satisfies_precondition(a)]
    searched = {
        f.args[0]
        for f in env.state.fluents
        if f.name == "searched" and not f.negated and f.args
    }

    def parts(a):
        return a.name.split()

    for a in sorted(applicable, key=lambda a: a.name):
        if parts(a)[0] == "search":
            return a.name
    moves = sorted(
        (a for a in applicable if parts(a)[0] == "move"
         and parts(a)[-1] not in searched and parts(a)[-1] != "start_loc"),
        key=lambda a: parts(a)[-1],
    )
    if moves:
        return moves[0].name
    any_move = sorted((a for a in applicable if parts(a)[0] == "move"), key=lambda a: a.name)
    return any_move[0].name if any_move else "NONE"


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


def _log(grid, start, subgoals, target="obj", goal=None) -> RolloutLog:
    return RolloutLog(
        recorded_grid=grid,
        goal_cell=(int(start[0]), int(start[1])),
        robot_starts={"robot1": (float(start[0]), float(start[1]), 0.0)},
        problem_class="object-search",
        goal=goal if goal is not None else F(f"found {target}"),
        subgoals=subgoals,
        config=dict(REPLAY_TEST_CONFIG),
    )


def test_search_replay_finds_object_at_true_container() -> None:
    grid, markers = parse_ascii_grid(ROOM)
    start = markers["S"][0]
    log = _log(grid, start, [_container_subgoal("box_1", markers["C"][0], ("obj",))])

    result = run_replay(
        build_replay_env(log),
        _policy(),
        max_planning_iterations=40,
        mcts=_search_mcts(1000),
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

    result = run_replay(
        build_replay_env(log),
        _policy(),
        max_planning_iterations=60,
        mcts=_search_mcts(1500),
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
    result = run_replay(
        build_replay_env(log),
        _policy(),
        max_planning_iterations=80,
        mcts=_search_mcts(2000),
    )
    assert result.goal_reached is True
    _assert_commit_based_bound(result, log)
    # The searched-empty container never commits, even if the candidate searches
    # it; only the unsearched one can.
    assert all(c.frontier_signature != "searched_empty" for c in result.commits)
    searched_by_candidate = {loc for loc, _, found in result.search_log if not found}
    if "unsearched" in searched_by_candidate:
        assert any(c.frontier_signature == "unsearched" for c in result.commits)


def test_compound_multi_object_goal() -> None:
    """The goal can be compound: two objects in two containers, goal =
    ``found apple & found book``. Both objects are searchable (derived from the
    goal's literals) and both must be found for the goal to be reached."""
    grid, markers = parse_ascii_grid(ROOM3)
    start = markers["S"][0]
    log = _log(
        grid,
        start,
        [
            _container_subgoal("box_a", markers["A"][0], ("apple",)),
            _container_subgoal("box_b", markers["B"][0], ("book",)),
        ],
        goal=F("found apple") & F("found book"),
    )
    env = build_replay_env(log)
    assert env.objects_by_type["object"] == {"apple", "book"}

    # Deterministic selector (no MCTS): search both boxes so the compound goal
    # (found apple & found book) is reliably reached.
    result = run_replay(env, _policy(), select_action=_search_all_select, max_planning_iterations=80)
    assert result.goal_reached is True
    found_locs = {loc for loc, _, ok in result.search_log if ok}
    assert "box_a" in found_locs and "box_b" in found_locs


def test_fresh_arena_per_candidate_from_one_log() -> None:
    """One recording replays many candidates: build a fresh policy-agnostic arena
    per candidate (run_replay mutates the arena it drives)."""
    grid, markers = parse_ascii_grid(ROOM)
    start = markers["S"][0]
    log = _log(grid, start, [_container_subgoal("box_1", markers["C"][0], ("obj",))])

    r1 = run_replay(
        build_replay_env(log), _policy(), max_planning_iterations=40,
        mcts=_search_mcts(800),
    )
    r2 = run_replay(
        build_replay_env(log), _policy(), max_planning_iterations=40,
        mcts=_search_mcts(800),
    )
    assert r1.goal_reached and r2.goal_reached
