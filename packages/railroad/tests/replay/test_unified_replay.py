"""Tests for the build_replay_env dispatch + run_replay across all three flavors.

``build_replay_env`` picks the replay environment by ``log.problem_class`` and
returns a policy-agnostic arena; ``run_replay`` applies a candidate policy and
reduces the run to commit-based bounds. One recording replays many candidates by
building a fresh arena per policy. These tests assert (a) dispatch picks the right
env, (b) each flavor runs and its bounds are commit-based, and (c) the neutral
policy + target handling behave.
"""

from __future__ import annotations

import dataclasses

import pytest

from railroad.core import Fluent
from railroad.lsp.frontier_statistics import FixedPriorFrontierStatistics
from railroad.experimental.unknown_search import CallableObjectFind
from railroad.replay import (
    MctsConfig,
    build_replay_env,
    run_replay,
)

from .conftest import build_log_from_ascii, explore_first_select, parse_ascii_grid
from .test_known_map_search_replay import (
    _drive,
    _env,
    _record_log,
    scripted_search,
)
from .test_search_replay_env import (
    ROOM,
    _assert_commit_based_bound,
    _container_subgoal,
    _log,
)

# Reachable goal along row 1, one unobserved pocket -> one frontier.
NAV_MAP = """
##########
#S......G#
#...?....#
##########
"""


def _ffp(r, f, o) -> float:  # frontier-search probability
    return 0.5


def _cfp(r, loc, o) -> float:  # container-search probability
    return 0.8


# The MCTS knobs the unknown-search flavor plans with in these tests.
SEARCH_MCTS = MctsConfig(iterations=1000, c=300.0, max_depth=20, heuristic_multiplier=2.0)


def _search_log():
    grid, markers = parse_ascii_grid(ROOM)
    start = markers["S"][0]
    return _log(grid, start, [_container_subgoal("box_1", markers["C"][0], ("obj",))])


def _known_map_log(target="ring"):
    dep = _env(
        {"container_c": {target}}, target,
        prob=lambda r, loc, o: 1.0 if loc == "container_c" else 0.0,
    )
    _drive(dep, target)
    return _record_log(dep, target=target)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def test_build_replay_env_dispatches_by_problem_class() -> None:
    assert type(build_replay_env(build_log_from_ascii(NAV_MAP))).__name__ == (
        "ReplayPointGoalNavEnvironment"
    )
    assert type(build_replay_env(_search_log())).__name__ == (
        "ReplayUnknownSearchEnvironment"
    )
    assert type(build_replay_env(_known_map_log())).__name__ == (
        "ReplayKnownMapSearchEnvironment"
    )


def test_unknown_problem_class_raises() -> None:
    log = dataclasses.replace(build_log_from_ascii(NAV_MAP), problem_class="frobnicate")
    with pytest.raises(ValueError, match="no replay environment"):
        build_replay_env(log)


# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------


def test_navigation_replay_commits_and_is_sound() -> None:
    log = build_log_from_ascii(NAV_MAP)
    res = run_replay(
        build_replay_env(log),
        FixedPriorFrontierStatistics(prob_feasible=0.8),
        select_action=explore_first_select,
    )
    # The frontier was committed, so the intercept + bound machinery ran.
    assert res.commits
    assert res.bounds.optimistic_lb <= res.total_cost + 1e-6


def test_navigation_bare_policy_is_agnostic() -> None:
    """run_replay with no policy still runs — neutral prior."""
    log = build_log_from_ascii(NAV_MAP)
    res = run_replay(build_replay_env(log), select_action=explore_first_select)
    assert res.termination in {"goal_reached", "planner_none", "no_actions"}


# --------------------------------------------------------------------------
# Object search (unknown map)
# --------------------------------------------------------------------------


def test_object_search_runs_and_bounds_are_commit_based() -> None:
    log = _search_log()
    res = run_replay(
        build_replay_env(log),
        CallableObjectFind(_cfp),
        max_planning_iterations=40,
        mcts=SEARCH_MCTS,
    )
    assert res.goal_reached
    assert any(loc == "box_1" and found for loc, _, found in res.search_log)
    _assert_commit_based_bound(res, log)


def test_object_search_without_goal_raises() -> None:
    log = dataclasses.replace(_search_log(), goal=None)
    with pytest.raises(ValueError, match="needs a goal"):
        build_replay_env(log)


# --------------------------------------------------------------------------
# Known-map search
# --------------------------------------------------------------------------


def test_known_map_reads_goal_from_self_describing_log() -> None:
    log = _known_map_log(target="ring")
    # The recorder captured the goal, so no goal= arg is needed at replay.
    assert log.problem_class == "known-map-search"
    assert log.goal is not None
    assert log.goal.evaluate({Fluent("found ring")})

    res = run_replay(
        build_replay_env(log),
        CallableObjectFind(lambda r, loc, o: 0.5),
        select_action=scripted_search,
    )
    assert res.goal_reached
    outcomes = {loc: found for loc, _, found in res.search_log}
    assert outcomes["container_c"] is True
    # scripted_search searches EVERY container → no unverified subgoal → exact.
    assert not res.commits
    assert res.bounds.optimistic_lb == res.total_cost
    assert res.bounds.simply_connected_lb == res.total_cost
