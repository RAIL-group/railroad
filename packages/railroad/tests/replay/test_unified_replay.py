"""Tests for the unified ``replay()`` entry + ``ReplayDomain`` dispatch.

One entry replays a :class:`RolloutLog` from any of the three domains, chosen by
``log.problem_class``. These tests assert (a) dispatch picks the right domain,
(b) the unified path is at parity with the legacy ``run_*`` drivers — exact for
the deterministic navigation and known-map selectors, and on the log-derived
optimistic bound for the MCTS-driven object-search domain — and (c) the policy
container + target handling behave.
"""

from __future__ import annotations

import dataclasses

import pytest

from railroad.lsp.frontier_statistics import FixedPriorFrontierStatistics
from railroad.replay import (
    CandidatePolicy,
    KnownMapSearchDomain,
    MctsParams,
    NavigationDomain,
    UnknownSearchDomain,
    get_domain,
    replay,
)
from railroad.replay.known_map_search_replay_env import (
    KnownMapSearchReplayEnvironment,
    build_known_map_search_log,
    run_known_map_search_replay,
)
from railroad.replay.replay_env import run_replay
from railroad.replay.search_replay_env import (
    SearchReplayEnvironment,
    run_search_replay,
)

from .conftest import build_log_from_ascii, explore_first_select, parse_ascii_grid
from .test_known_map_search_replay import (
    _coords,
    _drive,
    _env,
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


def _cfp(r, l, o) -> float:  # container-search probability
    return 0.8


# Same MCTS knobs the object-search domain defaults to (so the wrapper and the
# unified call plan with identical settings).
SEARCH_MCTS = MctsParams(iterations=1000, c=300.0, max_depth=20, heuristic_multiplier=2.0)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def test_get_domain_maps_problem_class_to_domain() -> None:
    assert isinstance(get_domain("navigation"), NavigationDomain)
    assert isinstance(get_domain("object-search"), UnknownSearchDomain)
    assert isinstance(get_domain("known-map-search"), KnownMapSearchDomain)


def test_unknown_problem_class_raises() -> None:
    log = dataclasses.replace(build_log_from_ascii(NAV_MAP), problem_class="frobnicate")
    with pytest.raises(ValueError, match="no replay domain"):
        replay(log, CandidatePolicy(), select_action=explore_first_select)


# --------------------------------------------------------------------------
# Navigation: exact parity (deterministic selector)
# --------------------------------------------------------------------------


def test_navigation_dispatch_matches_run_replay() -> None:
    log = build_log_from_ascii(NAV_MAP)
    legacy = run_replay(
        log, FixedPriorFrontierStatistics(prob_feasible=0.8),
        select_action=explore_first_select,
    )
    unified = replay(
        log,
        CandidatePolicy(frontier_statistics=FixedPriorFrontierStatistics(prob_feasible=0.8)),
        select_action=explore_first_select,
    )
    assert unified.termination == legacy.termination
    assert unified.total_cost == legacy.total_cost
    assert unified.bounds == legacy.bounds
    assert [c.frontier_signature for c in unified.commits] == [
        c.frontier_signature for c in legacy.commits
    ]
    # The frontier was committed, so the intercept + bound machinery ran.
    assert unified.commits


def test_navigation_policy_defaults_to_agnostic() -> None:
    """A bare replay(log) (no policy) still runs — neutral prior."""
    log = build_log_from_ascii(NAV_MAP)
    res = replay(log, select_action=explore_first_select)
    assert res.termination in {"goal_reached", "planner_none", "no_actions"}


# --------------------------------------------------------------------------
# Object search (unknown map): dispatch + log-derived optimistic-bound parity
# --------------------------------------------------------------------------


def _search_log():
    grid, markers = parse_ascii_grid(ROOM)
    start = markers["S"][0]
    return _log(grid, start, [_container_subgoal("box_1", markers["C"][0], ("obj",))])


def test_object_search_dispatch_runs_and_bounds_are_commit_based() -> None:
    log = _search_log()
    arena = SearchReplayEnvironment.from_log(log, target_object="obj")
    legacy = run_search_replay(
        arena, frontier_find_prob=_ffp, container_find_prob=_cfp,
        max_planning_iterations=40, mcts_iterations=SEARCH_MCTS.iterations,
    )
    unified = replay(
        log,
        CandidatePolicy(frontier_find_prob=_ffp, container_find_prob=_cfp),
        target_object="obj",
        max_planning_iterations=40,
        mcts=SEARCH_MCTS,
    )
    assert unified.goal_reached and legacy.goal_reached
    assert any(loc == "box_1" and found for loc, _, found in unified.search_log)
    # The optimistic bound is the commit-based min over the candidate's own
    # commits (trajectory-dependent under MCTS, so not equal across independent
    # runs) — assert the invariant on each run instead of cross-run equality.
    for res in (unified, legacy):
        _assert_commit_based_bound(res, log)


def test_object_search_without_target_raises() -> None:
    log = _search_log()  # object-search, log.target_object == ""
    with pytest.raises(ValueError, match="target object"):
        replay(log, CandidatePolicy(frontier_find_prob=_ffp, container_find_prob=_cfp))


# --------------------------------------------------------------------------
# Known-map search: exact parity (deterministic scripted selector)
# --------------------------------------------------------------------------


def _known_map_log(target="ring"):
    dep = _env(
        {"container_c": {target}}, target,
        prob=lambda r, l, o: 1.0 if l == "container_c" else 0.0,
    )
    _drive(dep, target)
    start = _coords()["start_loc"]
    return build_known_map_search_log(
        dep, robot_starts={"robot1": (float(start[0]), float(start[1]), 0.0)},
        target_object=target,
    )


def test_known_map_dispatch_matches_run() -> None:
    log = _known_map_log()
    # The recorder captured target_object, so the log is self-describing.
    assert log.problem_class == "known-map-search"
    assert log.target_object == "ring"

    legacy = run_known_map_search_replay(
        KnownMapSearchReplayEnvironment.from_log(log, target_object="ring"),
        container_find_prob=lambda r, l, o: 0.5,
        select_action=scripted_search,
    )
    unified = replay(
        log,
        CandidatePolicy(container_find_prob=lambda r, l, o: 0.5),
        select_action=scripted_search,
    )
    assert unified.goal_reached
    assert unified.bounds == legacy.bounds
    assert unified.total_cost == legacy.total_cost
    # This deployment's scripted_search searches EVERY container, so replay has no
    # unverified subgoal → no commits → exact replay, bounds collapse onto the
    # realized cost. (With an unsearched container it would not — see
    # test_known_map_search_replay for the commit case.)
    assert not unified.commits
    assert unified.bounds.optimistic_lb == unified.total_cost
    assert unified.bounds.simply_connected_lb == unified.total_cost


def test_known_map_target_from_log_no_override_needed() -> None:
    """Dispatch reads the target off the self-describing log (no target= arg)."""
    log = _known_map_log(target="ring")
    res = replay(
        log, CandidatePolicy(container_find_prob=lambda r, l, o: 0.5),
        select_action=scripted_search,
    )
    outcomes = {loc: found for loc, _, found in res.search_log}
    assert outcomes["container_c"] is True
