"""Tests for known-map object-search replay (design §7 / §7.1).

GL-free and procthor-free: a hand-drawn ASCII floorplan is the known map, the
container coordinates are markers, and outcomes resolve from recorded contents.
A scripted "search every container in order" policy keeps bounds exact.
"""

from __future__ import annotations

import math

import numpy as np

from railroad._bindings import Fluent as RF, State
from railroad.core import Fluent as F, get_action_by_name
from railroad.environment.symbolic import LocationRegistry
from railroad.replay.known_map_search_replay_env import (
    KnownMapSearchReplayEnvironment,
    build_known_map_search_log,
    run_known_map_search_replay,
)

from .conftest import parse_ascii_grid

# S start; A/B/C searchable containers in a single open room (all reachable).
MAP = """
##########
#S..A...B#
#........#
#...C....#
##########
"""


def _coords():
    _, markers = parse_ascii_grid(MAP)
    return {
        "start_loc": markers["S"][0],
        "container_a": markers["A"][0],
        "container_b": markers["B"][0],
        "container_c": markers["C"][0],
    }


def _grid():
    grid, _ = parse_ascii_grid(MAP)
    return grid


def _env(contents, target, prob=lambda r, l, o: 0.5):
    coords = _coords()
    fluents = {F("revealed start_loc"), F("at robot1 start_loc"), F("free robot1")}
    return KnownMapSearchReplayEnvironment(
        known_grid=_grid(),
        recorded_object_locations=contents,
        container_find_prob=prob,
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": {"robot1"},
            "location": set(coords),
            "object": {target},
        },
        location_registry=LocationRegistry(
            {k: np.array(v, dtype=float) for k, v in coords.items()}
        ),
    )


def scripted_search(env, actions, goal) -> str:
    """Deterministic: search the current container, else move to the lowest-named
    not-yet-searched container."""
    applicable = [a for a in actions if env.state.satisfies_precondition(a)]
    searched = {
        f.args[0] for f in env.state.fluents if f.name == "searched" and f.args
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
    any_move = sorted((a for a in applicable if parts(a)[0] == "move"),
                      key=lambda a: a.name)
    return any_move[0].name if any_move else "NONE"


def _drive(env, target, max_iter=30, select=None) -> None:
    select = select or scripted_search
    goal = F(f"found {target}")
    for _ in range(max_iter):
        if goal.evaluate(env.state.fluents):
            return
        actions = env.get_actions()
        if not actions:
            return
        name = select(env, actions, goal)
        if name == "NONE":
            return
        env.act(get_action_by_name(actions, name))


def _beeline_search(container):
    """A deployment selector that goes straight to *container* and searches only
    it — leaving the other containers revealed but UNsearched."""
    def select(env, actions, goal) -> str:
        app = [a for a in actions if env.state.satisfies_precondition(a)]
        for a in app:
            p = a.name.split()
            if p[0] == "search" and len(p) >= 3 and p[2] == container:
                return a.name
        for a in sorted(app, key=lambda a: a.name):
            p = a.name.split()
            if p[0] == "move" and p[-1] == container:
                return a.name
        return "NONE"

    return select


# --------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------


def test_recorder_captures_searched_contents_only() -> None:
    """The recorder records every searchable container's coords, but contents
    only for the ones actually inspected (uninspected -> empty)."""
    target = "ring"
    # Deployment 'truth': ring truly in container_c; the deployment (informed)
    # searches only container_c.
    dep = _env({"container_c": {target}}, target,
               prob=lambda r, l, o: 1.0 if l == "container_c" else 0.0)
    _drive(dep, target)
    assert F(f"found {target}").evaluate(dep.state.fluents)

    log = build_known_map_search_log(
        dep, robot_starts={"robot1": (float(_coords()["start_loc"][0]),
                                      float(_coords()["start_loc"][1]), 0.0)},
    )
    assert log.problem_class == "known-map-search"
    assert log.actual_total_cost > 0
    by_sig = {s.signature: s for s in log.subgoals}
    assert set(by_sig) == {"container_a", "container_b", "container_c"}
    # Only the searched true container carries contents.
    assert target in by_sig["container_c"].contents
    assert by_sig["container_a"].contents == ()
    assert by_sig["container_b"].contents == ()


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------


def test_replay_finds_target_from_recorded_truth() -> None:
    target = "ring"
    dep = _env({"container_c": {target}}, target,
               prob=lambda r, l, o: 1.0 if l == "container_c" else 0.0)
    _drive(dep, target)
    log = build_known_map_search_log(
        dep, robot_starts={"robot1": (float(_coords()["start_loc"][0]),
                                      float(_coords()["start_loc"][1]), 0.0)},
    )

    arena = KnownMapSearchReplayEnvironment.from_log(log, target_object=target)
    res = run_known_map_search_replay(
        arena, container_find_prob=lambda r, l, o: 0.5,
        select_action=scripted_search,
    )
    assert res.goal_reached
    # A wrong container the deployment never searched resolves not-found...
    outcomes = {loc: found for loc, _, found in res.search_log}
    assert outcomes["container_a"] is False
    assert outcomes["container_b"] is False
    # ...and the true container resolves found from the recording.
    assert outcomes["container_c"] is True


def test_replay_bounds_are_seconds_and_admissible() -> None:
    target = "ring"
    dep = _env({"container_c": {target}}, target,
               prob=lambda r, l, o: 1.0 if l == "container_c" else 0.0)
    _drive(dep, target)
    log = build_known_map_search_log(
        dep, robot_starts={"robot1": (float(_coords()["start_loc"][0]),
                                      float(_coords()["start_loc"][1]), 0.0)},
    )
    arena = KnownMapSearchReplayEnvironment.from_log(log, target_object=target)
    res = run_known_map_search_replay(
        arena, container_find_prob=lambda r, l, o: 0.5,
        select_action=scripted_search,
    )
    # This deployment (scripted_search) searched EVERY container, so replay has
    # no unverified subgoal → no commits → exact replay: both bounds collapse
    # onto the realized cost. (A partial-search deployment does not — see
    # test_replay_logs_commits_for_unsearched_containers below.)
    assert not res.commits
    assert math.isfinite(res.bounds.optimistic_lb)
    assert res.bounds.optimistic_lb == res.total_cost
    assert res.bounds.simply_connected_lb == res.total_cost
    assert res.total_cost > 0


def test_replay_logs_commits_for_unsearched_containers() -> None:
    """Dropping the one-container-per-object assumption: when the deployment
    searched only the true container, the others are revealed-but-unsearched, so
    a candidate searching them commits (optimistic_to_goal=0). The bound is then a
    real lower bound below the makespan, not the collapsed exact cost."""
    target = "ring"
    dep = _env({"container_c": {target}}, target,
               prob=lambda r, l, o: 1.0 if l == "container_c" else 0.0)
    _drive(dep, target, select=_beeline_search("container_c"))
    dep_searched = {f.args[0] for f in dep.state.fluents
                    if f.name == "searched" and not f.negated and f.args}
    assert dep_searched == {"container_c"}, "deployment should search only the true container"

    log = build_known_map_search_log(
        dep, robot_starts={"robot1": (float(_coords()["start_loc"][0]),
                                      float(_coords()["start_loc"][1]), 0.0)},
        target_object=target,
    )
    by_sig = {s.signature: s for s in log.subgoals}
    assert by_sig["container_a"].searched is False and by_sig["container_a"].contents == ()
    assert by_sig["container_c"].searched is True and target in by_sig["container_c"].contents

    arena = KnownMapSearchReplayEnvironment.from_log(log, target_object=target)
    res = run_known_map_search_replay(
        arena, container_find_prob=lambda r, l, o: 0.5, select_action=scripted_search,
    )
    assert res.goal_reached
    commit_sigs = {c.frontier_signature for c in res.commits}
    # Unsearched containers the candidate searched → commits; the searched true
    # container (found) → no commit.
    assert "container_a" in commit_sigs and "container_b" in commit_sigs
    assert "container_c" not in commit_sigs
    # A genuine optimism gap now: opt < makespan, and (optimistic_to_goal=0) opt is
    # the earliest unsearched-container search time.
    assert res.bounds.optimistic_lb < res.total_cost
    assert res.bounds.optimistic_lb == min(c.cost_accrued for c in res.commits)


def test_replay_is_deterministic() -> None:
    target = "ring"
    dep = _env({"container_c": {target}}, target,
               prob=lambda r, l, o: 1.0 if l == "container_c" else 0.0)
    _drive(dep, target)
    log = build_known_map_search_log(
        dep, robot_starts={"robot1": (float(_coords()["start_loc"][0]),
                                      float(_coords()["start_loc"][1]), 0.0)},
    )
    arena = KnownMapSearchReplayEnvironment.from_log(log, target_object=target)

    def go():
        return run_known_map_search_replay(
            arena, container_find_prob=lambda r, l, o: 0.5,
            select_action=scripted_search,
        )

    a, b = go(), go()
    assert a.bounds == b.bounds
    assert a.total_cost == b.total_cost
    assert [s[0] for s in a.search_log] == [s[0] for s in b.search_log]
