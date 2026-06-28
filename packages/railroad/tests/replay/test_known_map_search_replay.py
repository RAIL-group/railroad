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


def _drive(env, target, max_iter=30) -> None:
    goal = F(f"found {target}")
    for _ in range(max_iter):
        if goal.evaluate(env.state.fluents):
            return
        actions = env.get_actions()
        if not actions:
            return
        name = scripted_search(env, actions, goal)
        if name == "NONE":
            return
        env.act(get_action_by_name(actions, name))


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
    # Optimal (straight to the true container) <= the naive exact replay cost.
    assert math.isfinite(res.bounds.optimistic_lb)
    assert res.bounds.optimistic_lb <= res.total_cost + 1e-6
    assert res.bounds.simply_connected_lb == res.total_cost
    assert res.total_cost > 0


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
