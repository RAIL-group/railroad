"""Action-set pruning for probabilistic achievers.

For each probabilistic fluent appearing in the goal, and for each robot
that can achieve it, keep only the top-K achievers by success probability
and the top-K by time-to-reach (deduplicated), discarding the rest.
Non-probabilistic actions and achievers of purely deterministic goal
fluents pass through untouched.

Per-robot grouping matters because a single robot can otherwise dominate
both ranking lists, leaving another robot with no surviving options for
the same fluent --- which would force the planner to route all
probabilistic attempts through one robot.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from railroad._bindings import (
    Action,
    Fluent,
    Goal,
    State,
    get_achievers_for_fluent,
    get_effective_goal_fluents,
)

_FF_TOLERANCE = 1e-9


def _robot_key(action: Action) -> Tuple[str, ...]:
    """Identify the robot(s) this action is executed by, for per-robot pruning.

    Uses the `free <robot>` precondition pattern. Every robot-executing
    operator in this codebase has `free ?r` as a precondition; after
    grounding, the arg is the concrete robot id. Actions bound to multiple
    robots (e.g. wait) return a sorted tuple of all of them. Actions with
    no free-robot precondition fall back to a single shared key so they
    don't inadvertently split into a group of one.
    """
    robots: List[str] = []
    for f in action.preconditions:
        if f.name == "free" and not f.negated:
            robots.extend(f.args)
    if not robots:
        return ("__no_robot__",)
    return tuple(sorted(robots))


def _collect_goal_fluents(
    state: State, goal: Goal, actions: List[Action]
) -> Set[Fluent]:
    """Return the set of fluents the pruner should consider as pruning targets.

    Delegates to the C++ `get_effective_goal_fluents`, which runs the same
    forward + per-branch `build_backward_extraction` pipeline used by the
    heuristic. The result is the union of effective goals across DNF
    branches, including the at-implies-found auto-landmarks (so `at <obj>
    <loc>` in the goal also pulls in `found <obj>`). Keeping this logic in
    one place (C++) avoids drift with the heuristic.
    """
    return set(get_effective_goal_fluents(state, goal, actions))


def prune_probabilistic_achievers(
    state: State,
    goal: Goal,
    actions: List[Action],
    top_k: int = 3,
) -> List[Action]:
    """Return a pruned copy of ``actions`` with redundant probabilistic achievers removed.

    For each probabilistic fluent ``f`` appearing in the goal, keeps the top
    ``top_k`` achievers by probability and the top ``top_k`` by time-to-reach
    (wait_cost + exec_cost), takes their union, and discards the remaining
    achievers of ``f``. Actions that are not achievers of any pruned fluent are
    preserved unchanged.

    Note: issues one `get_achievers_for_fluent` call per candidate fluent, each
    of which reruns the relaxed forward phase. Acceptable as a once-per-call
    preprocess; if it becomes a hotspot, add a companion C++ binding that
    returns all achievers from a single forward pass.
    """
    goal_fluents = _collect_goal_fluents(state, goal, actions)
    if not goal_fluents:
        return list(actions)

    # Name-to-action map, used to group achievers by robot during pruning.
    actions_by_name: Dict[str, Action] = {a.name: a for a in actions}

    prunable: Set[str] = set()
    keepers: Set[str] = set()

    for fluent in goal_fluents:
        achievers = get_achievers_for_fluent(state, fluent, actions)
        if not achievers:
            continue
        if all(prob >= 1.0 - _FF_TOLERANCE for _, _, _, prob in achievers):
            continue

        # Group achievers by robot. A robot group with <= top_k candidates is
        # small enough that pruning within it is a no-op for that robot, but
        # we still need to register its actions as "achievers of this fluent"
        # so other robots' pruning doesn't accidentally sweep them into the
        # prunable-but-not-kept bucket.
        by_robot: Dict[Tuple[str, ...], List] = {}
        for ach in achievers:
            name = ach[0]
            a = actions_by_name.get(name)
            key = _robot_key(a) if a is not None else ("__no_robot__",)
            by_robot.setdefault(key, []).append(ach)

        for robot_key, group in by_robot.items():
            # Record all group members as candidates (so the global keeper set
            # must contain them for them to survive).
            for name, _, _, _ in group:
                prunable.add(name)

            if len(group) <= top_k:
                # Too few to prune; keep them all.
                for name, _, _, _ in group:
                    keepers.add(name)
                continue

            by_prob = sorted(group, key=lambda a: (-a[3], a[1] + a[2]))[:top_k]
            by_time = sorted(group, key=lambda a: (a[1] + a[2], -a[3]))[:top_k]
            for name, _, _, _ in by_prob:
                keepers.add(name)
            for name, _, _, _ in by_time:
                keepers.add(name)

    if not prunable:
        return list(actions)

    return [a for a in actions if a.name not in prunable or a.name in keepers]
