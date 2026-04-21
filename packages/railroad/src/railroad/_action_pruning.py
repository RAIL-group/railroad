"""Action-set pruning for probabilistic achievers.

For each probabilistic fluent appearing in the goal, keep only the top-K
achievers by success probability and the top-K by time-to-reach (deduplicated),
discarding the rest. Non-probabilistic actions and achievers of purely
deterministic goal fluents pass through untouched.
"""
from __future__ import annotations

from typing import List, Set

from railroad._bindings import (
    Action,
    Fluent,
    Goal,
    State,
    get_achievers_for_fluent,
)

_FF_TOLERANCE = 1e-9


def _collect_goal_fluents(goal: Goal) -> Set[Fluent]:
    """Union of fluents across all DNF branches."""
    fluents: Set[Fluent] = set()
    for branch in goal.get_dnf_branches():
        fluents.update(branch)
    return fluents


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
    goal_fluents = _collect_goal_fluents(goal)
    if not goal_fluents:
        return list(actions)

    prunable: Set[str] = set()
    keepers: Set[str] = set()

    for fluent in goal_fluents:
        achievers = get_achievers_for_fluent(state, fluent, actions)
        if len(achievers) <= top_k:
            continue
        if all(prob >= 1.0 - _FF_TOLERANCE for _, _, _, prob in achievers):
            continue

        for name, _, _, _ in achievers:
            prunable.add(name)

        by_prob = sorted(
            achievers,
            key=lambda a: (-a[3], a[1] + a[2]),
        )[:top_k]
        by_time = sorted(
            achievers,
            key=lambda a: (a[1] + a[2], -a[3]),
        )[:top_k]

        for name, _, _, _ in by_prob:
            keepers.add(name)
        for name, _, _, _ in by_time:
            keepers.add(name)

    if not prunable:
        return list(actions)

    return [a for a in actions if a.name not in prunable or a.name in keepers]
