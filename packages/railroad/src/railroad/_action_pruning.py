"""Action-set pruning for probabilistic achievers.

In domains with many frontier-style actions (e.g. LSP point-goal
navigation, where every reachable frontier yields an ``lsp-explore``
action that probabilistically reveals the goal), the grounded action set
is too large for MCTS to branch over tractably. This module bounds it.

For each *probabilistic fluent on the relaxed path to the goal* and each
robot, we keep only the ``top_n`` achievers by success probability and
the ``cheapest_m`` achievers by time-to-execute (``wait_cost +
exec_cost`` -- reach the frontier plus explore it), take their union, and
discard the remaining achievers. This mirrors the LSP heuristic from the
original paper: prefer high-probability subgoals, but keep a few nearby
ones cheap enough to be worth a look.

Which fluents are probabilistic, which actions achieve them, and their
probability/cost all come from the FF heuristic's own forward + backward
extraction, exposed via ``get_probabilistic_path_achievers`` -- so the
pruner never drifts from what the heuristic actually plans over.

An optional second pass (``prune_orphaned_supports``) removes support
actions that achiever pruning *newly* orphaned -- e.g. a ``move`` whose
only destination frontier was just discarded. It diffs the relaxed
backward closure from the goal (``get_goal_relevant_action_names``)
before and after achiever pruning and drops only actions that were
goal-relevant before but are not after. Actions that were never on a path
to the goal in the relaxed graph (e.g. ``no_op`` / waiting, whose effects
only re-assert an already-true fluent) are left untouched, since the real
concurrent planner may still need them. In densely connected location
graphs (every frontier also a routable location) a discarded frontier can
remain a relevant waypoint, so this pass may prune few moves; the reliable
lever is the achiever pruning above.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from ._bindings import (
    get_goal_relevant_action_names,
    get_probabilistic_path_achievers,
)

if TYPE_CHECKING:
    from .core import Action, Goal, State

__all__ = ["prune_probabilistic_achievers"]


def _robot_key(action: Action) -> tuple[str, ...]:
    """Identify the robot(s) this action is executed by, for per-robot pruning.

    Uses the ``free <robot>`` precondition pattern: every robot-executing
    operator in this codebase has ``free ?r`` as a precondition, so after
    grounding the arg is the concrete robot id. Actions bound to multiple
    robots return a sorted tuple of all of them. Actions with no positive
    ``free`` precondition fall back to a single shared key so they don't
    inadvertently split into singleton groups.
    """
    robots: list[str] = []
    for precondition in action.preconditions:
        if precondition.name == "free" and not precondition.negated:
            robots.extend(precondition.args)
    return tuple(sorted(robots)) if robots else ("",)


def prune_probabilistic_achievers(
    state: State,
    goal: Goal,
    actions: list[Action],
    *,
    top_n: int = 4,
    cheapest_m: int = 2,
    prune_orphaned_supports: bool = True,
    frontier_objects: set[str] | None = None,
) -> list[Action]:
    """Return a pruned copy of ``actions``.

    For each probabilistic fluent on the relaxed path to ``goal``, and for
    each robot, keep the union of the ``top_n`` achievers by probability
    and the ``cheapest_m`` achievers by time-to-execute (``wait_cost +
    exec_cost``); discard the rest. Actions that are not achievers of any
    such fluent are preserved unchanged.

    If ``frontier_objects`` is given, any frontier with no surviving
    achiever and nothing located at it is treated as dead and *all* actions
    referencing it (e.g. moves routing to/through it) are dropped -- a
    frontier is an exploration-only object, so once it has no reason to be
    visited it cannot lie on a useful path. This is what cuts the move
    blowup in densely connected location graphs.

    If ``prune_orphaned_supports`` is set, a final pass drops any support
    action that achiever pruning *newly* orphaned (was on a relaxed path to
    the goal before pruning, but is not after).
    """
    # Probabilistic fluents still on the path to the goal, with their achievers.
    # When empty (e.g. the goal is already revealed, so no exploration remains
    # to reason about), there is nothing to rank -- but the dead-frontier pass
    # below still fires, dropping every now-purposeless frontier action.
    by_fluent = get_probabilistic_path_achievers(state, goal, actions)

    keep_names: set[str] = set()
    candidate_names: set[str] = set()
    name_to_action = {a.name: a for a in actions}

    for achievers in by_fluent.values():
        # Group this fluent's achievers by robot: (name, probability, attempt_cost).
        by_robot: dict[tuple[str, ...], list[tuple[str, float, float]]] = defaultdict(list)
        for name, probability, exec_cost, wait_cost in achievers:
            action = name_to_action.get(name)
            if action is None:
                continue
            by_robot[_robot_key(action)].append((name, probability, wait_cost + exec_cost))

        for group in by_robot.values():
            candidate_names.update(name for name, _, _ in group)
            most_probable = sorted(group, key=lambda item: item[1], reverse=True)[:top_n]
            cheapest = sorted(group, key=lambda item: item[2])[:cheapest_m]
            keep_names.update(name for name, _, _ in most_probable)
            keep_names.update(name for name, _, _ in cheapest)

    drop = candidate_names - keep_names

    if frontier_objects:
        # A frontier is alive if a surviving achiever still targets it or
        # something is currently located there (e.g. a robot mid-explore);
        # otherwise it is dead and every action mentioning it is removed.
        surviving_args: set[str] = set()
        for achievers in by_fluent.values():
            for name, *_ in achievers:
                if name not in drop:
                    surviving_args.update(name.split()[1:])
        located = {
            f.args[1]
            for f in state.fluents
            if f.name == "at" and not f.negated and len(f.args) >= 2
        }
        dead = {
            fr for fr in frontier_objects
            if fr not in surviving_args and fr not in located
        }
        if dead:
            for action in actions:
                if dead.intersection(action.name.split()[1:]):
                    drop.add(action.name)

    if prune_orphaned_supports and drop:
        # Drop only what achiever pruning *newly* orphaned: actions that were on
        # a relaxed path to the goal before pruning but no longer are. Diffing
        # the closure (rather than just intersecting with it) leaves always-
        # irrelevant actions such as no_op/wait — which only re-assert an
        # already-true fluent and never appear in the closure — untouched, since
        # the real concurrent planner may still need them.
        after = [a for a in actions if a.name not in drop]
        relevant_before = set(get_goal_relevant_action_names(state, goal, actions))
        relevant_after = set(get_goal_relevant_action_names(state, goal, after))
        drop = drop | (relevant_before - relevant_after)

    if not drop:
        return actions
    return [a for a in actions if a.name not in drop]
