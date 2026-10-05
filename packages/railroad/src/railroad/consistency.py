"""One-step consistency of the planner's heuristic: a diagnostic, never used in planning.

MCTS values a leaf as elapsed time plus ``h`` and compares siblings by that
value. A heuristic that agrees with its own one-step lookahead satisfies

    h(s) = min_a [ extra_cost(a) + sum_o p_o * ((t_o - t_s) + h(o)) ]

over the actions MCTS branches on at ``s`` (those of the first free robot)
and the outcomes ``o`` of each transition, with ``h = 0`` at goal states and
dead ends valued as MCTS values them (by default ``h = 0``: an outcome from
which the goal is unreachable costs only the time spent reaching it). The
*gap* ``min_a Q(s, a) - h(s)`` measures how far one step of lookahead moves
the estimate:

- ``gap > 0``: ``h(s)`` promised more than any action delivers (optimistic);
- ``gap < 0``: some action does better than ``h(s)`` expected (pessimistic).

A gap on a *commit* step -- a robot starts an action while another is still
free, so no time passes -- is a jump MCTS sees directly between a node and
its child. ``check_consistency`` samples states by rollouts and reports the
gaps; ``record_consistency`` checks the root state of every planner call made
inside a block, so any example or benchmark can be checked unchanged.
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple, Union

from railroad._bindings import Action, Fluent, Goal, State, get_next_actions, transition
from railroad.planner import MCTSPlanner, _normalize_goal


@dataclass(frozen=True)
class Residual:
    """A state's heuristic value against its best one-step lookahead."""

    state: State
    h: float
    q: float  # min over actions of Q(s, a)
    action: str  # the action attaining it
    commit: bool  # that action leaves another robot free (no time passes)

    @property
    def gap(self) -> float:
        return self.q - self.h

    @property
    def rel_gap(self) -> float:
        return self.gap / max(self.h, 1.0)


def _value(planner: MCTSPlanner, state: State, goal: Goal) -> float:
    """A successor's cost-to-go as an MCTS leaf values it (multiplier 1)."""
    if goal.evaluate(state.fluents):
        return 0.0
    h = planner.heuristic(state, goal)
    if math.isfinite(h):
        return h
    # MCTS charges a dead end its flat penalty (in place of the time spent so
    # far, which this lookahead has already added) or, by default, nothing.
    return planner._dead_end_penalty if planner._dead_end_penalty is not None else 0.0


def lookahead(
    planner: MCTSPlanner, state: State, goal: Union[Goal, Fluent], actions: Sequence[Action]
) -> List[Tuple[Action, float, bool]]:
    """``(action, Q(s, a), commit)`` for every action MCTS branches on at ``state``."""
    goal = _normalize_goal(goal)
    out = []
    for action in get_next_actions(state, list(actions)):
        outcomes = transition(state, action)
        q = action.extra_cost
        for succ, prob in outcomes:
            q += prob * ((succ.time - state.time) + _value(planner, succ, goal))
        commit = len(outcomes) == 1 and outcomes[0][0].time == state.time
        out.append((action, q, commit))
    return out


def residual(
    planner: MCTSPlanner, state: State, goal: Union[Goal, Fluent], actions: Sequence[Action]
) -> Optional[Residual]:
    """The state's gap, or None at a goal, a dead end, or a state with no actions."""
    goal = _normalize_goal(goal)
    if goal.evaluate(state.fluents):
        return None
    h = planner.heuristic(state, goal)
    options = [o for o in lookahead(planner, state, goal, actions) if math.isfinite(o[1])]
    if not math.isfinite(h) or not options:
        return None
    action, q, commit = min(options, key=lambda o: o[1])
    return Residual(state, h, q, action.name, commit)


@dataclass
class ConsistencyReport:
    """Gaps over a set of states; ``tol`` is the relative gap counted as consistent."""

    residuals: List[Residual] = field(default_factory=list)
    tol: float = 0.01

    def summary(self, commit: Optional[bool] = None) -> dict:
        """Counts and gap statistics, over all states or only commit / non-commit ones."""
        rs = [r for r in self.residuals if commit is None or r.commit == commit]
        if not rs:
            return {"n": 0}
        rel = sorted(abs(r.rel_gap) for r in rs)
        return {
            "n": len(rs),
            "consistent": sum(x <= self.tol for x in rel) / len(rs),
            "optimistic": sum(r.rel_gap > self.tol for r in rs) / len(rs),
            "pessimistic": sum(r.rel_gap < -self.tol for r in rs) / len(rs),
            "mean_abs_rel_gap": sum(rel) / len(rs),
            "p90_abs_rel_gap": rel[min(len(rel) - 1, int(0.9 * len(rel)))],
            "max_abs_rel_gap": rel[-1],
        }

    def worst(self, n: int = 5) -> List[Residual]:
        return sorted(self.residuals, key=lambda r: -abs(r.rel_gap))[:n]

    def __str__(self) -> str:
        lines = [f"{'steps':<10}{'n':>6}{'consistent':>12}{'optimistic':>12}"
                 f"{'pessimistic':>13}{'mean |gap|':>12}{'p90 |gap|':>11}{'max |gap|':>11}"]
        for label, commit in (("all", None), ("commit", True), ("advance", False)):
            s = self.summary(commit)
            if not s["n"]:
                continue
            lines.append(
                f"{label:<10}{s['n']:>6}{s['consistent']:>12.0%}{s['optimistic']:>12.0%}"
                f"{s['pessimistic']:>13.0%}{s['mean_abs_rel_gap']:>12.1%}"
                f"{s['p90_abs_rel_gap']:>11.1%}{s['max_abs_rel_gap']:>11.1%}")
        lines.append("gaps are relative to h; consistent means |gap| <= "
                     f"{self.tol:.0%}; optimistic means lookahead is worse than h promised")
        return "\n".join(lines)


def check_consistency(
    planner: MCTSPlanner,
    state: State,
    goal: Union[Goal, Fluent],
    actions: Sequence[Action],
    *,
    rollouts: int = 10,
    depth: int = 20,
    epsilon: float = 0.3,
    seed: int = 0,
) -> ConsistencyReport:
    """Gaps along rollouts from ``state``.

    Each rollout follows the lookahead's best action, or with probability
    ``epsilon`` a random one, and samples the transition's outcome.
    """
    goal = _normalize_goal(goal)
    rng = random.Random(seed)
    report = ConsistencyReport()
    for _ in range(rollouts):
        s = state
        for _ in range(depth):
            if goal.evaluate(s.fluents):
                break
            options = [o for o in lookahead(planner, s, goal, actions) if math.isfinite(o[1])]
            h = planner.heuristic(s, goal)
            if not options or not math.isfinite(h):
                break
            best = min(options, key=lambda o: o[1])
            report.residuals.append(Residual(s, h, best[1], best[0].name, best[2]))
            action = rng.choice(options)[0] if rng.random() < epsilon else best[0]
            outcomes = transition(s, action)
            s = rng.choices([o[0] for o in outcomes], weights=[o[1] for o in outcomes])[0]
    return report


@contextmanager
def record_consistency(report: Optional[ConsistencyReport] = None) -> Iterator[ConsistencyReport]:
    """Check the root state of every ``MCTSPlanner`` call made inside the block.

    Planning itself is unchanged; each call costs one lookahead (a heuristic
    evaluation per outcome of each of the first free robot's actions).
    """
    report = report if report is not None else ConsistencyReport()
    original = MCTSPlanner.__call__

    def checked(self, state, goal, *args, **kwargs):
        r = residual(self, state, goal, self._original_actions)
        if r is not None:
            report.residuals.append(r)
        return original(self, state, goal, *args, **kwargs)

    MCTSPlanner.__call__ = checked
    try:
        yield report
    finally:
        MCTSPlanner.__call__ = original
