"""Plan/act loop for converted PDDL problems.

Uses ``MCTSPlanner`` or ``GreedyPlanner`` for action selection and the C++
``transition`` function for execution, sampling probabilistic outcomes with a
seeded RNG. This bypasses the environment layer: converted problems carry a
pre-grounded action set and need none of the skill machinery. Both paths now
ground through ``railroad.core.ground_operators``, but
``Environment.get_actions`` pins ``allow_duplicate_bindings=False`` (the
legacy all-distinct rule) while PDDL semantics require ``True``.
"""

import random
import time as _time
from dataclasses import dataclass, field
from typing import List, Optional

from railroad.core import (
    State,
    get_action_by_name,
    get_next_actions,
    transition,
)
from railroad.planner import GreedyPlanner, MCTSPlanner, seed_planner_rng

from .converter import ConvertedProblem


@dataclass
class RunResult:
    success: bool
    plan: List[str] = field(default_factory=list)
    # Simulated completion time; equals total cost under the cost->duration
    # mapping and plan length under the unit-duration mapping.
    sim_time: float = 0.0
    wall_time: float = 0.0
    failure_reason: Optional[str] = None


def solve(
    problem: ConvertedProblem,
    *,
    seed: int = 0,
    max_steps: int = 500,
    max_iterations: int = 4000,
    max_depth: int = 40,
    c: float = 100.0,
    planner: str = "mcts",
    verbose: bool = False,
) -> RunResult:
    """Repeatedly plan and apply the chosen action until the goal holds.

    ``planner`` selects the action-selection policy:

    - ``"mcts"``: full MCTS search per step (best plans, can wander on
      probabilistic domains with many degenerate actions).
    - ``"greedy"``: pick the applicable action minimizing the expected FF
      heuristic over its outcome distribution — a one-step-lookahead policy
      that is fast and surprisingly robust on IPPC-style domains.
    """
    start = _time.perf_counter()
    actions = problem.ground_actions()
    if not actions:
        return RunResult(False, failure_reason="no grounded actions")
    if planner not in ("mcts", "greedy"):
        raise ValueError(f"Unknown planner {planner!r}; use 'mcts' or 'greedy'")
    # ``seed`` covers both the MCTS search and the outcome sampling below,
    # so runs are reproducible end-to-end.
    seed_planner_rng(seed)
    mcts = MCTSPlanner(actions) if planner == "mcts" else None
    greedy = GreedyPlanner(actions) if planner == "greedy" else None
    rng = random.Random(seed)
    state = problem.initial_state
    plan: List[str] = []

    def finish(success: bool, reason: Optional[str] = None) -> RunResult:
        return RunResult(
            success,
            plan=plan,
            sim_time=state.time,
            wall_time=_time.perf_counter() - start,
            failure_reason=reason,
        )

    for _ in range(max_steps):
        if problem.goal.evaluate(state.fluents):
            return finish(True)
        if mcts is not None:
            action_name = mcts(
                state,
                problem.goal,
                max_iterations=max_iterations,
                max_depth=max_depth,
                c=c,
            )
            if action_name == "NONE":
                return finish(False, "planner returned NONE")
            action = get_action_by_name(actions, action_name)
        else:
            assert greedy is not None
            action = greedy.select_action(state, problem.goal)
            if action is None:
                if not get_next_actions(state, actions):
                    return finish(False, "no applicable action (dead end)")
                return finish(
                    False, "all applicable actions have infinite heuristic (dead end)"
                )
        if verbose:
            print(f"  t={state.time:8.3f}  {action.name}")
        state = _apply(state, action, rng)
        plan.append(action.name)

    if problem.goal.evaluate(state.fluents):
        return finish(True)
    return finish(False, f"goal not reached within {max_steps} steps")


def _apply(state: State, action, rng: random.Random) -> State:
    """Sample a successor from the transition distribution."""
    successors = transition(state, action)
    if not successors:
        raise RuntimeError(f"Action {action.name} produced no successors")
    weights = [prob for _, prob in successors]
    index = rng.choices(range(len(successors)), weights=weights, k=1)[0]
    return successors[index][0]
