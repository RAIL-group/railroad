"""Plan/act loop for converted PDDL problems.

Uses ``MCTSPlanner`` for action selection and the C++ ``transition`` function
for execution, sampling probabilistic outcomes with a seeded RNG. This
bypasses the environment layer: converted problems carry a pre-grounded action
set and need none of the skill machinery. Both paths now ground through
``railroad.core.ground_operators``, but ``Environment.get_actions`` pins
``allow_duplicate_bindings=False`` (the legacy all-distinct rule) while PDDL
semantics require ``True``.
"""

import random
import time as _time
from dataclasses import dataclass, field
from typing import List, Optional

from railroad.core import (
    State,
    get_action_by_name,
    transition,
)
from railroad.planner import MCTSPlanner, seed_planner_rng

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
    dead_end_penalty: Optional[float] = None,
    verbose: bool = False,
) -> RunResult:
    """Repeatedly plan and apply the chosen action until the goal holds.

    Replans with ``MCTSPlanner`` at every step, applying the chosen action to
    a sampled successor.

    ``dead_end_penalty`` is passed through to the planner: by default MCTS
    clamps an unreachable-goal state's heuristic to
    ``HEURISTIC_CANNOT_FIND_GOAL_PENALTY`` (0), which makes dead ends score
    *better* than reachable states, so domains that can strand the agent
    report failure even though the conversion is faithful. Setting a penalty
    that dominates typical plan costs (e.g. ``1e4``) makes those solvable —
    see the converter README's planner notes for what it costs elsewhere.
    """
    start = _time.perf_counter()
    actions = problem.ground_actions()
    if not actions:
        return RunResult(False, failure_reason="no grounded actions")
    # ``seed`` covers both the MCTS search and the outcome sampling below,
    # so runs are reproducible end-to-end.
    seed_planner_rng(seed)
    mcts = MCTSPlanner(actions, dead_end_penalty=dead_end_penalty)
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
