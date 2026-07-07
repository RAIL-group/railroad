"""Plan/act loop for converted PDDL problems.

Uses ``MCTSPlanner`` for action selection and the C++ ``transition`` function
for execution, sampling probabilistic outcomes with a seeded RNG. This
deliberately bypasses ``SymbolicEnvironment``, whose action filters and skill
machinery assume railroad's robot/location domains.
"""

import random
import time as _time
from dataclasses import dataclass, field
from typing import List, Optional

from railroad.core import State, get_action_by_name, transition
from railroad.planner import MCTSPlanner

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
    verbose: bool = False,
) -> RunResult:
    """Repeatedly plan with MCTS and apply the chosen action until the goal holds."""
    start = _time.perf_counter()
    actions = problem.ground_actions()
    if not actions:
        return RunResult(False, failure_reason="no grounded actions")
    planner = MCTSPlanner(actions)
    rng = random.Random(seed)
    state = problem.initial_state
    plan: List[str] = []

    for _ in range(max_steps):
        if problem.goal.evaluate(state.fluents):
            return RunResult(
                True,
                plan=plan,
                sim_time=state.time,
                wall_time=_time.perf_counter() - start,
            )
        action_name = planner(
            state,
            problem.goal,
            max_iterations=max_iterations,
            max_depth=max_depth,
            c=c,
        )
        if action_name == "NONE":
            return RunResult(
                False,
                plan=plan,
                sim_time=state.time,
                wall_time=_time.perf_counter() - start,
                failure_reason="planner returned NONE",
            )
        if verbose:
            print(f"  t={state.time:8.3f}  {action_name}")
        action = get_action_by_name(actions, action_name)
        state = _apply(state, action, rng)
        plan.append(action_name)

    return RunResult(
        False,
        plan=plan,
        sim_time=state.time,
        wall_time=_time.perf_counter() - start,
        failure_reason=f"goal not reached within {max_steps} steps",
    )


def _apply(state: State, action, rng: random.Random) -> State:
    """Sample a successor from the transition distribution."""
    successors = transition(state, action)
    if not successors:
        raise RuntimeError(f"Action {action.name} produced no successors")
    weights = [prob for _, prob in successors]
    index = rng.choices(range(len(successors)), weights=weights, k=1)[0]
    return successors[index][0]
