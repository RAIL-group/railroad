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

from railroad.core import (
    State,
    convert_action_effects,
    convert_action_to_positive_preconditions,
    convert_goal_to_positive_preconditions,
    convert_state_to_positive_preconditions,
    create_positive_fluent_mapping,
    extract_all_negative_fluents,
    ff_heuristic,
    get_action_by_name,
    get_next_actions,
    transition,
)
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
    mcts = MCTSPlanner(actions) if planner == "mcts" else None
    heuristic = (
        _GreedyHeuristic(actions, problem.goal) if planner == "greedy" else None
    )
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
            assert heuristic is not None
            action = _greedy_action(state, heuristic, actions)
            if action is None:
                return finish(False, "no applicable action (dead end)")
        if verbose:
            print(f"  t={state.time:8.3f}  {action.name}")
        state = _apply(state, action, rng)
        plan.append(action.name)

    return finish(False, f"goal not reached within {max_steps} steps")


class _GreedyHeuristic:
    """FF heuristic evaluator that handles negative preconditions and goals.

    ``ff_heuristic`` reasons over positive literals only, so negated fluents
    (in goals like ``(not (broken vase))`` or in preconditions) must first be
    rewritten to positive ``not-*`` bookkeeping fluents — the same
    preprocessing ``MCTSPlanner`` applies internally.
    """

    def __init__(self, actions, goal):
        negatives = extract_all_negative_fluents(actions, goal)
        if negatives:
            self._mapping = create_positive_fluent_mapping(negatives)
            self._actions = [
                convert_action_effects(
                    convert_action_to_positive_preconditions(a, self._mapping),
                    self._mapping,
                )
                for a in actions
            ]
            self._goal = convert_goal_to_positive_preconditions(goal, self._mapping)
        else:
            self._mapping = None
            self._actions = actions
            self._goal = goal

    def __call__(self, state: State) -> float:
        if self._mapping is not None:
            state = convert_state_to_positive_preconditions(state, self._mapping)
        return ff_heuristic(state, self._goal, self._actions)


def _greedy_action(state: State, heuristic: _GreedyHeuristic, actions):
    """The applicable action minimizing expected FF heuristic over outcomes."""
    best_action, best_value = None, float("inf")
    for action in get_next_actions(state, actions):
        expected = 0.0
        for successor, prob in transition(state, action):
            expected += prob * (successor.time + heuristic(successor))
        if expected < best_value:
            best_action, best_value = action, expected
    return best_action


def _apply(state: State, action, rng: random.Random) -> State:
    """Sample a successor from the transition distribution."""
    successors = transition(state, action)
    if not successors:
        raise RuntimeError(f"Action {action.name} produced no successors")
    weights = [prob for _, prob in successors]
    index = rng.choices(range(len(successors)), weights=weights, k=1)[0]
    return successors[index][0]
