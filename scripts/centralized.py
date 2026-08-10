"""Centralized (joint) planning methods: one search over the *combined*
action space of all robots, against a single AND-goal. No coordination
protocol needed — mutual exclusion on a shared resource falls out of the
state model for free (once one robot holds it, it's no longer `at`
anywhere for another robot's pick to find).

Two methods:
- plan_joint_astar: native C++ AStarPlanner, provably optimal — correct in
  principle but empirically intractable at any real scale (measured: 6GB+
  and climbing within 30s on a reduced 2-robot goal, killed before
  converging). Kept for completeness/comparison, not general use.
- plan_joint_mcts: native C++ MCTSPlanner, satisficing — tractable (seconds)
  but not optimal; needs a no_op operator for the momentary
  zero-legal-actions deadlock case (e.g. right after a move, a robot is
  nominally "free" but blocked from another move by `just-moved` and has
  nothing else to do for an instant).

Both take a `Domain` (see common.py) and return a `PlanResult`.
"""

import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from railroad.core import AndGoal, Fluent as F, State, get_action_by_name, transition as _transition
from railroad.environment.symbolic import SymbolicEnvironment
from railroad.planner import AStarPlanner, MCTSPlanner
from railroad import operators

from planner_interface import _str_to_fluent, _goal_from_str  # pyrefly: ignore [missing-import]

from common import (  # pyrefly: ignore [missing-import]
    Domain,
    PlanResult,
    PlanStep,
    make_logger,
    free_again_duration,
    restrict_objects_by_type,
)


def _combined_goal(domain: Domain):
    """AND of every robot's *first queued* goal, parsed and combined into one Goal.

    domain.robot_goals is a queue per robot (see Domain's docstring), but
    joint search over even a single goal per robot is already at the edge of
    tractable (plan_joint_astar) — there's no multi-task-queue support here,
    unlike the decentralized methods (run_decentralized), which work through
    each robot's whole queue automatically.
    """
    goals = [_goal_from_str(tasks[0]) for tasks in domain.robot_goals.values()]
    return goals[0] if len(goals) == 1 else AndGoal(goals)


def _ground_joint(domain: Domain, my_operators: List):
    fluents = {_str_to_fluent(s) for s in domain.initial_state}
    state = State(0.0, fluents)
    # Restrict grounding to what's actually named across every robot's
    # first-queued goal — same reasoning as run_decentralized: an
    # unrestricted objects_by_type grounds pick/place for every irrelevant
    # object too, which bloats the *joint* action space even more severely
    # than the per-robot decentralized case (see restrict_objects_by_type).
    combined_goal_text = " ".join(tasks[0] for tasks in domain.robot_goals.values())
    task_objects = restrict_objects_by_type(
        domain.objects_by_type, combined_goal_text, set(domain.contested_resources.keys()),
    )
    env = SymbolicEnvironment(state, task_objects, my_operators)
    all_actions = env.get_actions()  # unfiltered: every robot's actions together
    goal = _combined_goal(domain)
    return state, env, all_actions, goal


def _print_steps(robots: List[str], steps: List[PlanStep]) -> None:
    log = make_logger(robots)
    merged: List[Tuple[float, str, str]] = sorted(
        [(s.start, s.robot, f"START  {s.action}") for s in steps]
        + [(s.end, s.robot, f"END    {s.action}") for s in steps],
        key=lambda x: x[0],
    )
    for t, robot, msg in merged:
        log(t, robot, msg)


# ---------------------------------------------------------------------------
# 1. Joint A* — optimal, exhaustive
# ---------------------------------------------------------------------------


def plan_joint_astar(domain: Domain, verbose: bool = False) -> PlanResult:
    pick_op = operators.construct_pick_operator_blocking(pick_time=domain.pick_time)
    place_op = operators.construct_place_operator_blocking(place_time=domain.place_time)
    my_operators = list(domain.base_operators) + [pick_op, place_op]

    state, _env, all_actions, goal = _ground_joint(domain, my_operators)

    planner = AStarPlanner(all_actions)
    plan = planner.plan_sequence(state, goal)

    if not plan:
        return PlanResult(success=False, cost=None, steps=[], message="No joint plan found")

    # _prepare applies the same negative-precondition -> positive-fluent
    # conversion plan_sequence used internally, giving a converted_state
    # consistent with the returned (also-converted) plan actions, so we can
    # replay via the same `transition` the search itself used.
    converted_state, _ = planner._prepare(state, goal)

    s = converted_state
    steps: List[PlanStep] = []
    for action in plan:
        robot = action.name.split()[1]
        start = s.time
        duration = free_again_duration(action)
        steps.append(PlanStep(robot=robot, action=action.name, start=start, end=start + duration))
        for successor, prob in _transition(s, action):
            if prob == 0.0:
                continue
            s = successor
            break

    if verbose:
        _print_steps(domain.robots, steps)

    cost = max(st.end for st in steps)
    return PlanResult(success=True, cost=cost, steps=steps, message="")


# ---------------------------------------------------------------------------
# 2. Joint MCTS — satisficing, tractable
# ---------------------------------------------------------------------------


def plan_joint_mcts(
    domain: Domain,
    max_steps: int = 80,
    verbose: bool = False,
    max_iterations: int = 5000,
    c: float = 300,
    max_depth: int = 25,
    heuristic_multiplier: float = 3,
    no_op_time: float = 0.5,
) -> PlanResult:
    pick_op = operators.construct_pick_operator_blocking(pick_time=domain.pick_time)
    place_op = operators.construct_place_operator_blocking(place_time=domain.place_time)
    no_op_op = operators.construct_no_op_operator(no_op_time=no_op_time, extra_cost=10.0)
    my_operators = list(domain.base_operators) + [pick_op, place_op, no_op_op]

    _state, env, _all_actions, goal = _ground_joint(domain, my_operators)

    steps: List[PlanStep] = []

    for _ in range(max_steps):
        if goal.evaluate(env.state.fluents):
            break
        all_actions = env.get_actions()  # unfiltered: both robots' actions together
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(
            env.state, goal, max_iterations=max_iterations, c=c,
            max_depth=max_depth, heuristic_multiplier=heuristic_multiplier,
        )
        if action_name == "NONE":
            break

        action = get_action_by_name(all_actions, action_name)
        robot = action.name.split()[1]
        start = env.state.time
        duration = free_again_duration(action)
        steps.append(PlanStep(robot=robot, action=action.name, start=start, end=start + duration))
        env.act(action)

    success = goal.evaluate(env.state.fluents)

    if verbose:
        _print_steps(domain.robots, steps)

    steps.sort(key=lambda s: s.start)
    cost = max((st.end for st in steps), default=0.0) if success else None
    message = "" if success else f"Goal not achieved after {max_steps} steps"
    return PlanResult(success=success, cost=cost, steps=steps, message=message)
