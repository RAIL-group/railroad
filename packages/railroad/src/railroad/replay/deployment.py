"""Run a deployment and record it — the mirror image of :mod:`~railroad.replay.driver`.

Replay is two calls (``build_replay_env`` then ``run_replay``). This gives the
other half the same shape: :func:`run_deployment` drives the shared plan->act
loop over a live environment and returns the :class:`RolloutLog` that replay
consumes, so the two phases read symmetrically::

    log    = run_deployment(dep_env, goal, goal_cell=..., robot_starts=...)
    result = run_replay(build_replay_env(log), candidate)

It also assembles the log in **one** place, so the flavor-specific ``goal_cell``
/ ``robot_starts`` / ``problem_class`` wiring is not repeated per experiment
script, where it would be free to drift.

The planner is *not* defaulted here: pass the same :class:`MctsConfig` the replay
will use, so the counterfactual varies only the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .loop import (
    ActionSelector,
    MctsConfig,
    mcts_selector,
    plan_act_loop,
    run_dashboard_loop,
)
from .recorder import build_rollout_log
from .types import RolloutLog


@dataclass
class DeploymentResult:
    """What one recorded deployment produced."""

    #: The recording offline replay consumes.
    log: RolloutLog
    #: Why the loop stopped: goal_reached / no_actions / planner_none / max_iterations.
    termination: str
    #: Realized makespan in seconds (``env.state.time``) — the cost replay bounds
    #: are compared against.
    total_cost: float
    goal_reached: bool


def run_deployment(
    env: Any,
    goal: Any,
    *,
    goal_cell: tuple,
    robot_starts: Mapping[str, Any],
    problem_class: str = "navigation",
    mcts: Optional[MctsConfig] = None,
    max_planning_iterations: int = 100,
    select_action: Optional[ActionSelector] = None,
    dashboard: bool = False,
    scene: Optional[Any] = None,
    save_video: Optional[str] = None,
    label: str = "",
    fluent_keywords: tuple = ("at", "found", "searched", "explored", "revealed"),
    env_name: str = "",
    seed: Optional[int] = None,
) -> DeploymentResult:
    """Drive *env* toward *goal*, then snapshot it into a :class:`RolloutLog`.

    The policy is expected to be **already installed** on *env* — assigned to
    ``env.frontier_statistics`` for navigation, or ``env.object_find_statistics``
    for search. That keeps this function about *running and recording*, and
    leaves how a policy attaches to each problem class where it belongs.

    Silent by default; ``dashboard=True`` renders the standard
    ``PlannerDashboard`` (optionally writing *save_video*, with *scene* supplying
    the overhead map) — the same loop replay runs, so the recording is identical
    either way.
    """
    select = select_action or mcts_selector(mcts or MctsConfig())

    if dashboard:
        termination = run_dashboard_loop(
            env,
            goal,
            select=select,
            max_iterations=max_planning_iterations,
            fluent_keywords=fluent_keywords,
            scene=scene,
            save_video=save_video,
            label=label,
        )
    else:
        termination = plan_act_loop(
            env, goal, select=select, max_iterations=max_planning_iterations
        )

    log = build_rollout_log(
        env,
        goal_cell=goal_cell,
        robot_starts=robot_starts,
        env_name=env_name,
        seed=seed,
        problem_class=problem_class,
        goal=goal,
    )
    return DeploymentResult(
        log=log,
        termination=termination,
        total_cost=float(env.state.time),
        goal_reached=bool(goal.evaluate(env.state.fluents)),
    )
