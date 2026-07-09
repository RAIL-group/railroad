"""The replay entry points: build an arena from a log, replay a policy over it.

Two calls make up the public flow:

* :func:`build_replay_env` reconstructs the *policy-agnostic* replay environment
  for a recorded :class:`~railroad.replay.types.RolloutLog`, dispatching on
  ``log.problem_class``. The returned arena carries only the deployment-observed
  world (recorded map, subgoals, panoramas) — no candidate policy yet.
* :func:`run_replay` applies a :class:`~railroad.replay.policy.CandidatePolicy` to
  that arena and drives the shared plan->act loop, returning the counterfactual
  :class:`~railroad.replay.types.ReplayResult` (cost bounds + provenance). It is
  silent by default; pass ``dashboard=True`` for the same rendered dashboard the
  deployment used.

To compare policies over one recording, build a fresh arena per candidate and
replay each — the loop the user runs::

    for policy in candidates:
        env = build_replay_env(log)
        result = run_replay(env, policy, mcts=cfg)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from .loop import (
    ActionSelector,
    MctsConfig,
    mcts_selector,
    plan_act_loop,
    run_dashboard_loop,
)
from .policy import CandidatePolicy
from .types import ReplayResult, RolloutLog


def _replay_env_classes() -> Dict[str, Type[Any]]:
    """problem_class -> replay env class (imported lazily to keep this GL-free)."""
    from .environments import (
        ReplayKnownMapSearchEnvironment,
        ReplayPointGoalNavEnvironment,
        ReplayUnknownSearchEnvironment,
    )

    return {
        "navigation": ReplayPointGoalNavEnvironment,
        "object-search": ReplayUnknownSearchEnvironment,
        "known-map-search": ReplayKnownMapSearchEnvironment,
    }


def build_replay_env(log: RolloutLog, *, config: Optional[Any] = None) -> Any:
    """Reconstruct the policy-agnostic replay env for *log* (dispatch on class).

    *config* optionally overrides the ``NavigationConfig`` rebuilt from the log
    (unknown-map flavors only); by default the deployment's recorded config is
    used so replay senses and maps exactly as the deployment did.
    """
    classes = _replay_env_classes()
    try:
        env_class = classes[log.problem_class]
    except KeyError:
        raise ValueError(
            f"no replay environment for problem_class={log.problem_class!r}; "
            f"known: {sorted(classes)}"
        ) from None
    return env_class.from_log(log, config=config)


def run_replay(
    env: Any,
    policy: Optional[CandidatePolicy] = None,
    *,
    dashboard: bool = False,
    scene: Optional[Any] = None,
    save_video: Optional[str] = None,
    label: str = "",
    mcts: Optional[MctsConfig] = None,
    max_planning_iterations: Optional[int] = None,
    select_action: Optional[ActionSelector] = None,
) -> ReplayResult:
    """Replay *policy* over the arena *env*; return its cost bounds.

    Applies *policy* (a bare :class:`CandidatePolicy` if ``None`` → neutral
    priors), then runs the shared plan->act loop and reduces the terminal state
    to a :class:`ReplayResult`. The planner is held fixed at the env's per-flavor
    defaults unless *mcts* / *max_planning_iterations* / *select_action* override
    them — pass the deployment's :class:`MctsConfig` so the counterfactual varies
    only the policy.

    Silent by default. With ``dashboard=True`` it renders through the standard
    ``PlannerDashboard`` (optionally writing *save_video* and exposing *scene* for
    the overhead map) — the same loop the deployment ran, so the reported bounds
    match either way.
    """
    env.apply_policy(policy or CandidatePolicy())
    goal = env.goal
    select = select_action or mcts_selector(mcts or env.default_mcts)
    max_iterations = (
        max_planning_iterations
        if max_planning_iterations is not None
        else env.default_max_planning_iterations
    )

    if dashboard:
        termination = run_dashboard_loop(
            env,
            goal,
            select=select,
            max_iterations=max_iterations,
            fluent_keywords=env.dashboard_fluent_keywords,
            scene=scene,
            save_video=save_video,
            label=label,
        )
    else:
        termination = plan_act_loop(
            env, goal, select=select, max_iterations=max_iterations
        )
    return env.finalize(termination)
