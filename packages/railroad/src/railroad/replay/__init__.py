"""Offline replay: counterfactual cost bounds from a single deployment.

From one real deployment of a chosen policy, *offline replay* computes a lower
bound on what an alternative policy would have cost — without deploying it — by
replaying the alternative over the recorded observations and bounding its cost
the moment it commits to a subgoal whose outcome the deployment never recorded.

The flow is three calls, one per stage::

    log    = build_rollout_log(deployment_env, ...)   # record (no ground truth)
    env    = build_replay_env(log)                     # reconstruct the world
    result = run_replay(env, candidate_policy, ...)    # replay -> cost bounds

Change the policy, hold everything else constant: that is the whole idea. GL-free
and torch-free at the core (the dashboard/render path imports lazily). The bound
math lives in :mod:`railroad.replay.cost`; the recorder in
:mod:`railroad.replay.recorder`; the replay environments in
:mod:`railroad.replay.environments`; the driver in :mod:`railroad.replay.driver`.
"""

from .cost import (
    Bounds,
    Commit,
    accumulate_bounds,
    cost_at_cell,
    optimistic_cost_grid_from_goal,
    optimistic_cost_to_goal,
)
from .deployment import DeploymentResult, run_deployment
from .driver import build_replay_env, run_replay
from .environments import (
    ReplayKnownMapSearchEnvironment,
    ReplayPointGoalNavEnvironment,
    ReplayUnknownSearchEnvironment,
    frontier_sweep_select,
    goal_fluent,
    navigation_config_from_log,
)
from .loop import MctsConfig, mcts_selector, plan_act_loop, run_dashboard_loop
from .policy import (
    ConstantFrontierStatisticsModel,
    OracleObjectFind,
    constant_frontier_statistics,
    learned_container_find,
    learned_frontier_statistics,
    oracle_frontier_statistics,
    oracle_object_find,
    scene_goal_cell,
    target_container_cells,
)
from .recorder import build_rollout_log
from .selection import select_policy
from .serialization import load_rollout_log, save_rollout_log
from .types import ReplayResult, RolloutLog, StepRecord, SubgoalRecord

__all__ = [
    # Records + bounds
    "RolloutLog",
    "SubgoalRecord",
    "StepRecord",
    "ReplayResult",
    "Bounds",
    "Commit",
    "accumulate_bounds",
    "optimistic_cost_grid_from_goal",
    "optimistic_cost_to_goal",
    "cost_at_cell",
    # Selection
    "select_policy",
    # Policies. A policy IS an estimator, and which one depends on the problem
    # class; these builders come in navigation/search pairs. The *registry* of
    # which policies a study compares lives with the experiment, not here.
    "oracle_frontier_statistics",
    "oracle_object_find",
    "constant_frontier_statistics",
    "learned_frontier_statistics",
    "learned_container_find",
    "scene_goal_cell",
    "target_container_cells",
    "OracleObjectFind",
    # The deploy -> record -> replay flow
    "run_deployment",
    "DeploymentResult",
    "build_rollout_log",
    "save_rollout_log",
    "load_rollout_log",
    "build_replay_env",
    "run_replay",
    # Loop / planner config (shared by deployment and replay)
    "MctsConfig",
    "mcts_selector",
    "plan_act_loop",
    "run_dashboard_loop",
    # Replay environments + helpers
    "ReplayPointGoalNavEnvironment",
    "ReplayUnknownSearchEnvironment",
    "ReplayKnownMapSearchEnvironment",
    "goal_fluent",
    "frontier_sweep_select",
    "navigation_config_from_log",
    # Stand-in for a trained net: fakes only the numbers, runs the real pipeline
    "ConstantFrontierStatisticsModel",
]
