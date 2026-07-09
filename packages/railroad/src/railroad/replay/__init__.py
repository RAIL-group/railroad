"""Offline replay: counterfactual cost bounds from a single deployment.

From one real deployment of a chosen policy, *offline replay* computes a
lower bound on what an alternative policy would have cost — without
deploying it — by replaying the alternative over the recorded
observations and bounding its cost the moment it commits to a subgoal
whose outcome the deployment never recorded.

GL-free and torch-free. The bound math lives in
:mod:`railroad.replay.cost` (pure functions); the replay environment and
driver in :mod:`railroad.replay.replay_env`; on-disk logs in
:mod:`railroad.replay.serialization`.
"""

from .cost import (
    Bounds,
    Commit,
    accumulate_bounds,
    optimistic_cost_grid_from_goal,
    optimistic_cost_to_goal,
)
from .domains import (
    DOMAINS,
    KnownMapSearchDomain,
    MctsParams,
    NavigationDomain,
    ReplayDomain,
    UnknownSearchDomain,
    get_domain,
    replay,
)
from .policy import CandidatePolicy
from .known_map_search_replay_env import (
    KnownMapSearchReplayEnvironment,
    build_known_map_search_log,
    build_known_map_search_replay_env,
    run_known_map_search_replay,
)
from .recorder import build_rollout_log
from .selection import select_policy
from .replay_env import (
    ReplayEnvironment,
    build_replay_env,
    frontier_sweep_select,
    goal_fluent,
    run_replay,
)
from .search_replay_env import (
    SearchReplayEnvironment,
    build_search_replay_env,
    learned_frontier_search_prob,
    run_search_replay,
)
from .serialization import load_rollout_log, save_rollout_log
from .stub_model import (
    PresetFrontierStatisticsModel,
    PresetSearchModel,
    preset_model,
)
from .types import ReplayResult, RolloutLog, StepRecord, SubgoalRecord

__all__ = [
    "Bounds",
    "CandidatePolicy",
    "Commit",
    "DOMAINS",
    "KnownMapSearchDomain",
    "KnownMapSearchReplayEnvironment",
    "MctsParams",
    "NavigationDomain",
    "ReplayDomain",
    "UnknownSearchDomain",
    "get_domain",
    "replay",
    "PresetFrontierStatisticsModel",
    "PresetSearchModel",
    "ReplayEnvironment",
    "ReplayResult",
    "RolloutLog",
    "SearchReplayEnvironment",
    "StepRecord",
    "SubgoalRecord",
    "build_known_map_search_log",
    "build_known_map_search_replay_env",
    "build_search_replay_env",
    "run_known_map_search_replay",
    "learned_frontier_search_prob",
    "preset_model",
    "run_search_replay",
    "accumulate_bounds",
    "build_replay_env",
    "build_rollout_log",
    "frontier_sweep_select",
    "goal_fluent",
    "load_rollout_log",
    "optimistic_cost_grid_from_goal",
    "optimistic_cost_to_goal",
    "run_replay",
    "save_rollout_log",
    "select_policy",
]
