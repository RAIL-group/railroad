"""The three replay environments, one per deployment flavor.

Each mirrors the deployment environment it replays (same operators and dynamics)
but sources its "world" from a recorded :class:`~railroad.replay.types.RolloutLog`
instead of a live simulator, and is *policy-agnostic* until
:meth:`~railroad.replay.environments.base.ReplayArenaMixin.apply_policy` swaps in a
candidate. Build one with :func:`~railroad.replay.driver.build_replay_env` (which
dispatches on ``log.problem_class``) rather than constructing directly.
"""

from .base import (
    ReplayArenaMixin,
    ReplayConfinementMixin,
    ServedPano,
    navigation_config_from_log,
    objects_in_goal,
    require_goal,
    robot_from_free,
)
from .known_map_search import ReplayKnownMapSearchEnvironment
from .point_goal_nav import (
    ReplayPointGoalNavEnvironment,
    frontier_sweep_select,
    goal_fluent,
)
from .unknown_search import (
    ReplayUnknownSearchEnvironment,
    learned_frontier_search_prob,
)

__all__ = [
    "ReplayArenaMixin",
    "ReplayConfinementMixin",
    "ReplayKnownMapSearchEnvironment",
    "ReplayPointGoalNavEnvironment",
    "ReplayUnknownSearchEnvironment",
    "ServedPano",
    "frontier_sweep_select",
    "goal_fluent",
    "learned_frontier_search_prob",
    "navigation_config_from_log",
    "objects_in_goal",
    "require_goal",
    "robot_from_free",
]
