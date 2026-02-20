"""Unknown-space navigation environment for frontier-based exploration.

This module provides grid-based navigation with laser sensing, frontier
extraction, and hidden-site discovery for multi-robot planning.
"""

from ..types import Pose
from .types import NavigationConfig
from .occupancy_grid_mixin import OccupancyGridPathingMixin

__all__ = [
    "NavigationConfig",
    "Pose",
    "OccupancyGridPathingMixin",
    "NavigationMoveSkill",
    "InterruptibleNavigationMoveSkill",
    "UnknownSpaceEnvironment",
]


def __getattr__(name: str):
    if name == "UnknownSpaceEnvironment":
        from .environment import UnknownSpaceEnvironment
        return UnknownSpaceEnvironment
    if name == "NavigationMoveSkill":
        from ..skill import NavigationMoveSkill
        return NavigationMoveSkill
    if name == "InterruptibleNavigationMoveSkill":
        from ..skill import InterruptibleNavigationMoveSkill
        return InterruptibleNavigationMoveSkill
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
