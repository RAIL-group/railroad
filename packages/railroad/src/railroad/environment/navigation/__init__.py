"""Unknown-space navigation environment for frontier-based exploration.

This module provides grid-based navigation with laser sensing, frontier
extraction, and hidden-site discovery for multi-robot planning.
Requires optional dependencies: pip install railroad[navigation]
"""

from ..types import Pose
from .types import NavigationConfig

__all__ = [
    "is_available",
    "NavigationConfig",
    "Pose",
    "NavigationMoveSkill",
    "InterruptibleNavigationMoveSkill",
    "UnknownSpaceEnvironment",
]

_REQUIRED_PACKAGES = ["scipy", "skimage"]

_INSTALL_MSG = (
    "Navigation dependencies not installed. "
    "Install with: pip install railroad[navigation]"
)


def is_available() -> bool:
    """Check if navigation dependencies are installed."""
    import importlib.util

    return all(importlib.util.find_spec(pkg) is not None for pkg in _REQUIRED_PACKAGES)


def __getattr__(name: str):
    if name == "UnknownSpaceEnvironment":
        try:
            from .environment import UnknownSpaceEnvironment
            return UnknownSpaceEnvironment
        except ImportError as e:
            raise ImportError(_INSTALL_MSG) from e
    if name == "NavigationMoveSkill":
        try:
            from .skill import NavigationMoveSkill
            return NavigationMoveSkill
        except ImportError as e:
            raise ImportError(_INSTALL_MSG) from e
    if name == "InterruptibleNavigationMoveSkill":
        try:
            from .skill import InterruptibleNavigationMoveSkill
            return InterruptibleNavigationMoveSkill
        except ImportError as e:
            raise ImportError(_INSTALL_MSG) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
