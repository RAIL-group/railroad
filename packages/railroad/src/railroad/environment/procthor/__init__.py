"""ProcTHOR environment for AI2-THOR/ProcTHOR simulation.

This module provides integration with ProcTHOR 3D indoor environments.
Requires optional dependencies: pip install railroad[procthor]

Resources are automatically downloaded on first import when dependencies are
available. Set PROCTHOR_AUTO_DOWNLOAD=0 to disable auto-download.
"""

import os

__all__ = ["ProcTHORScene", "ProcTHOREnvironment", "is_available", "ensure_all_resources"]

_INSTALL_MSG = (
    "ProcTHOR dependencies not installed. "
    "Install with: pip install railroad[procthor]"
)

# Required packages for procthor functionality
_REQUIRED_PACKAGES = [
    "ai2thor",
    "sentence_transformers",
    "prior",
    "shapely",
    "networkx",
    "scipy",
    "skimage",
]


def is_available() -> bool:
    """Check if all procthor dependencies are installed.

    This is a lightweight check that doesn't load heavy modules.
    """
    import importlib.util

    return all(importlib.util.find_spec(pkg) is not None for pkg in _REQUIRED_PACKAGES)


# Auto-download resources on import if dependencies are available
if os.environ.get("PROCTHOR_AUTO_DOWNLOAD", "1") != "0" and is_available():
    from .resources import ensure_all_resources

    ensure_all_resources()


def __getattr__(name: str):
    if name == "ProcTHORScene":
        try:
            from .scene import ProcTHORScene
            return ProcTHORScene
        except ImportError as e:
            raise ImportError(_INSTALL_MSG) from e
    elif name == "ProcTHOREnvironment":
        try:
            from .environment import ProcTHOREnvironment
            return ProcTHOREnvironment
        except ImportError as e:
            raise ImportError(_INSTALL_MSG) from e
    elif name == "ensure_all_resources":
        try:
            from .resources import ensure_all_resources
            return ensure_all_resources
        except ImportError as e:
            raise ImportError(_INSTALL_MSG) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
