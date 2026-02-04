"""ProcTHOR environment for AI2-THOR/ProcTHOR simulation.

This module provides integration with ProcTHOR 3D indoor environments.
Requires optional dependencies: pip install railroad[procthor]
"""

__all__ = ["ProcTHORScene", "ProcTHOREnvironment"]

_INSTALL_MSG = (
    "ProcTHOR dependencies not installed. "
    "Install with: pip install railroad[procthor]"
)


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
