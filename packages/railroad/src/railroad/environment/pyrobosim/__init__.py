"""PyRoboSim environment: railroad plans, PyRoboSim robots act.

- :class:`PyRoboSimScene` loads a PyRoboSim world and exposes what planning
  needs: robots, locations (with coordinates), objects, ground-truth object
  locations, and path-planned move times.
- :class:`PyRoboSimEnvironment` is a :class:`~railroad.environment.PhysicalEnvironment`
  whose ``execute_skill`` runs move, search, pick, place and no_op on the
  PyRoboSim robots, one worker thread per skill; subclasses provide
  ``define_operators()``.

Requires optional dependencies: pip install railroad[pyrobosim]
"""

__all__ = [
    "DEFAULT_WORLD_FILE",
    "MatplotlibWorldCanvas",
    "PyRoboSimEnvironment",
    "PyRoboSimScene",
    "is_available",
]

_INSTALL_MSG = (
    "PyRoboSim dependencies not installed. "
    "Install with: pip install railroad[pyrobosim]"
)

# Required packages for pyrobosim functionality
_REQUIRED_PACKAGES = [
    "pyrobosim",
]


def is_available() -> bool:
    """Check if all pyrobosim dependencies are installed.

    This is a lightweight check that doesn't load heavy modules.
    """
    import importlib.util

    return all(importlib.util.find_spec(pkg) is not None for pkg in _REQUIRED_PACKAGES)


def __getattr__(name: str):
    if name in ("PyRoboSimScene", "DEFAULT_WORLD_FILE"):
        try:
            from . import scene
            return getattr(scene, name)
        except ImportError as e:
            raise ImportError(_INSTALL_MSG) from e
    elif name == "PyRoboSimEnvironment":
        try:
            from .environment import PyRoboSimEnvironment
            return PyRoboSimEnvironment
        except ImportError as e:
            raise ImportError(_INSTALL_MSG) from e
    elif name == "MatplotlibWorldCanvas":
        try:
            from .canvas import MatplotlibWorldCanvas
            return MatplotlibWorldCanvas
        except ImportError as e:
            raise ImportError(_INSTALL_MSG) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
