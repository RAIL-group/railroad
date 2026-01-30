"""Environment interface classes - backward compatibility shim.

This module re-exports classes from railroad.environment for backward compatibility.
New code should import directly from railroad.environment.
"""

# Re-export everything from railroad.environment
from railroad.environment import (
    EnvironmentInterface,
    OngoingAction,
    OngoingSearchAction,
    OngoingPickAction,
    OngoingPlaceAction,
    OngoingMoveAction,
    OngoingNoOpAction,
)

__all__ = [
    "EnvironmentInterface",
    "OngoingAction",
    "OngoingSearchAction",
    "OngoingPickAction",
    "OngoingPlaceAction",
    "OngoingMoveAction",
    "OngoingNoOpAction",
]
