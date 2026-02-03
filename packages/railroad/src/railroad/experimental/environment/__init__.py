"""Legacy environment classes for robot simulation and planning execution.

This module contains the original environment interface that has been superseded
by the newer Environment/ActiveSkill architecture in railroad.environment.

For new code, prefer using:
- railroad.environment.Environment (abstract base class)
- railroad.environment.SymbolicEnvironment (concrete implementation)
- railroad.environment.SimpleSymbolicEnvironment (backward compatibility alias)
- railroad.environment.ActiveSkill (protocol)
- railroad.environment.SymbolicSkill

This legacy interface is preserved for backward compatibility.
"""

from .base import (
    AbstractEnvironment,
    SimpleEnvironment,
    SkillStatus,
    SimulatedRobot,
    Pose,
)

from .interface import (
    EnvironmentInterface,
    OngoingAction,
    OngoingSearchAction,
    OngoingPickAction,
    OngoingPlaceAction,
    OngoingMoveAction,
    OngoingNoOpAction,
)

# Backward compatibility alias
BaseEnvironment = AbstractEnvironment

__all__ = [
    # Base classes
    "AbstractEnvironment",
    "BaseEnvironment",
    "SimpleEnvironment",
    "SkillStatus",
    "SimulatedRobot",
    "Pose",
    # Interface classes
    "EnvironmentInterface",
    "OngoingAction",
    "OngoingSearchAction",
    "OngoingPickAction",
    "OngoingPlaceAction",
    "OngoingMoveAction",
    "OngoingNoOpAction",
]
