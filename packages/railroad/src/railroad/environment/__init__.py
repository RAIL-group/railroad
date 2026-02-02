"""Environment classes for robot simulation and planning execution.

This module provides the recommended interface for robot environments
used in PDDL planning and simulation.

Recommended Usage:
    from railroad.environment import (
        Environment,           # Protocol for environments
        ActiveSkill,          # Protocol for skill execution
        SymbolicSkill,        # Symbolic skill implementation
        SimpleSymbolicEnvironment,  # Simple environment for symbolic execution
        EnvironmentInterfaceV2,     # Main interface for planning/execution
    )

Legacy classes (from railroad.experimental.environment):
    The following are re-exported for backward compatibility but are deprecated.
    For new code, use the classes above instead.
    - AbstractEnvironment, BaseEnvironment, SimpleEnvironment
    - EnvironmentInterface
    - OngoingAction and subclasses
    - SkillStatus, SimulatedRobot, Pose
"""

# New interface components (recommended)
from .skill import (
    ActiveSkill,
    Environment,
    SymbolicSkill,
)

from .symbolic import SimpleSymbolicEnvironment

from .interface_v2 import EnvironmentInterfaceV2


# Re-export legacy classes from experimental for backward compatibility
from railroad.experimental.environment import (
    AbstractEnvironment,
    SimpleEnvironment,
    SkillStatus,
    SimulatedRobot,
    Pose,
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
    # Recommended (v2) interface
    "ActiveSkill",
    "Environment",
    "SymbolicSkill",
    "SimpleSymbolicEnvironment",
    "EnvironmentInterfaceV2",
    # Legacy (re-exported from experimental for backward compatibility)
    "AbstractEnvironment",
    "BaseEnvironment",
    "SimpleEnvironment",
    "SkillStatus",
    "SimulatedRobot",
    "Pose",
    "EnvironmentInterface",
    "OngoingAction",
    "OngoingSearchAction",
    "OngoingPickAction",
    "OngoingPlaceAction",
    "OngoingMoveAction",
    "OngoingNoOpAction",
]
