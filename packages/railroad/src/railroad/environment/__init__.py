"""Environment classes for robot simulation and planning execution.

This module provides the recommended interface for robot environments
used in PDDL planning and simulation.

Usage:
    from railroad.environment import (
        Environment,           # Abstract base class for environments
        ActiveSkill,          # Protocol for skill execution
        SymbolicSkill,        # Symbolic skill implementation
        SymbolicEnvironment,  # Environment for symbolic execution
        SimpleSymbolicEnvironment,  # Alias for backward compatibility
        EnvironmentInterfaceV2,     # Main interface for planning/execution
    )

Legacy classes have been moved to railroad.experimental.environment:
    from railroad.experimental.environment import (
        AbstractEnvironment, BaseEnvironment, SimpleEnvironment,
        EnvironmentInterface, OngoingAction, SkillStatus, SimulatedRobot, Pose,
    )
"""

from .environment import Environment
from .interface_v2 import EnvironmentInterfaceV2
from .skill import (
    ActiveSkill,
    InterruptableMoveSymbolicSkill,
    SymbolicSkill,
)
from .symbolic import SimpleSymbolicEnvironment, SymbolicEnvironment

__all__ = [
    # Legacy
    "EnvironmentInterfaceV2",
    # New architecture
    "ActiveSkill",
    "Environment",
    "InterruptableMoveSymbolicSkill",
    "SimpleSymbolicEnvironment",
    "SymbolicEnvironment",
    "SymbolicSkill",
]
