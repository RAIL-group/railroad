"""Environment classes for robot simulation and planning execution.

This module provides the recommended interface for robot environments
used in PDDL planning and simulation.

Usage:
    from railroad.environment import (
        Environment,           # Protocol for environments
        ActiveSkill,          # Protocol for skill execution
        SymbolicSkill,        # Symbolic skill implementation
        SimpleSymbolicEnvironment,  # Simple environment for symbolic execution
        EnvironmentInterfaceV2,     # Main interface for planning/execution
    )

Legacy classes have been moved to railroad.experimental.environment:
    from railroad.experimental.environment import (
        AbstractEnvironment, BaseEnvironment, SimpleEnvironment,
        EnvironmentInterface, OngoingAction, SkillStatus, SimulatedRobot, Pose,
    )
"""

from .skill import (
    ActiveSkill,
    Environment,
    SymbolicSkill,
)

from .symbolic import SimpleSymbolicEnvironment

from .interface_v2 import EnvironmentInterfaceV2

__all__ = [
    "ActiveSkill",
    "Environment",
    "SymbolicSkill",
    "SimpleSymbolicEnvironment",
    "EnvironmentInterfaceV2",
]
