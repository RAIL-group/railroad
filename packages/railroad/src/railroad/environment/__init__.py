"""Environment classes for robot simulation and planning execution.

This module provides the recommended interface for robot environments
used in PDDL planning and simulation.

Usage:
    from railroad.environment import (
        Environment,           # Abstract base class for environments
        SymbolicEnvironment,  # Environment for symbolic execution
        SimpleSymbolicEnvironment,  # Alias for backward compatibility
        ActiveSkill,          # Protocol for skill execution
        SymbolicSkill,        # Symbolic skill implementation
    )

Legacy classes have been moved to railroad.experimental.environment:
    from railroad.experimental.environment import (
        AbstractEnvironment, BaseEnvironment, SimpleEnvironment,
        EnvironmentInterface, OngoingAction, SkillStatus, SimulatedRobot, Pose,
    )
"""

from .environment import Environment
from .skill import (
    ActiveSkill,
    InterruptableMoveSymbolicSkill,
    SymbolicSkill,
)
from .symbolic import SimpleSymbolicEnvironment, SymbolicEnvironment

__all__ = [
    # Core classes
    "ActiveSkill",
    "Environment",
    "InterruptableMoveSymbolicSkill",
    "SimpleSymbolicEnvironment",
    "SymbolicEnvironment",
    "SymbolicSkill",
]
