"""Environment classes for robot simulation and planning execution.

This module provides the recommended interface for robot environments
used in PDDL planning and simulation.

Usage:
    from railroad.environment import (
        Environment,           # Abstract base class for environments
        SymbolicEnvironment,   # Environment for symbolic execution
        ActiveSkill,           # Protocol for skill execution
        SymbolicSkill,         # Symbolic skill implementation
    )

Legacy classes have been moved to railroad.experimental.environment:
    from railroad.experimental.environment import (
        AbstractEnvironment, BaseEnvironment, SimpleEnvironment,
        EnvironmentInterface, OngoingAction, SkillStatus, SimulatedRobot, Pose,
    )
"""

from .environment import Environment
from .physical import PhysicalEnvironment, PhysicalScene, PhysicalSkill
from .skill import ActiveSkill, SkillStatus
from .symbolic import (
    InterruptableMoveSymbolicSkill,
    LocationRegistry,
    SymbolicEnvironment,
    SymbolicSkill,
)

__all__ = [
    # Core classes
    "ActiveSkill",
    "Environment",
    "InterruptableMoveSymbolicSkill",
    "LocationRegistry",
    "PhysicalEnvironment",
    "PhysicalScene",
    "PhysicalSkill",
    "SkillStatus",
    "SymbolicEnvironment",
    "SymbolicSkill",
]
