"""Environment classes for robot simulation and planning execution.

This module provides base classes and interfaces for robot environments
used in PDDL planning and simulation.

Usage:
    from railroad.environment import AbstractEnvironment, EnvironmentInterface

Available classes:
- AbstractEnvironment: Abstract base class for environment implementations
- SimpleEnvironment: Reference implementation for testing and examples
- SkillStatus: Enum for skill execution status (IDLE, RUNNING, DONE)
- SimulatedRobot: Simple robot state tracking for simulation
- EnvironmentInterface: Bridge between PDDL planning and environment execution
- OngoingAction: Base class for tracking action execution
- OngoingSearchAction, OngoingPickAction, OngoingPlaceAction,
  OngoingMoveAction, OngoingNoOpAction: Specialized action trackers
- ActiveSkill: Protocol for tracking execution of a single action
- Environment: Protocol for environment that owns world state
- SymbolicSkill: Symbolic skill execution (action-driven, no subclasses needed)
- SimpleSymbolicEnvironment: Simple environment for symbolic execution
- EnvironmentInterfaceV2: New interface using Environment/ActiveSkill architecture
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

from .skill import (
    ActiveSkill,
    Environment,
    SymbolicSkill,
)

from .symbolic import SimpleSymbolicEnvironment

from .interface_v2 import EnvironmentInterfaceV2

# Backward compatibility alias
BaseEnvironment = AbstractEnvironment

__all__ = [
    # Base classes
    "AbstractEnvironment",
    "BaseEnvironment",  # Backward compatibility alias
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
    # New interface components (v2)
    "ActiveSkill",
    "Environment",
    "SymbolicSkill",
    "SimpleSymbolicEnvironment",
    "EnvironmentInterfaceV2",
]
