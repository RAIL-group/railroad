"""Environment classes for robot simulation and planning execution.

This module provides base classes and interfaces for robot environments
used in PDDL planning and simulation.

Usage:
    from railroad.environment import AbstractEnvironment, EnvironmentInterface

Available classes:
- AbstractEnvironment: Abstract base class for environment implementations
- SimpleEnvironment: Reference implementation for testing and examples
- SimpleOperatorEnvironment: Minimal wrapper where timing comes from operators
- SkillStatus: Enum for skill execution status (IDLE, RUNNING, DONE)
- SimulatedRobot: Simple robot state tracking for simulation
- EnvironmentInterface: Bridge between PDDL planning and environment execution
- OngoingAction: Base class for tracking action execution
- OngoingSearchAction, OngoingPickAction, OngoingPlaceAction,
  OngoingMoveAction, OngoingNoOpAction: Specialized action trackers
"""

from .base import (
    AbstractEnvironment,
    SimpleEnvironment,
    SimpleOperatorEnvironment,
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
    "BaseEnvironment",  # Backward compatibility alias
    "SimpleEnvironment",
    "SimpleOperatorEnvironment",
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
