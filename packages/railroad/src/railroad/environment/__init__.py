"""Environment classes for robot simulation and planning execution.

This module provides base classes and interfaces for robot environments
used in PDDL planning and simulation.

Usage:
    from railroad.environment import BaseEnvironment, EnvironmentInterface

Available classes:
- BaseEnvironment: Abstract base class for environment implementations
- SkillStatus: Enum for skill execution status (IDLE, RUNNING, DONE)
- SimulatedRobot: Simple robot state tracking for simulation
- EnvironmentInterface: Bridge between PDDL planning and environment execution
- OngoingAction: Base class for tracking action execution
- OngoingSearchAction, OngoingPickAction, OngoingPlaceAction,
  OngoingMoveAction, OngoingNoOpAction: Specialized action trackers
"""

from .base import (
    BaseEnvironment,
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

__all__ = [
    # Base classes
    "BaseEnvironment",
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
