"""Environment classes for robot simulation and planning execution.

This module provides the recommended interface for robot environments
used in PDDL planning and simulation.

Usage:
    from railroad.environment import (
        Environment,             # Abstract base class for environments
        SymbolicEnvironment,     # Generic symbolic execution
        ObjectSearchEnvironment, # Object-search domain conventions
        ActiveSkill,             # Protocol for skill execution
        SymbolicSkill,           # Symbolic skill implementation
    )

Running actions on real robots or an external simulator:
    from railroad.environment import (
        PhysicalEnvironment,       # Subclass: define_operators, execute_skill,
                                   #   get_skill_status, stop_robot
        SkillStatus,               # IDLE / RUNNING / DONE / FAILED
    )

Legacy classes have been moved to railroad.experimental.environment:
    from railroad.experimental.environment import (
        AbstractEnvironment, BaseEnvironment, SimpleEnvironment,
        EnvironmentInterface, OngoingAction, SkillStatus, SimulatedRobot, Pose,
    )

Unknown-space frontier search lives in railroad.experimental.unknown_search.
"""

from .environment import ActiveSkill, Environment
from .skill import (
    InterruptibleNavigationMoveSkill,
    MotionSkill,
    NavigationMoveSkill,
    PhysicalSkill,
    SkillStatus,
)
from .types import Pose, PoseLike, TopDownView
from .symbolic import (
    LocationRegistry,
    SymbolicEnvironment,
    SymbolicSkill,
)
from .object_search import ObjectSearchEnvironment
from .physical import PhysicalEnvironment

__all__ = [
    # Core classes
    "ActiveSkill",
    "Environment",
    "InterruptibleNavigationMoveSkill",
    "LocationRegistry",
    "MotionSkill",
    "NavigationMoveSkill",
    "ObjectSearchEnvironment",
    "Pose",
    "PoseLike",
    "PhysicalEnvironment",
    "PhysicalSkill",
    "SkillStatus",
    "SymbolicEnvironment",
    "SymbolicSkill",
    "TopDownView",
]
