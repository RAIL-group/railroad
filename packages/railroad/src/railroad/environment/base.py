"""Base environment classes for robot simulation.

This module provides abstract base classes and common implementations
for robot environments used in planning and simulation.
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Callable, Dict, Set, Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Pose can be an ndarray or a tuple of coordinates
Pose = Union[NDArray[np.floating[Any]], Tuple[float, ...]]


class SkillStatus(IntEnum):
    """Status of a robot skill execution."""

    IDLE = -1
    RUNNING = 0
    DONE = 1


class BaseEnvironment(ABC):
    """Abstract base class for all environments.

    Provides the interface for robot environments used in planning
    and simulation. Subclasses must implement all abstract methods.
    """

    def __init__(self) -> None:
        self.time: float = 0.0

    @abstractmethod
    def get_objects_at_location(self, location: str) -> Dict[str, Set[str]]:
        """Get objects at a location (perception method).

        This is a perception method that returns objects visible at a location.
        In simulators, this comes from ground truth. In real robots, this would
        be replaced by a perception module.

        Args:
            location: The location to query.

        Returns:
            Dictionary mapping object types to sets of object names.
        """
        ...

    @abstractmethod
    def remove_object_from_location(self, obj: str, location: str) -> None:
        """Remove an object from a location.

        Called when an object is picked up.

        Args:
            obj: Name of the object to remove.
            location: Name of the location.
        """
        ...

    @abstractmethod
    def add_object_at_location(self, obj: str, location: str) -> None:
        """Add an object to a location.

        Called when an object is placed.

        Args:
            obj: Name of the object to add.
            location: Name of the location.
        """
        ...

    @abstractmethod
    def execute_skill(self, robot_name: str, skill_name: str, *args: Any, **kwargs: Any) -> None:
        """Execute a skill on a robot.

        Args:
            robot_name: Name of the robot.
            skill_name: Name of the skill to execute.
            *args: Positional arguments for the skill.
            **kwargs: Keyword arguments for the skill.
        """
        ...

    @abstractmethod
    def get_skills_time_fn(self, skill_name: str) -> Callable[..., float]:
        """Get a time function for a skill.

        Args:
            skill_name: Name of the skill.

        Returns:
            Function that computes the time for the skill.
        """
        ...

    @abstractmethod
    def get_executed_skill_status(self, robot_name: str, skill_name: str) -> SkillStatus:
        """Get the execution status of a skill.

        Args:
            robot_name: Name of the robot.
            skill_name: Name of the skill.

        Returns:
            Current status of the skill execution.
        """
        ...

    @abstractmethod
    def stop_robot(self, robot_name: str) -> None:
        """Stop a robot's current action.

        Args:
            robot_name: Name of the robot to stop.
        """
        ...


class SimulatedRobot:
    """A simulated robot with basic state tracking.

    Tracks robot name, pose, current action, and availability.
    """

    def __init__(self, name: str, pose: Optional[Pose] = None) -> None:
        """Initialize a simulated robot.

        Args:
            name: Name of the robot.
            pose: Initial pose (optional).
        """
        self.name = name
        self.current_action_name: Optional[str] = None
        self.pose = pose
        self.target_pose: Optional[Pose] = None
        self.is_free = True

    def __repr__(self) -> str:
        return f"SimulatedRobot(name={self.name}, pose={self.pose})"

    def move(self, new_pose: Pose) -> None:
        """Start moving to a new pose.

        Args:
            new_pose: Target pose to move to.
        """
        self.current_action_name = "move"
        self.is_free = False
        self.target_pose = new_pose

    def pick(self) -> None:
        """Start a pick action."""
        self.current_action_name = "pick"
        self.is_free = False

    def place(self) -> None:
        """Start a place action."""
        self.current_action_name = "place"
        self.is_free = False

    def search(self) -> None:
        """Start a search action."""
        self.current_action_name = "search"
        self.is_free = False

    def no_op(self) -> None:
        """Start a no-op (wait) action."""
        self.current_action_name = "no_op"
        self.is_free = False

    def stop(self) -> None:
        """Stop the current action."""
        self.current_action_name = None
        self.is_free = True
        self.target_pose = None
