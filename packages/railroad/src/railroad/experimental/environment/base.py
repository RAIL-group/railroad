"""Base environment classes for robot simulation.

This module provides abstract base classes and common implementations
for robot environments used in planning and simulation.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Set, Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from railroad.environment import SkillStatus

# Pose can be an ndarray or a tuple of coordinates
Pose = Union[NDArray[np.floating[Any]], Tuple[float, ...]]


class AbstractEnvironment(ABC):
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


class SimpleEnvironment(AbstractEnvironment):
    """Simple environment for testing multi-object manipulation.

    A reference implementation of AbstractEnvironment that provides:
    - In-memory storage of locations and objects
    - Euclidean distance-based movement costs
    - Fixed times for pick/place/search/no_op skills
    - Multi-robot support with time-based skill completion

    This serves as an example of how to implement an environment
    and is suitable for unit tests and simple simulations.
    """

    def __init__(
        self,
        locations: Dict[str, NDArray[np.floating[Any]]],
        objects_at_locations: Dict[str, Dict[str, Set[str]]],
        robot_locations: Dict[str, str],
    ) -> None:
        """Initialize the simple environment.

        Args:
            locations: Dictionary mapping location names to coordinates.
            objects_at_locations: Ground truth objects at each location.
            robot_locations: Dictionary mapping robot names to their initial location names.
        """
        super().__init__()

        self.locations: Dict[str, NDArray[np.floating[Any]]] = locations.copy()
        self._ground_truth = objects_at_locations
        self._objects_at_locations: Dict[str, Dict[str, Set[str]]] = {
            loc: {"object": set()} for loc in locations
        }
        self.robots: Dict[str, SimulatedRobot] = {
            r_name: SimulatedRobot(name=r_name, pose=locations[f"{r_loc}"].copy())
            for r_name, r_loc in robot_locations.items()
        }
        self.robot_skill_start_time_and_duration: Dict[str, Tuple[float, Optional[float]]] = {
            robot_name: (0.0, None) for robot_name in self.robots.keys()
        }
        self.min_time: Optional[float] = None

    def get_objects_at_location(self, location: str) -> Dict[str, Set[str]]:
        """Return objects at a location (simulates perception)."""
        objects_found = self._ground_truth.get(location, {}).copy()
        # Update internal knowledge
        if "object" in objects_found:
            for obj in objects_found["object"]:
                self.add_object_at_location(obj, location)
        return objects_found

    def _get_move_cost_fn(self) -> Callable[[str, str, str], float]:
        """Return a function that computes movement time between locations."""

        def get_move_time(robot: str, loc_from: str, loc_to: str) -> float:
            distance: float = float(np.linalg.norm(self.locations[loc_from] - self.locations[loc_to]))
            return distance  # 1 unit of distance = 1 second

        return get_move_time

    def _get_intermediate_coordinates(
        self,
        time: float,
        coord_from: NDArray[np.floating[Any]],
        coord_to: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Compute intermediate position during movement (for visualization)."""
        dist: float = float(np.linalg.norm(coord_to - coord_from))
        if dist < 0.01:
            return coord_to
        elif time > dist:
            return coord_to
        direction = (coord_to - coord_from) / dist
        new_coord: NDArray[np.floating[Any]] = coord_from + direction * time
        return new_coord

    def remove_object_from_location(
        self, obj: str, location: str, object_type: str = "object"
    ) -> None:
        """Remove an object from a location (e.g., when picked up)."""
        self._objects_at_locations[location][object_type].discard(obj)
        # Also update ground truth
        if location in self._ground_truth and object_type in self._ground_truth[location]:
            self._ground_truth[location][object_type].discard(obj)

    def add_object_at_location(
        self, obj: str, location: str, object_type: str = "object"
    ) -> None:
        """Add an object to a location (e.g., when placed down)."""
        self._objects_at_locations[location][object_type].add(obj)
        # Also update ground truth
        if location not in self._ground_truth:
            self._ground_truth[location] = {}
        if object_type not in self._ground_truth[location]:
            self._ground_truth[location][object_type] = set()
        self._ground_truth[location][object_type].add(obj)

    def execute_skill(self, robot_name: str, skill_name: str, *args: Any, **kwargs: Any) -> None:
        """Execute a skill on a robot."""
        if skill_name == "move":
            loc_from = args[0]
            loc_to = args[1]
            target_coords = self.locations[loc_to]
            self.robots[robot_name].move(target_coords)

            # Keep track of move start time and duration
            move_cost_fn = self._get_move_cost_fn()
            move_time = move_cost_fn(robot_name, loc_from, loc_to) / 1.0  # robot velocity = 1.0
            self.robot_skill_start_time_and_duration[robot_name] = (self.time, move_time)

        elif skill_name in ["pick", "place", "search", "no_op"]:
            getattr(self.robots[robot_name], skill_name)()

            # Keep track of skill start time and duration
            skill_time = self.get_skills_time_fn(skill_name)(robot_name, *args, **kwargs)
            self.robot_skill_start_time_and_duration[robot_name] = (self.time, skill_time)
        else:
            raise ValueError(f"Skill '{skill_name}' not defined for robot '{robot_name}'.")

    def get_skills_time_fn(self, skill_name: str) -> Callable[..., float]:
        """Get a time function for a skill."""
        if skill_name == "move":
            return self._get_move_cost_fn()
        else:
            # Define fixed times for other skills
            skills_times: Dict[str, float] = {
                "pick": 5.0,
                "place": 5.0,
                "search": 5.0,
                "no_op": 5.0,
            }

            def get_skill_time(robot_name: str, *args: Any, **kwargs: Any) -> float:
                return skills_times[skill_name]

            return get_skill_time

    def stop_robot(self, robot_name: str) -> None:
        """Stop a robot's current action."""
        robot = self.robots[robot_name]
        if robot.current_action_name == "move":
            assert self.min_time is not None
            assert robot.pose is not None
            assert robot.target_pose is not None
            # Convert to arrays for intermediate coordinate calculation
            pose_arr = np.asarray(robot.pose)
            target_arr = np.asarray(robot.target_pose)
            robot_pose = self._get_intermediate_coordinates(self.min_time, pose_arr, target_arr)
            self.locations[f"{robot_name}_loc"] = robot_pose
            robot.pose = robot_pose

        self.robots[robot_name].stop()
        self.robot_skill_start_time_and_duration[robot_name] = (0.0, None)

    def get_robot_that_finishes_first_and_when(self) -> Tuple[List[str], float]:
        """Get the robot(s) that will finish their current action first."""
        robots_progress = np.array(
            [self.time - start_time for start_time, _ in self.robot_skill_start_time_and_duration.values()]
        )
        time_to_target = [(r_name, tc) for r_name, (_, tc) in self.robot_skill_start_time_and_duration.items()]

        remaining_times = [(r_name, t - p) for (r_name, t), p in zip(time_to_target, robots_progress) if t is not None]
        _, min_time = min(remaining_times, key=lambda x: x[1])
        min_robots = [n for n, t in remaining_times if t == min_time]
        return min_robots, min_time

    def get_executed_skill_status(self, robot_name: str, skill_name: str) -> SkillStatus:
        """Get the execution status of a skill."""
        if skill_name not in ["move", "pick", "place", "search", "no_op"]:
            print(f"Action: '{skill_name}' not verified in Simulation!")

        # For simulation we do the following:
        # If all robots are not assigned, return IDLE
        # If some robots are assigned, but this robot is not the one finishing first, return RUNNING
        # If this robot is among the ones finishing first, return DONE
        all_robots_assigned = all(not r.is_free for r in self.robots.values())
        if not all_robots_assigned:
            return SkillStatus.IDLE
        min_robots, self.min_time = self.get_robot_that_finishes_first_and_when()
        if robot_name not in min_robots:
            return SkillStatus.RUNNING
        return SkillStatus.DONE
