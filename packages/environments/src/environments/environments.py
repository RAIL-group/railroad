"""Environment implementations for robot simulation.

This module provides concrete environment implementations.
"""

from typing import Any, Callable, Dict, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Import base classes from railroad.environment
from railroad.environment import BaseEnvironment, SkillStatus, SimulatedRobot, Pose

# Re-export for backward compatibility
__all__ = ["BaseEnvironment", "SkillStatus", "SimulatedRobot", "SimpleEnvironment", "Pose"]


class SimpleEnvironment(BaseEnvironment):
    """Simple household environment for testing multi-object manipulation."""

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
        self.robot_skill_start_time_and_duration: Dict[str, tuple[float, float | None]] = {
            robot_name: (0.0, None) for robot_name in self.robots.keys()
        }
        self.min_time: float | None = None

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
            skill_time = self.get_skills_cost_fn(skill_name)(robot_name, *args, **kwargs)
            self.robot_skill_start_time_and_duration[robot_name] = (self.time, skill_time)
        else:
            raise ValueError(f"Skill '{skill_name}' not defined for robot '{robot_name}'.")

    def get_skills_cost_fn(self, skill_name: str) -> Callable[..., float]:
        if skill_name == "move":
            return self._get_move_cost_fn()
        else:
            # Define fixed times for other skills
            skills_costs: Dict[str, float] = {
                "pick": 5.0,
                "place": 5.0,
                "search": 5.0,
                "no_op": 5.0,
            }

            def get_skill_time(robot_name: str, *args: Any, **kwargs: Any) -> float:
                return skills_costs[skill_name]

            return get_skill_time

    # Alias for backward compatibility
    get_skills_time_fn = get_skills_cost_fn

    def stop_robot(self, robot_name: str) -> None:
        # If the robot was moving, it's now at a new intermediate location
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

    def get_robot_that_finishes_first_and_when(self) -> tuple[list[str], float]:
        robots_progress = np.array(
            [self.time - start_time for start_time, _ in self.robot_skill_start_time_and_duration.values()]
        )
        time_to_target = [(r_name, tc) for r_name, (_, tc) in self.robot_skill_start_time_and_duration.items()]

        remaining_times = [(r_name, t - p) for (r_name, t), p in zip(time_to_target, robots_progress) if t is not None]
        _, min_time = min(remaining_times, key=lambda x: x[1])
        min_robots = [n for n, t in remaining_times if t == min_time]
        return min_robots, min_time

    def get_executed_skill_status(self, robot_name: str, skill_name: str) -> SkillStatus:
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
