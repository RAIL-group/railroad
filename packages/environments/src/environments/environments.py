from typing import Dict, Callable
from enum import IntEnum
import numpy as np


class SkillStatus(IntEnum):
    IDLE = -1
    RUNNING = 0
    DONE = 1


class BaseEnvironment:
    '''Abstract class for all environments.'''
    def __init__(self):
        self.time = 0.0

    def get_objects_at_location(self, location) -> Dict[str, set]:
        '''This is supposed to be a perception method that updates _objects_at_locations. In simulators, we get this
        from ground truth. In real robots, this would be replaced by a perception module.'''
        raise NotImplementedError()

    def remove_object_from_location(self, obj, location):
        raise NotImplementedError()

    def add_object_at_location(self, obj, location):
        raise NotImplementedError()

    def execute_skill(self, robot_name: str, skill_name: str, *args, **kwargs):
        raise NotImplementedError()

    def get_skills_time_fn(self, skill_name: str) -> Callable[[str, str, str], float]:
        raise NotImplementedError()

    def get_executed_skill_status(self, robot_name: str, skill_name: str) -> SkillStatus:
        raise NotImplementedError()

    def stop_robot(self, robot_name: str):
        raise NotImplementedError()


class SimpleEnvironment(BaseEnvironment):
    """Simple household environment for testing multi-object manipulation."""

    def __init__(self, locations, objects_at_locations, robot_locations):
        super().__init__()

        self.locations = locations.copy()
        self._ground_truth = objects_at_locations
        self._objects_at_locations = {
            loc: {"object": set()} for loc in locations}
        self.robots = {
            r_name: SimulatedRobot(name=r_name,
                                   pose=locations[f"{r_loc}"].copy())
            for r_name, r_loc in robot_locations.items()
        }
        self.robot_skill_start_time_and_duration = {robot_name: (0, None) for robot_name in self.robots.keys()}
        self.min_time = None  # TODO: change name

    def get_objects_at_location(self, location):
        """Return objects at a location (simulates perception)."""
        objects_found = self._ground_truth.get(location, {}).copy()
        # Update internal knowledge
        if "object" in objects_found:
            for obj in objects_found["object"]:
                self.add_object_at_location(obj, location)
        return objects_found

    def _get_move_cost_fn(self):
        """Return a function that computes movement time between locations."""
        def get_move_time(robot, loc_from, loc_to):
            distance = np.linalg.norm(
                self.locations[loc_from] - self.locations[loc_to]
            )
            return distance  # 1 unit of distance = 1 second
        return get_move_time

    def _get_intermediate_coordinates(self, time, coord_from, coord_to):
        """Compute intermediate position during movement (for visualization)."""
        dist = np.linalg.norm(coord_to - coord_from)
        if dist < 0.01:
            return coord_to
        elif time > dist:
            return coord_to
        direction = (coord_to - coord_from) / dist
        new_coord = coord_from + direction * time
        return new_coord

    def remove_object_from_location(self, obj, location, object_type="object"):
        """Remove an object from a location (e.g., when picked up)."""
        self._objects_at_locations[location][object_type].discard(obj)
        # Also update ground truth
        if location in self._ground_truth and object_type in self._ground_truth[location]:
            self._ground_truth[location][object_type].discard(obj)

    def add_object_at_location(self, obj, location, object_type="object"):
        """Add an object to a location (e.g., when placed down)."""
        self._objects_at_locations[location][object_type].add(obj)
        # Also update ground truth
        if location not in self._ground_truth:
            self._ground_truth[location] = {}
        if object_type not in self._ground_truth[location]:
            self._ground_truth[location][object_type] = set()
        self._ground_truth[location][object_type].add(obj)

    def execute_skill(self, robot_name, skill_name, *args, **kwargs):
        if skill_name == 'move':
            loc_from = args[0]
            loc_to = args[1]
            target_coords = self.locations[loc_to]
            self.robots[robot_name].move(target_coords)

            # Keep track of move start time and duration
            move_cost_fn = self._get_move_cost_fn()
            move_time = move_cost_fn(robot_name, loc_from, loc_to) / 1.0  # robot velocity = 1.0
            self.robot_skill_start_time_and_duration[robot_name] = (self.time, move_time)

        elif skill_name in ['pick', 'place', 'search', 'no_op']:
            getattr(self.robots[robot_name], skill_name)()

            # Keep track of skill start time and duration
            skill_time = self.get_skills_cost_fn(skill_name)(robot_name, *args, **kwargs)
            self.robot_skill_start_time_and_duration[robot_name] = (self.time, skill_time)
        else:
            raise ValueError(f"Skill '{skill_name}' not defined for robot '{robot_name}'.")

    def get_skills_cost_fn(self, skill_name):
        if skill_name == 'move':
            return self._get_move_cost_fn()
        else:
            # Define fixed times for other skills
            skills_costs = {
                'pick': 5.0,
                'place': 5.0,
                'search': 5.0,
                'no_op': 5.0,
            }

            def get_skill_time(robot_name, *args, **kwargs):
                return skills_costs[skill_name]
            return get_skill_time

    def stop_robot(self, robot_name):
        # If the robot was moving, it's now at a new intermediate location
        robot = self.robots[robot_name]
        if robot.current_action_name == 'move':
            robot_pose = self._get_intermediate_coordinates(
                self.min_time, robot.pose, robot.target_pose)
            self.locations[f'{robot_name}_loc'] = robot_pose
            robot.pose = robot_pose

        self.robots[robot_name].stop()
        self.robot_skill_start_time_and_duration[robot_name] = (0, None)

    def get_robot_that_finishes_first_and_when(self):
        robots_progress = np.array(
            [self.time - start_time for start_time, _ in self.robot_skill_start_time_and_duration.values()])
        time_to_target = [(r_name, tc) for r_name, (_, tc) in self.robot_skill_start_time_and_duration.items()]

        remaining_times = [(r_name, t - p) for (r_name, t), p in zip(time_to_target, robots_progress)]
        _, min_time = min(remaining_times, key=lambda x: x[1])
        min_robots = [n for n, t in remaining_times if t == min_time]
        return min_robots, min_time

    def get_executed_skill_status(self, robot_name: str, skill_name: str) -> SkillStatus:
        if skill_name not in ['move', 'pick', 'place', 'search', 'no_op']:
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


class SimulatedRobot:
    def __init__(self, name: str, pose=None):
        self.name = name
        self.current_action_name = None
        self.pose = pose
        self.target_pose = None
        self.is_free = True

    def __repr__(self):
        return f"SimulatedRobot(name={self.name}, pose={self.pose})"

    def move(self, new_pose):
        self.current_action_name = 'move'
        self.is_free = False
        self.target_pose = new_pose

    def pick(self):
        self.current_action_name = 'pick'
        self.is_free = False

    def place(self):
        self.current_action_name = 'place'
        self.is_free = False

    def search(self):
        self.current_action_name = 'search'
        self.is_free = False

    def no_op(self):
        self.current_action_name = 'no_op'
        self.is_free = False

    def stop(self):
        self.current_action_name = None
        self.is_free = True
        self.target_pose = None
