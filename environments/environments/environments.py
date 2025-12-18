from typing import Dict, List, Tuple, Callable, Union
from enum import IntEnum
import numpy as np


class ActionStatus(IntEnum):
    IDLE = -1
    RUNNING = 0
    DONE = 1


class BaseEnvironment:
    '''Abstract class for all environments.'''
    def __init__(self):
        self.time = 0.0

    def get_move_cost_fn(self) -> Callable[[str, str, str], float]:
        raise NotImplementedError()

    def get_intermediate_coordinates(self, time, loc_from, loc_to) -> Union[List, Tuple]:
        raise NotImplementedError()

    def get_objects_at_location(self, location) -> Dict[str, set]:
        '''This is supposed to be a perception method that updates _objects_at_locations. In simulators, we get this
        from ground truth. In real robots, this would be replaced by a perception module.'''
        raise NotImplementedError()

    def remove_object_from_location(self, obj, location):
        raise NotImplementedError()

    def add_object_at_location(self, obj, location):
        raise NotImplementedError()

    def move_robot(self, robot_name: str, location: str):
        raise NotImplementedError()

    def pick_robot(self, robot_name: str):
        raise NotImplementedError()

    def place_robot(self, robot_name: str):
        raise NotImplementedError()

    def stop_robot(self, robot_name: str):
        raise NotImplementedError()

    def search_robot(self, robot_name: str):
        raise NotImplementedError()

    def get_action_status(self, robot_name: str, action_name: str) -> ActionStatus:
        raise NotImplementedError()


class SimpleEnvironment(BaseEnvironment):
    """Simple household environment for testing multi-object manipulation."""

    def __init__(self, locations, objects_at_locations, num_robots=1):
        super().__init__()
        SKILLS_TIME = {
            'robot1': {
                'pick': 5,
                'place': 5,
                'search': 5},
            'robot2': {
                'pick': 5,
                'place': 5,
                'search': 5}
        }
        self.locations = locations.copy()
        self._ground_truth = objects_at_locations
        self._objects_at_locations = {
            loc: {"object": set()} for loc in locations}
        self.robots = {
            f"robot{i + 1}": Robot(name=f"robot{i + 1}",
                                   pose=locations["living_room"].copy(),
                                   skills_time=SKILLS_TIME[f'robot{i + 1}']) for i in range(num_robots)
        }
        self.min_robot = None
        self.min_time = None

    def get_objects_at_location(self, location):
        """Return objects at a location (simulates perception)."""
        objects_found = self._ground_truth.get(location, {}).copy()
        # Update internal knowledge
        if "object" in objects_found:
            for obj in objects_found["object"]:
                self.add_object_at_location(obj, location)
        return objects_found

    def get_move_cost_fn(self):
        """Return a function that computes movement time between locations."""
        def get_move_time(robot, loc_from, loc_to):
            distance = np.linalg.norm(
                self.locations[loc_from] - self.locations[loc_to]
            )
            return distance  # 1 unit of distance = 1 second
        return get_move_time

    def get_intermediate_coordinates(self, time, loc_from, loc_to, is_coords=True):
        """Compute intermediate position during movement (for visualization)."""
        if not is_coords:
            coord_from = self.locations[loc_from]
            coord_to = self.locations[loc_to]
        else:
            coord_from = loc_from
            coord_to = loc_to
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

    def move_robot(self, robot_name, location):
        target_coords = self.locations[location]
        self.robots[robot_name].move(target_coords, self.time)

    def pick_robot(self, robot_name):
        self.robots[robot_name].pick(self.time)

    def place_robot(self, robot_name):
        self.robots[robot_name].place(self.time)

    def search_robot(self, robot_name):
        self.robots[robot_name].search(self.time)

    def stop_robot(self, robot_name):
        # If the robot was moving, it's now at a new intermediate location
        robot = self.robots[robot_name]
        if robot.current_action_name == 'move':
            robot_pose = self.get_intermediate_coordinates(
                self.min_time, robot.pose, robot.target_pose, is_coords=True)
            self.locations[f'{robot_name}_loc'] = robot_pose
            robot.pose = robot_pose

        self.robots[robot_name].stop()

    def no_op_robot(self, robot_name):
        self.robots[robot_name].no_op(self.time)

    def get_robot_that_finishes_first_and_when(self):
        robots_progress = np.array([self.time - r.start_time for r in self.robots.values()])
        time_to_target = [(n, r.time_to_completion) for n, r in self.robots.items()]

        remaining_times = [(n, t - p) for (n, t), p in zip(time_to_target, robots_progress)]
        min_robot, min_time = min(remaining_times, key=lambda x: x[1])
        return min_robot, min_time

    def _get_move_status(self, robot_name):
        all_robots_assigned = all(not r.is_free for r in self.robots.values())
        if not all_robots_assigned:
            return ActionStatus.IDLE
        self.min_robot, self.min_time = self.get_robot_that_finishes_first_and_when()
        if self.min_robot != robot_name:
            return ActionStatus.RUNNING
        # Otherwise, this robot has reached its target
        self.robots[robot_name].stop()
        return ActionStatus.DONE

    def _get_pick_place_search_status(self, robot_name, action_name):
        all_robots_assigned = all(not r.is_free for r in self.robots.values())
        if not all_robots_assigned:
            return ActionStatus.IDLE
        self.min_robot, self.min_time = self.get_robot_that_finishes_first_and_when()
        if self.min_robot != robot_name:
            return ActionStatus.RUNNING

        self.stop_robot(robot_name)
        return ActionStatus.DONE

    def get_action_status(self, robot_name, action_name):
        if action_name == 'move':
            return self._get_move_status(robot_name)
        if action_name in ['pick', 'place', 'search', 'no-op']:
            return self._get_pick_place_search_status(robot_name, action_name)
        raise ValueError(f"Unknown action name: {action_name}")


class Robot:
    def __init__(self, name: str, pose=None, skills_time: Dict[str, float] = None, robot_move_time_fn=None):
        self.name = name
        self.current_action_name = None
        self.pose = pose
        self.target_pose = None
        self.is_free = True
        self.skills_time = skills_time
        self.robot_move_time_fn = robot_move_time_fn
        self.start_time = 0.0
        self.time_to_completion = None
        self.robot_velocity = 1.0

    def __repr__(self):
        return f"Robot(name={self.name}, pose={self.pose})"

    def move(self, new_pose, start_time):
        self.current_action_name = 'move'
        self.is_free = False
        self.start_time = start_time
        self.target_pose = new_pose
        if not self.robot_move_time_fn:
            self.time_to_completion = np.linalg.norm(np.array(self.pose)[:2] - np.array(new_pose)[:2]) / self.robot_velocity
        else:
            self.time_to_completion = self.robot_move_time_fn(self.pose, new_pose) / self.robot_velocity

    def pick(self, start_time):
        self.current_action_name = 'pick'
        self.is_free = False
        self.start_time = start_time
        self.time_to_completion = self.skills_time['pick']

    def place(self, start_time):
        self.current_action_name = 'place'
        self.is_free = False
        self.start_time = start_time
        self.time_to_completion = self.skills_time['place']

    def search(self, start_time):
        self.current_action_name = 'search'
        self.is_free = False
        self.start_time = start_time
        self.time_to_completion = self.skills_time['search']

    def no_op(self, start_time):
        self.current_action_name = 'no-op'
        self.is_free = False
        self.start_time = start_time
        self.time_to_completion = 5.0  # TODO: change to skills time

    def stop(self):
        self.current_action_name = None
        self.is_free = True
        self.target_pose = None
        self.start_time = 0.0
        self.time_to_completion = None
