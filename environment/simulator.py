import numpy as np
from environment import get_location_object_likelihood

class SymbolicToRealSimulator():
    def __init__(self, locations, robots, state):
        self.robots = robots
        self.locations = locations
        self.state = state
        self.map_robot_location = {r.name: r.pose for r in robots}
        self.map_location_name = {loc.name: loc.location for loc in locations}

    def update_state(self, state):
        self.state = state
        # more complex than this, but fine for now

    def _get_robot_pose_from_symbolic(self, robot_symbolic):
        return self.map_robot_location[robot_symbolic]

    def _get_location_from_symbolic(self, location_symbolic):
        return self.map_location_name[location_symbolic]

    def get_move_cost(self, robot, from_loc):
        robot_pose = self._get_robot_pose_from_symbolic(robot)
        location_pose = self._get_location_from_symbolic(from_loc)
        return np.linalg.norm(np.array(robot_pose) - np.array(location_pose))

    def get_likelihood_of_object(self, robot, location, object):
        return get_location_object_likelihood(location, object)

    def execute_action(self, action):
        if not self.state.satisfies_precondition(action):
            raise ValueError("Precondition not satisfied")
        new_state = self.state.copy()

        pass
