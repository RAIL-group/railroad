import numpy as np
from mrppddl.core import State, Fluent, transition, get_next_actions, get_action_by_name
from environment import get_location_object_likelihood

robot_from_to = lambda s: (s.split()[1], s.split()[2], s.split()[3])
class SymbolicToRealSimulator():
    def __init__(self, locations, robots, state, goal_fluents):
        self.robots = robots
        self.locations = locations
        self.state = state
        self.goal_fluents = goal_fluents
        self.map_robot_name = {r.name: r for r in robots}
        self.map_location_name = {loc.name: loc.location for loc in locations}

    def is_goal(self):
        return all(gf in self.state.fluents for gf in self.goal_fluents)

    def _get_robot_pose_from_symbolic(self, robot_symbolic):
        return self.map_robot_name[robot_symbolic].pose

    def _get_location_from_symbolic(self, location_symbolic):
        return self.map_location_name[location_symbolic]

    def get_move_cost(self, robot, from_loc, to_loc):
        robot_pose = self._get_robot_pose_from_symbolic(robot)
        location_from = self._get_location_from_symbolic(from_loc)
        location_to = self._get_location_from_symbolic(to_loc)
        return np.linalg.norm(np.array(location_from) - np.array(location_to))

    def get_likelihood_of_object(self, robot, location, object):
        return get_location_object_likelihood(location, object)

    def execute_action(self, action):
        if not self.state.satisfies_precondition(action):
            raise ValueError("Precondition not satisfied")

        self.state = transition(self.state, action)[0][0]
        robot_locations = self._process_fluents_to_get_robot_locations()

        self._move_robots(robot_locations)

        print(self.state)
        print("Fluents:", self.state.fluents)
        print("Upcoming Effects:", self.state.upcoming_effects)

    def _move_robots(self, robot_locations):
        for r_symb, l_symb in robot_locations:
            robot = self.map_robot_name[r_symb]
            location = self.map_location_name[l_symb]
            robot.move(location)

        # move robots
        for robot in self.robots:
            print(f'{robot.name}| PREV_POSE: {robot.prev_pose} | CURR_POSE: {robot.pose} | Net motion: {robot.net_motion}')

    def _process_fluents_to_get_robot_locations(self):
        robot_locations = []
        fluents = self.state.fluents
        for fluent in fluents:
            if fluent.name == 'at':
                robot_locations.append((fluent.args[0], fluent.args[1]))
        return robot_locations
