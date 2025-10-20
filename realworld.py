import itertools
import roslibpy
import environments
from mrppddl.core import Fluent as F, State, get_action_by_name
from mrppddl.helper import construct_move_visited_operator
from mrppddl.core import transition

from mrppddl.core import OptCallable, Operator, Effect
from mrppddl.helper import _make_callable, _invert_prob

from typing import Dict, Set, List, Tuple, Callable


class RealEnvironment(environments.BaseEnvironment):
    def __init__(self, client):
        super().__init__()
        self._get_locations_service = roslibpy.Service(client, '/get_locations', 'GetLocations')
        self._move_robot_service = roslibpy.Service(client, '/move_robot', 'MoveRobot')
        self._get_distance_service = roslibpy.Service(client, '/get_distance', 'GetDistance')

        self.locations = self._get_locations()

    def get_move_cost_fn(self):
        locations = set(self.locations) - {'r1_loc', 'r2_loc'}
        location_distances = {}
        for loc1, loc2 in itertools.combinations(locations, 2):
            distance = self._get_distance(loc1, loc2)
            location_distances[frozenset([loc1, loc2])] = distance

        def get_move_time(robot, loc_from, loc_to):
            if frozenset([loc_from, loc_to]) in location_distances:
                return location_distances[frozenset([loc_from, loc_to])]
            distance = self._get_distance(loc_from, loc_to)
            return distance
        return get_move_time

    def _get_distance(self, location1, location2):
        request = roslibpy.ServiceRequest({'location1': location1, 'location2': location2})
        result = self._get_distance_service.call(request)
        return result['distance']

    def _get_locations(self):
        request = roslibpy.ServiceRequest()
        result = self._get_locations_service.call(request)
        return result['locations']

    def move_robot(self, robot_name, location):
        print("Moving robot", robot_name, "to", location)
        request = roslibpy.ServiceRequest({'robot_name': robot_name, 'location': location})
        result = self._move_robot_service.call(request)
        return result['success']


class PlanningLoop():
    def __init__(self, initial_state, goal_fluents, num_robots=1):
        self.state = initial_state
        self.goal_fluents = goal_fluents
        self.num_robots = num_robots
        self.robot_assigned = [False] * self.num_robots

    @property
    def state(self):
        """The state is the internal state with future effects added."""
        effects = []
        for act in self.ongoing_actions:
            effects += act.upcoming_effects
        self.ongoing_actions = [
            act for act in self.ongoing_actions
            if not act.is_done
        ]
        return State(
            self._state.time,
            self._state.fluents,
            sorted(self._state.upcoming_effects + effects,
                   key=lambda el: el[0])
        )

    def advance(self, action):
        pass

    def goal_reached(self):
        if all(fluent in self.state.fluents for fluent in self.goal_fluents):
            return True
        return False




if __name__ == '__main__':

    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    env = RealEnvironment(client)

    objects_by_type = {
        "robot": {"r1", "r2"},
        "location": env.locations,
    }

    initial_state = State(
        time=0,
        fluents={
            F("at", "r1", "r1_loc"),
            F("at", "r2", "r2_loc"),
            F("free", "r1"),
            F("free", "r2"),
        },
    )

    move_op = construct_move_visited_operator(move_time=env.get_move_cost_fn())
    all_actions = move_op.instantiate(objects_by_type)

    mcts = MCTSPlanner(all_actions)
    goal_fluents = {F(f"found {env.target_object}")}

    for _ in range(5):
        action_name = mcts(initial_state, goal_fluents, max_iterations=1000, c=10)
        if action_name != 'NONE':
            action = get_action_by_name(all_actions, action_name)

    client.terminate()
