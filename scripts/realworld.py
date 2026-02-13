"""
Real-world robot planning demonstration.

Uses the new Goal API for defining planning objectives.
"""

from functools import reduce
from operator import and_

import itertools
import roslibpy
from railroad.core import Fluent as F, State, get_action_by_name

from railroad.planner import MCTSPlanner
from railroad.experimental.environment import BaseEnvironment, SkillStatus, EnvironmentInterface as PlanningLoop
from railroad.operators import construct_move_visited_operator

STATUS_MAP = {'moving': SkillStatus.RUNNING, 'reached': SkillStatus.DONE, 'stopped': SkillStatus.IDLE}


class RealEnvironment(BaseEnvironment):
    def __init__(self, client):
        super().__init__()
        self._get_locations_service = roslibpy.Service(client, '/get_locations', 'planner_msgs/GetLocations')
        self._move_robot_service = roslibpy.Service(client, '/move_robot', 'planner_msgs/MoveRobot')
        self._get_distance_service = roslibpy.Service(client, '/get_distance', 'planner_msgs/GetDistance')
        self._move_status_service = roslibpy.Service(client, '/get_move_status', 'planner_msgs/MoveStatus')
        self._stop_robot_service = roslibpy.Service(client, '/stop_robot', 'planner_msgs/StopRobot')
        self.locations = self._get_locations()

    def get_objects_at_location(self, location: str):
        return {}

    def remove_object_from_location(self, obj: str, location: str) -> None:
        pass

    def add_object_at_location(self, obj: str, location: str) -> None:
        pass

    def execute_skill(self, robot_name: str, skill_name: str, *args, **kwargs) -> None:
        if skill_name == "move":
            self.move_robot(robot_name, args[1])

    def get_skills_time_fn(self, skill_name: str):
        if skill_name == "move":
            return self.get_move_cost_fn()
        return lambda *args, **kwargs: 1.0

    def get_executed_skill_status(self, robot_name: str, skill_name: str) -> SkillStatus:
        if skill_name == "move":
            return self._get_move_status(robot_name)
        return SkillStatus.DONE

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
        if location1 == 'r1_loc':
            location1, location2 = location2, location1
        request = roslibpy.ServiceRequest({'to_name': location1, 'from_name': location2})
        result = self._get_distance_service.call(request)
        if result['ok']:
            return result['distance']
        else:
            raise ValueError(result['message'])

    def _get_locations(self):
        request = roslibpy.ServiceRequest()
        result = self._get_locations_service.call(request)
        return result['locations']

    def move_robot(self, robot_name, location):
        request = roslibpy.ServiceRequest({'robot_id': robot_name, 'target': location})
        result = self._move_robot_service.call(request)
        if result['accepted']:
            return True
        else:
            raise ValueError(result['message'])

    def _get_move_status(self, robot_name):
        request = roslibpy.ServiceRequest({'robot_id': robot_name})
        result = self._move_status_service.call(request)
        return STATUS_MAP.get(result['status'], None)

    def stop_robot(self, robot_name):
        print("Stopping robot", robot_name)
        request = roslibpy.ServiceRequest({'robot_id': robot_name})
        result = self._stop_robot_service.call(request)
        return result['stopped']


if __name__ == '__main__':
    # host = 'localhost'
    host = '192.168.1.71'
    client = roslibpy.Ros(host=host, port=9090)
    client.run()
    env = RealEnvironment(client)
    print("Locations in environment:", env.locations)
    robot_locations = {"r1": "r1_loc", "r2": "r2_loc"}
    objects_by_type = {
        "robot": robot_locations.keys(),
        # "robot": {"r1"},
        "location": env.locations,
    }
    initial_state = State(
        time=0.0,
        fluents={
            F("at", "r1", "r1_loc"), F('visited', 'r1_loc'),
            F("at", "r2", "r2_loc"), F('visited', 'r2_loc'),
            F("free", "r1"),
            F("free", "r2"),
        },
    )

    move_op = construct_move_visited_operator(move_time=env.get_move_cost_fn())
    planning_loop = PlanningLoop(initial_state, objects_by_type, [move_op], env)
    # Goal: Visit all target locations
    # Using Goal API: reduce(and_, [...]) creates an AndGoal
    # goal = reduce(and_, [F("visited t1"), F("visited t2"), F("visited t3")])
    goal = reduce(and_, (F("visited", loc) for loc in env.locations if loc not in {"r1_loc", "r2_loc"}))
    print(f"Goal: {goal}")
    actions_taken = []
    for _ in range(10):
        if goal.evaluate(planning_loop.state.fluents):
            print("Goal reached!")
            break

        all_actions = planning_loop.get_actions()
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(planning_loop.state, goal, max_iterations=20000, c=10)
        if action_name != 'NONE':
            action = get_action_by_name(all_actions, action_name)
            print(action_name)
            planning_loop.advance(action)
            print(planning_loop.state.fluents)
            actions_taken.append(action_name)
        else:
            print("No action.")

    client.terminate()
