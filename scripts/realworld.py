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
import argparse

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
        locations = set(self.locations) - {'v4w1_loc', 'v4w2_loc'}
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
        if location2 == 'v4w1_loc' or location2 == 'v4w2_loc':
            location1, location2 = location2, location1
        if location1.endswith('_loc') and location2.endswith('_loc'):
            return float('inf')
        request = roslibpy.ServiceRequest({'to_name': location2, 'from_name': location1})
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
    parser = argparse.ArgumentParser(description='Real-world robot planning demonstration.')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='ROS bridge host')
    parser.add_argument('--port', type=int, default=9090, help='ROS bridge port')
    args = parser.parse_args()

    print(f"Connecting to ROS bridge at {args.host}:{args.port}...")
    client = roslibpy.Ros(host=args.host, port=args.port)
    client.run()
    env = RealEnvironment(client)
    print("Locations in environment:", env.locations)
    # test move cost fn:
    robot_locations = {"v4w1": "v4w1_loc", "v4w2": "v4w2_loc"}
    move_cost_fn = env.get_move_cost_fn()
    for loc1, loc2 in itertools.combinations(env.locations, 2):
        print(f"Distance from {loc1} to {loc2}: {move_cost_fn('v4w1', loc1, loc2)}")
    for robot_name, r_loc in robot_locations.items():
        for loc in env.locations:
            if loc != r_loc:
                print(f"Computing distance from {r_loc} to {loc}...")
                print(f"Distance from {r_loc} to {loc}: {move_cost_fn(robot_name, r_loc, loc)}")

    objects_by_type = {
        "robot": robot_locations.keys(),
        "location": env.locations,
    }
    initial_state = State(
        time=0.0,
        fluents={
            F("at", "v4w1", "v4w1_loc"), F('visited', 'v4w1_loc'),
            F("at", "v4w2", "v4w2_loc"), F('visited', 'v4w2_loc'),
            F("free", "v4w1"),
            F("free", "v4w2"),
        },
    )

    move_op = construct_move_visited_operator(move_time=env.get_move_cost_fn())
    planning_loop = PlanningLoop(initial_state, objects_by_type, [move_op], env)
    # Goal: Visit all target locations
    # Using Goal API: reduce(and_, [...]) creates an AndGoal
    # goal = reduce(and_, [F("visited t1"), F("visited t2"), F("visited t3")])
    goal = reduce(and_, (F("visited", loc) for loc in env.locations if loc not in {"v4w1_loc", "v4w2_loc"}))
    print(f"Goal: {goal}")

    actions_taken = []
    input("Press Enter to start planning...")
    for _ in range(10):
        if goal.evaluate(planning_loop.state.fluents):
            print("Goal reached!")
            break

        all_actions = planning_loop.get_actions()
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(planning_loop.state, goal, max_iterations=30000, c=100)
        if action_name != 'NONE':
            action = get_action_by_name(all_actions, action_name)
            print(action_name)
            planning_loop.advance(action, do_interrupt=False)
            print(planning_loop.state.fluents)
            actions_taken.append(action_name)
        else:
            print("No action.")

    client.terminate()
