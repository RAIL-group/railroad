"""
Real-world robot planning demonstration.

railroad plans; ROS robots (through rosbridge) act. As for ProcTHOR and
PyRoboSim, a scene describes the world and an environment acts in it:

- `RealScene` asks ROS what planning needs: the location names and the move
  times between them.
- `RealEnvironment` is a PhysicalEnvironment: besides its operators, it has
  execute_skill(), get_skill_status() and stop_robot().

The planning loop is written like the example scripts.
"""

from functools import reduce
from operator import and_

import itertools
import roslibpy
from railroad.core import Fluent as F, State, get_action_by_name


from railroad.dashboard import PlannerDashboard
from railroad.planner import MCTSPlanner
from railroad.environment import PhysicalEnvironment, SkillStatus
from railroad.operators import construct_move_visited_operator




# 'stopped' is the robot's initial state, with nothing assigned or running.
STATUS_MAP = {'moving': SkillStatus.RUNNING, 'reached': SkillStatus.DONE, 'stopped': SkillStatus.IDLE}


class RealScene:
    """The world as ROS describes it: location names and move times."""

    def __init__(self, client):
        self._get_locations_service = roslibpy.Service(client, '/get_locations', 'planner_msgs/GetLocations')
        self._get_distance_service = roslibpy.Service(client, '/get_distance', 'planner_msgs/GetDistance')
        # Names only: ROS gives no coordinates (``locations`` would be read as them).
        self.location_names = self._get_locations()

    def get_move_cost_fn(self):
        locations = set(self.location_names) - {'r1_loc', 'r2_loc'}
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


class RealEnvironment(PhysicalEnvironment):
    """Runs the scene's actions on the ROS robots."""

    def __init__(self, scene, client, state, objects_by_type):
        self.scene = scene
        self._move_robot_service = roslibpy.Service(client, '/move_robot', 'planner_msgs/MoveRobot')
        self._move_status_service = roslibpy.Service(client, '/get_move_status', 'planner_msgs/MoveStatus')
        self._stop_robot_service = roslibpy.Service(client, '/stop_robot', 'planner_msgs/StopRobot')
        super().__init__(state=state, objects_by_type=objects_by_type)

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

    def stop_robot(self, robot):
        print("Stopping robot", robot)
        request = roslibpy.ServiceRequest({'robot_id': robot})
        result = self._stop_robot_service.call(request)
        return result['stopped']

    # Start the skill and return; railroad polls get_skill_status() for the outcome.
    def execute_skill(self, robot, skill, *args):
        if skill == 'move':
            loc_from, loc_to = args
            self.move_robot(robot, loc_to)

    def get_skill_status(self, robot, skill):
        if skill == 'move':
            return self._get_move_status(robot)
        return SkillStatus.DONE  # skills the robots don't run complete at once

    def define_operators(self):
        return [construct_move_visited_operator(move_time=self.scene.get_move_cost_fn())]


def main(host: str = '192.168.0.16', port: int = 9090) -> None:
    # host = 'localhost'
    client = roslibpy.Ros(host=host, port=port)
    client.run()

    scene = RealScene(client)

    robot_locations = {"r1": "r1_loc", "r2": "r2_loc"}
    objects_by_type = {
        "robot": set(robot_locations.keys()),
        # "robot": {"r1"},
        "location": set(scene.location_names),
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

    env = RealEnvironment(scene, client, initial_state, objects_by_type)
    # Goal: Visit all target locations
    # Using Goal API: reduce(and_, [...]) creates an AndGoal
    goal = reduce(and_, [F("visited t1"), F("visited t2"), F("visited t3")])

    # Planning loop
    with PlannerDashboard(goal, env) as dashboard:
        for _ in range(1000):
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]Goal reached![/green]")
                break

            all_actions = env.get_actions()
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(env.state, goal, max_iterations=20000, c=10)

            if action_name == 'NONE':
                dashboard.console.print("No more actions available. Goal may not be achievable.")
                break

            action = get_action_by_name(all_actions, action_name)
            env.act(action)
            dashboard.update(mcts, action_name)

    client.terminate()


if __name__ == '__main__':
    main()
