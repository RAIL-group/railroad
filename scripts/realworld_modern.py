"""
Modern Real-world robot planning demonstration.

Uses PhysicalEnvironment for clean integration with ROS-based hardware.
Follows the Scene/Environment separation pattern.
"""

import roslibpy
from typing import Any, Callable, Dict, List, Set, Optional, Type
from railroad.core import Fluent as F, State, Operator
from railroad.environment import PhysicalEnvironment, PhysicalScene, SkillStatus, ActiveSkill
from railroad.planner import MCTSPlanner
from railroad.operators import construct_move_visited_operator
import argparse


STATUS_MAP = {
    'moving': SkillStatus.RUNNING,
    'reached': SkillStatus.DONE,
    'stopped': SkillStatus.IDLE
}

ROBOTS = ['v4w1', 'drone1']


class ROSRealScene(PhysicalScene):
    """Example of a ROS-integrated PhysicalScene provider."""

    def __init__(self, client: roslibpy.Ros):
        self._get_locations_service = roslibpy.Service(client, '/get_locations', 'planner_msgs/GetLocations')
        self._get_distance_service = roslibpy.Service(client, '/get_distance', 'planner_msgs/GetDistance')

        self._location_names = self._get_locations()
        self._objects = set()
        self._object_locations = {}

    def _get_locations(self) -> List[str]:
        request = roslibpy.ServiceRequest()
        result = self._get_locations_service.call(request, timeout=10)
        return result['locations']

    def _get_distance(self, location1: str, location2: str) -> float:
        if location2.endswith('_loc'):
            location1, location2 = location2, location1
            if location2.endswith('_loc'):
                return float('inf')
        request = roslibpy.ServiceRequest({'to_name': location2, 'from_name': location1})
        result = self._get_distance_service.call(request, timeout=10)
        if result['ok']:
            return result['distance']
        else:
            raise ValueError(result['message'])

    def get_skills_time_fn(self, skill_name: str):
        if skill_name == "move":
            return self.get_move_cost_fn()
        raise NotImplementedError(f"Skill '{skill_name}' time function not implemented in ROSRealScene.")

    @property
    def locations(self) -> Dict[str, Any]:
        return {name: None for name in self._location_names}

    @property
    def objects(self) -> Set[str]:
        return self._objects

    @property
    def object_locations(self) -> Dict[str, Set[str]]:
        return self._object_locations

    def get_move_cost_fn(self) -> Callable[[str, str, str], float]:
        def get_move_time(robot: str, loc_from: str, loc_to: str) -> float:
            return self._get_distance(loc_from, loc_to)
        return get_move_time


class ROSRealEnvironment(PhysicalEnvironment):
    """Example of a ROS-integrated PhysicalEnvironment."""

    def __init__(
        self,
        scene: ROSRealScene,
        client: roslibpy.Ros,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        operators: List[Operator],
        skill_overrides: Optional[Dict[str, Type[ActiveSkill]]] = None,
    ):
        self._move_robot_service = roslibpy.Service(client, '/move_robot', 'planner_msgs/MoveRobot')
        self._move_status_service = roslibpy.Service(client, '/get_move_status', 'planner_msgs/MoveStatus')
        self._stop_robot_service = roslibpy.Service(client, '/stop_robot', 'planner_msgs/StopRobot')

        super().__init__(
            scene=scene,
            state=state,
            objects_by_type=objects_by_type,
            operators=operators,
            skill_overrides=skill_overrides
        )

    def execute_skill(self, robot_name: str, skill_name: str, *args, **kwargs) -> None:
        if skill_name == "move":
            result = self.move_robot(robot_name, args[1])
            return result
        raise NotImplementedError(f"Skill '{skill_name}' not implemented in ROSRealEnvironment.")

    def move_robot(self, robot_name, location):
        request = roslibpy.ServiceRequest({'robot_id': robot_name, 'target': location})
        result = self._move_robot_service.call(request, timeout=10)
        if result['accepted']:
            return True
        else:
            raise ValueError(result['message'])

    def get_executed_skill_status(self, robot_name: str, skill_name: str) -> SkillStatus:
        if skill_name == "move":
            return self._get_move_status(robot_name)
        raise NotImplementedError(f"Skill '{skill_name}' status retrieval not implemented in ROSRealEnvironment.")

    def _get_move_status(self, robot_name):
        request = roslibpy.ServiceRequest({'robot_id': robot_name})
        result = self._move_status_service.call(request, timeout=10)
        return STATUS_MAP.get(result['status'], SkillStatus.IDLE)

    def stop_robot(self, robot_name: str) -> None:
        request = roslibpy.ServiceRequest({'robot_id': robot_name})
        self._stop_robot_service.call(request, timeout=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-world robot planning demonstration.')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='ROS bridge host')
    parser.add_argument('--port', type=int, default=9090, help='ROS bridge port')
    args = parser.parse_args()

    print(f"Connecting to ROS bridge at {args.host}:{args.port}...")
    client = roslibpy.Ros(host=args.host, port=args.port)
    client.run()

    scene = ROSRealScene(client)

    initial_state = State(
        time=0.0,
        fluents={
            F("at", f"{ROBOTS[0]}", f"{ROBOTS[0]}_loc"), F("at", f"{ROBOTS[1]}", f"{ROBOTS[1]}_loc"),
            F("free", f"{ROBOTS[0]}"), F("free", f"{ROBOTS[1]}"),
        },
    )

    move_op = construct_move_visited_operator(move_time=scene.get_move_cost_fn())
    operators = [move_op]

    objects_by_type = {
        "robot": set(ROBOTS),
        "location": set(scene.locations.keys()),
    }

    env = ROSRealEnvironment(scene, client, initial_state, objects_by_type, operators)

    goal = F("visited", "roomB") & F("visited", "roomD")

    all_actions = env.get_actions()
    planner = MCTSPlanner(all_actions)

    print("\nStarting planning loop...")
    for i in range(20):
        if goal.evaluate(env.state.fluents):
            print("\nGoal reached!")
            break

        print(f"\n--- Iteration {i} ---")
        print(f"Time: {env.state.time:.2f}s")
        print(f"Active Fluents: {[str(f) for f in env.state.fluents if not f.negated]}")

        action_name = planner(env.state, goal, max_iterations=10000)
        if action_name == "NONE":
            print("No action")
            break

        from railroad.core import get_action_by_name
        action = get_action_by_name(all_actions, action_name)
        print(f"Dispatching: {action_name}")

        env.act(action, do_interrupt=False)

    client.terminate()
