"""
Modern Real-world robot planning demonstration.

Uses PhysicalEnvironment for clean integration with ROS-based hardware.
Follows the Scene/Environment separation pattern.
"""

import itertools
import roslibpy
from typing import Any, Callable, Dict, List, Set, Optional, Type
from railroad.core import Fluent as F, State, Operator
from railroad.environment import PhysicalEnvironment, PhysicalScene, SkillStatus, ActiveSkill
from railroad.planner import MCTSPlanner
from railroad.operators import construct_move_visited_operator


STATUS_MAP = {
    'moving': SkillStatus.RUNNING,
    'reached': SkillStatus.DONE,
    'stopped': SkillStatus.IDLE
}


class ROSRealScene(PhysicalScene):
    """Example of a ROS-integrated PhysicalScene provider."""

    def __init__(self, client: roslibpy.Ros):
        self._get_locations_service = roslibpy.Service(client, '/get_locations', 'planner_msgs/GetLocations')
        self._get_distance_service = roslibpy.Service(client, '/get_distance', 'planner_msgs/GetDistance')

        self._location_names = self._get_locations()
        self._objects = set() # Would be populated from perception
        self._object_locations = {} # Would be populated from perception

    def _get_locations(self) -> List[str]:
        try:
            request = roslibpy.ServiceRequest()
            result = self._get_locations_service.call(request)
            return result['locations']
        except Exception:
            return ["kitchen", "bedroom", "r1_loc", "r2_loc"]

    def _get_distance(self, location1: str, location2: str) -> float:
        try:
            request = roslibpy.ServiceRequest({'to_name': location1, 'from_name': location2})
            result = self._get_distance_service.call(request)
            if result['ok']:
                return result['distance']
        except Exception:
            pass
        return 10.0 # Default

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

    # --- PhysicalEnvironment implementation ---

    def execute_skill(self, robot_name: str, skill_name: str, *args: Any, **kwargs: Any) -> None:
        if skill_name == "move":
            location = args[0]
            request = roslibpy.ServiceRequest({'robot_id': robot_name, 'target': location})
            self._move_robot_service.call(request)

    def get_executed_skill_status(self, robot_name: str, skill_name: str) -> SkillStatus:
        if skill_name == "move":
            try:
                request = roslibpy.ServiceRequest({'robot_id': robot_name})
                result = self._move_status_service.call(request)
                return STATUS_MAP.get(result['status'], SkillStatus.IDLE)
            except Exception:
                return SkillStatus.DONE
        return SkillStatus.DONE

    def stop_robot(self, robot_name: str) -> None:
        request = roslibpy.ServiceRequest({'robot_id': robot_name})
        self._stop_robot_service.call(request)


if __name__ == '__main__':
    # Initialize ROS client
    host = 'localhost'
    client = roslibpy.Ros(host=host, port=9090)
    client.run()

    # 1. Initialize Scene
    scene = ROSRealScene(client)

    # 2. Define initial state
    initial_state = State(
        time=0.0,
        fluents={
            F("at", "r1", "r1_loc"), F("at", "r2", "r2_loc"),
            F("free", "r1"), F("free", "r2"),
        },
    )

    # 3. Create operators using scene info
    move_op = construct_move_visited_operator(move_time=scene.get_move_cost_fn())
    operators = [move_op]

    # 4. Define objects by type
    objects_by_type = {
        "robot": {"r1", "r2"},
        "location": set(scene.locations.keys()),
    }

    # 5. Initialize Environment
    env = ROSRealEnvironment(scene, client, initial_state, objects_by_type, operators)

    # Goal: Visit some locations
    goal = F("visited", "kitchen") & F("visited", "bedroom")

    # Planning and execution loop
    all_actions = env.get_actions()
    planner = MCTSPlanner(all_actions)

    while not env.is_goal_reached(goal.fluents if hasattr(goal, "fluents") else [goal]):
        action_name = planner(env.state, goal, max_iterations=1000)
        if action_name == "NONE":
            print("No plan found!")
            break

        from railroad.core import get_action_by_name
        action = get_action_by_name(all_actions, action_name)
        print(f"Executing: {action_name}")
        env.act(action)

    client.terminate()
