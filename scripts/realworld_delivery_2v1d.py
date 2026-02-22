"""
Real-world Vertiwheeler Delivery demonstration.

Adapts the scenario from vertiwheeler_delivery.py to use ROS-based hardware/simulator.
Uses PhysicalEnvironment for multi-agent concurrency and wall-time tracking.
"""

import roslibpy
import time as time_module
import argparse
from typing import Any, Callable, Dict, List, Set, Optional, Type
from railroad.core import Fluent as F, State, Operator, Effect
from railroad.environment import PhysicalEnvironment, PhysicalScene, SkillStatus, ActiveSkill
from railroad.planner import MCTSPlanner
from railroad import operators

# Constants from vertiwheeler_delivery.py
MONITOR_TIME = 10.0
SKILLS_TIME = {
    "v4w1": {"pick": 2.0, "place": 2.0, "search": 5.0},
    "v4w2": {"pick": 2.0, "place": 2.0, "search": 5.0},
    "drone": {"monitor": MONITOR_TIME, "search": 5.0},
}
DRONE_SPEED_MULTIPLIER = 3.0

STATUS_MAP = {
    'moving': SkillStatus.RUNNING,
    'reached': SkillStatus.DONE,
    'stopped': SkillStatus.IDLE
}

ROBOTS = ['v4w1', 'drone']


class ROSRealScene(PhysicalScene):
    """ROS-integrated PhysicalScene provider."""

    def __init__(self, client: roslibpy.Ros):
        self._get_locations_service = roslibpy.Service(client, '/get_locations', 'planner_msgs/GetLocations')
        self._get_distance_service = roslibpy.Service(client, '/get_distance', 'planner_msgs/GetDistance')

        # Cache locations
        request = roslibpy.ServiceRequest()
        result = self._get_locations_service.call(request, timeout=10)
        self._location_names = result['locations']

        self._objects = {"supplies"}
        # Ground truth for simulation logic (supplies are at roomB)
        # self._object_locations = {"roomB": {"supplies"}}
        self._object_locations = {"t2": {"supplies"}}

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
        # For other skills, return fixed durations from SKILLS_TIME
        def get_skill_duration(robot: str, *args):
            return SKILLS_TIME.get(robot, {}).get(skill_name, float("inf"))
        return get_skill_duration

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
            # Assume drones move faster
            multiplier = DRONE_SPEED_MULTIPLIER if "drone" in robot else 1.0
            return self._get_distance(loc_from, loc_to) / multiplier
        return get_move_time


class ROSRealEnvironment(PhysicalEnvironment):
    """ROS-integrated PhysicalEnvironment with support for virtual skills."""

    def __init__(
        self,
        scene: ROSRealScene,
        client: roslibpy.Ros,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        operators: List[Operator],
    ):
        self._move_robot_service = roslibpy.Service(client, '/move_robot', 'planner_msgs/MoveRobot')
        self._move_status_service = roslibpy.Service(client, '/get_move_status', 'planner_msgs/MoveStatus')
        self._stop_robot_service = roslibpy.Service(client, '/stop_robot', 'planner_msgs/StopRobot')

        # Track start times for skills that aren't backed by ROS services
        self._virtual_skills_start: Dict[str, float] = {}

        super().__init__(
            scene=scene,
            state=state,
            objects_by_type=objects_by_type,
            operators=operators
        )

    def execute_skill(self, robot_name: str, skill_name: str, *args, **kwargs) -> None:
        if skill_name == "move":
            target_loc = args[1]
            print(f"[{robot_name}] Executing physical MOVE to {target_loc}")
            return self.move_robot(robot_name, target_loc)

        # Virtual skill execution: just record the wall time
        print(f"[{robot_name}] Starting virtual skill: {skill_name}")
        self._virtual_skills_start[robot_name] = time_module.time()

    def move_robot(self, robot_name, location):
        request = roslibpy.ServiceRequest({'robot_id': robot_name, 'target': location})
        result = self._move_robot_service.call(request, timeout=10)
        if not result['accepted']:
            raise ValueError(result['message'])
        return True

    def get_executed_skill_status(self, robot_name: str, skill_name: str) -> SkillStatus:
        if skill_name == "move":
            request = roslibpy.ServiceRequest({'robot_id': robot_name})
            result = self._move_status_service.call(request, timeout=10)
            return STATUS_MAP.get(result['status'], SkillStatus.IDLE)

        # Check virtual skill status based on duration
        if robot_name in self._virtual_skills_start:
            start_time = self._virtual_skills_start[robot_name]
            duration = SKILLS_TIME[robot_name][skill_name]
            if time_module.time() - start_time >= duration:
                print(f"[{robot_name}] Virtual skill {skill_name} DONE")
                del self._virtual_skills_start[robot_name]
                return SkillStatus.DONE
            # print(f"[{robot_name}] Virtual skill {skill_name} RUNNING")
            return SkillStatus.RUNNING

        return SkillStatus.IDLE

    def stop_robot(self, robot_name: str) -> None:
        if robot_name in self._virtual_skills_start:
            del self._virtual_skills_start[robot_name]

        request = roslibpy.ServiceRequest({'robot_id': robot_name})
        self._stop_robot_service.call(request, timeout=10)


def construct_monitor_operator(monitor_time: float):
    return Operator(
        name="monitor",
        parameters=[("?r", "robot"), ("?loc", "location")],
        preconditions=[F("at", "?r", "?loc"), F("free", "?r"), F("is_drone", "?r")],
        effects=[
            Effect(time=0, resulting_fluents={~F("free", "?r"), F("monitoring", "?loc")}),
            Effect(
                time=monitor_time,
                resulting_fluents={F("free", "?r"), ~F("monitoring", "?loc")}
            )
        ]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-world Vertiwheeler Delivery.')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='ROS bridge host')
    parser.add_argument('--port', type=int, default=9090, help='ROS bridge port')
    args = parser.parse_args()

    print(f"Connecting to ROS bridge at {args.host}:{args.port}...")
    client = roslibpy.Ros(host=args.host, port=args.port)
    client.run()

    scene = ROSRealScene(client)

    # Initial setup
    initial_fluents = {
        F("at", "drone", "drone_loc"),
        F("at", "v4w1", "v4w1_loc"),
        F("at", "v4w2", "v4w2_loc"),
        F("free", "drone"),
        F("free", "v4w1"),
        F("free", "v4w2"),
        F("is_drone", "drone"),
        F("is_vertiwheeler", "v4w1"),
        F("is_vertiwheeler", "v4w2"),
        F("revealed", "drone_loc"),
        F("revealed", "v4w1_loc"),
        F("revealed", "v4w2_loc"),
    }

    # Delivery Target: roomD. Must be monitored by drone while v4w1 places supplies.
    goal = (
        F("at", "supplies", "t3")
        & F("found", "supplies")
        & F("monitoring", "t3")
    )

    objects_by_type = {
        "robot": {"drone", "v4w1", "v4w2"},
        "location": set(scene.locations.keys()),
        "object": {"supplies"},
    }

    # Operators
    move_cost_fn = scene.get_move_cost_fn()

    # Generic move operator for both
    move_op = Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?r", "?from"), F("free", "?r")],
        effects=[
            Effect(time=0, resulting_fluents={~F("free", "?r"), ~F("at", "?r", "?from")}),
            Effect(
                time=(move_cost_fn, ["?r", "?from", "?to"]),
                resulting_fluents={F("free", "?r"), F("at", "?r", "?to")},
            ),
        ]
    )

    monitor_op = construct_monitor_operator(MONITOR_TIME)

    # Search probability logic
    def object_find_prob(robot: str, loc: str, obj: str) -> float:
        likelihoods = {
            "t1": 0.5,
            "t2": 0.9,
            "t3": 0.1,
            "v4w1_loc": 0.0,
            "v4w2_loc": 0.0,
            "drone_loc": 0.0,
        }
        return likelihoods[loc]

    search_op = operators.construct_search_operator(
        object_find_prob, scene.get_skills_time_fn("search")
    )

    pick_op_base = operators.construct_pick_operator_blocking(scene.get_skills_time_fn("pick"))
    pick_op = Operator(
        name=pick_op_base.name,
        parameters=pick_op_base.parameters,
        preconditions=pick_op_base.preconditions + [F("is_vertiwheeler", "?r")],
        effects=pick_op_base.effects
    )

    place_op_base = operators.construct_place_operator_blocking(scene.get_skills_time_fn("place"))
    # Vertiwheeler can only deliver if location is monitored
    place_op = Operator(
        name=place_op_base.name,
        parameters=place_op_base.parameters,
        preconditions=place_op_base.preconditions + [F("is_vertiwheeler", "?r"), F("monitoring", "?loc")],
        effects=place_op_base.effects
    )

    env = ROSRealEnvironment(
        scene=scene,
        client=client,
        state=State(0.0, initial_fluents, []),
        objects_by_type=objects_by_type,
        operators=[move_op, search_op, pick_op, monitor_op, place_op]
    )

    all_actions = env.get_actions()
    planner = MCTSPlanner(all_actions)

    input("Press enter to start planning...")
    for i in range(50):
        current_fluents = env.state.fluents
        if goal.evaluate(current_fluents):
            print("\nGoal reached!")
            break

        print(f"\n--- Iteration {i} ---")
        print(f"Time: {env.state.time:.2f}s")
        # Filter fluents for readability
        important = [
            str(f)
            for f in current_fluents
            if not f.negated and any(k in f.name for k in ["at", "holding", "found", "monitoring", "free"])
        ]
        print(f"Active Fluents: {important}")

        action_name = planner(env.state, goal, max_iterations=40000, c=300, max_depth=60)
        if action_name == "NONE":
            print("No action found")
            time_module.sleep(1)
            continue

        from railroad.core import get_action_by_name
        action = get_action_by_name(all_actions, action_name)
        print(f"Dispatching: {action_name}")
        env.act(action, do_interrupt=False)

    client.terminate()
