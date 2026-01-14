import threading
import functools
from pyrobosim.core.yaml_utils import WorldYamlLoader
from .environments import BaseEnvironment, SkillStatus
import time


def run_async(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(
            target=func,
            args=args,
            kwargs=kwargs,
            daemon=True
        )
        thread.start()
        return thread
    return wrapper


class PyRoboSimEnv(BaseEnvironment):
    def __init__(self, world_file: str):
        self.world = WorldYamlLoader().from_file(world_file)
        self.locations = self.world.get_location_names()
        self.robots = {robot.name: robot for robot in self.world.robots}
        self.is_robot_assigned = {robot: False for robot in self.robots}
        self.is_no_op_running = {robot: False for robot in self.robots}

    def get_objects_at_location(self, location):
        location = self.world.get_location_by_name(location)
        objects = {'object': set()}
        if location:
            for spawn in location.children:
                for obj in spawn.children:
                    objects['object'].add(obj.name)
        return objects

    def remove_object_from_location(self, obj, location):
        '''Remove an object from location is handled when a pick skill is executed.'''
        pass

    def add_object_at_location(self, obj, location):
        '''Add an object at location is handled when a place skill is executed.'''
        pass

    def execute_skill(self, robot_name: str, skill_name: str, *args, **kwargs):
        self.is_robot_assigned[robot_name] = True
        if skill_name == 'pick':
            object_name = args[1]
            self._pick(robot_name, object_name)
        elif skill_name == 'place':
            self._place(robot_name)
        elif skill_name == 'move':
            loc_to = args[1]
            self._move(robot_name, loc_to)
        elif skill_name == 'search':
            self._search(robot_name)
        elif skill_name == 'no_op':
            self._no_op(robot_name)
        else:
            raise ValueError(f"Skill '{skill_name}' not recognized.")
        time.sleep(0.1)  # Give some time for the skill to start

    def get_skills_time_fn(self, skill_name: str):
        if skill_name == 'move':
            return self._get_move_cost_fn()
        else:
            def get_skill_time(robot_name, *args, **kwargs):
                return 5.0  # TODO: Get skills time from pyrobosim
            return get_skill_time

    def _get_move_cost_fn(self):
        def get_move_time(robot, loc_from, loc_to):
            from_pose = self.world.get_location_by_name(loc_from).pose
            to_pose = self.world.get_location_by_name(loc_to).pose
            plan = self.robots[robot].path_planner.plan_path(from_pose, to_pose)
            if plan is None:
                return float('inf')
            return plan.length / 1.0  # robot velocity = 1.0
        return get_move_time

    def get_executed_skill_status(self, robot_name: str, skill_name: str):
        if not all(self.is_robot_assigned.values()):
            return SkillStatus.IDLE

        if skill_name == 'no_op' and self.is_no_op_running[robot_name]:
            return SkillStatus.RUNNING

        is_busy = self.robots[robot_name].is_busy()
        skill_status = SkillStatus.RUNNING if is_busy else SkillStatus.DONE
        return skill_status

    def stop_robot(self, robot_name: str):
        self.is_robot_assigned[robot_name] = False
        self.robots[robot_name].cancel_actions()

    @run_async
    def _pick(self, robot_name, object_name):
        self.robots[robot_name].pick_object(object_name)

    @run_async
    def _place(self, robot_name):
        self.robots[robot_name].place_object()

    @run_async
    def _move(self, robot_name, loc_to):
        self.robots[robot_name].navigate(goal=loc_to)

    @run_async
    def _search(self, robot_name):
        self.robots[robot_name].detect_objects()

    @run_async
    def _no_op(self, robot_name):
        self.is_no_op_running[robot_name] = True
        time.sleep(self.get_skills_time_fn('no_op')(robot_name))
        self.is_no_op_running[robot_name] = False
