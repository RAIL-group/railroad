from pyrobosim.core.world import World
from pyrobosim.core.yaml_utils import WorldYamlLoader
from .environments import BaseEnvironment


class PyRoboSimEnv(BaseEnvironment):
    def __init__(self, world_file: str):
        self.world = WorldYamlLoader().from_file(world_file)
        self.locations = self.world.get_location_names()
        self.robots = {robot.name: robot for robot in self.world.robots}

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
        if skill_name == 'pick':
            location_name = args[0]
            object_name = args[1]
            return self._pick(robot_name, location_name, object_name, **kwargs)
        elif skill_name == 'place':
            location_name = args[0]
            object_name = args[1]
            return self._place(robot_name, location_name, object_name, **kwargs)
        elif skill_name == 'move':
            loc_from_name = args[0]
            loc_to_name = args[1]
            return self._move(robot_name, loc_from_name, loc_to_name, **kwargs)
        else:
            print(f"Error: Skill '{skill_name}' not recognized.")
            return False

    def get_skills_time_fn(self, skill_name: str):
        pass

    def get_executed_skill_status(self, robot_name: str, skill_name: str):
        pass

    def stop_robot(self, robot_name: str):
        pass

    def _pick(self, robot_name, location_name, object_name):
        self.robots[robot_name].pick_object(object_name)

    def _place(self, robot_name, location_name, object_name):
        self.robots[robot_name].place_object()

    def _move(self, robot_name, loc_from_name, loc_to_name):
        self.robots[robot_name].navigate(goal=loc_to_name)
