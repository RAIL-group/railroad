import threading
import functools
from pyrobosim.core.yaml_utils import WorldYamlLoader
from .environments import BaseEnvironment, SkillStatus
import time
import matplotlib.pyplot as plt
from pyrobosim.gui.world_canvas import WorldCanvas
from pyrobosim.gui.options import WorldCanvasOptions
from pyrobosim.utils.knowledge import query_to_entity, graph_node_from_entity



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
        self.canvas = MatplotlibWorldCanvas(self.world)
        self.initial_robot_locations = {
            f"{r.name}_loc": r.pose for r in self.world.robots
        }
        self.locations = {loc.name: loc.pose for loc in self.world.locations}
        self.locations.update(self.initial_robot_locations)
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

    def get_skills_cost_fn(self, skill_name: str):
        if skill_name == 'move':
            return self._get_move_cost_fn()
        else:
            def get_skill_time(robot_name, *args, **kwargs):
                return 1.0  # TODO: Get skills time from pyrobosim
            return get_skill_time


    def _get_feasible_pose_from_location_for_robot(self, robot, location_name):
            if location_name in self.initial_robot_locations:
                return self.initial_robot_locations[location_name]

            entity = query_to_entity(
                self.world,
                location_name,
                mode="location",
                robot=robot,
                resolution_strategy="nearest",
            )
            if entity is None:
                raise ValueError(f"Could not find entity for location '{location_name}'.")

            goal_node = graph_node_from_entity(self.world, entity, robot=robot)
            if goal_node is None:
                raise ValueError(f"Could not find graph node associated with location '{location_name}'.")
            goal = goal_node.pose
            return goal

    def _get_move_cost_fn(self):
        def get_move_time(robot, loc_from, loc_to):
            # Get feasible poses for the robot at the from and to locations
            from_pose = self._get_feasible_pose_from_location_for_robot(self.robots[robot], loc_from)
            to_pose = self._get_feasible_pose_from_location_for_robot(self.robots[robot], loc_to)

            plan = self.robots[robot].path_planner.plan(from_pose, to_pose)

            # Clear the latest path to avoid showing in the plot
            self.robots[robot].path_planner.latest_path = None

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
        if self.robots[robot_name].is_busy():
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
        # self.robots[robot_name].detect_objects()
        pass

    @run_async
    def _no_op(self, robot_name):
        self.is_no_op_running[robot_name] = True
        time.sleep(self.get_skills_cost_fn('no_op')(robot_name))
        self.is_no_op_running[robot_name] = False


class MatplotlibWorldCanvas(WorldCanvas):
    class MockSignal:
        def __init__(self, callback):
            self.callback = callback

        def emit(self, *args, **kwargs):
            self.callback()

    class MockMainWindow:
        def get_current_robot(self):
            return None

    def __init__(self, world):
        self.world = world
        self.options = WorldCanvasOptions()
        self.main_window = self.MockMainWindow()
        self.fig, self.axes = plt.subplots(
            dpi=self.options.dpi,
            tight_layout=True
        )
        plt.ion()

        # Hijack the signals BEFORE calling any methods like show()
        # This prevents the "Signal source has been deleted" error
        self.draw_signal = self.MockSignal(self.draw_signal_callback)
        # Add other signals as needed to prevent attribute errors
        self.show_robots_signal = self.MockSignal(self.show_robots)
        self.show_planner_and_path_signal = self.MockSignal(lambda: None)

        # Manually initialize the artist lists inherited from WorldCanvas
        self.robot_bodies = []
        self.robot_dirs = []
        self.robot_lengths = []
        self.robot_texts = []
        self.robot_sensor_artists = []
        self.obj_patches = []
        self.obj_texts = []
        self.hallway_patches = []
        self.room_patches = []
        self.room_texts = []
        self.location_patches = []
        self.location_texts = []
        self.path_planner_artists = {"graph": [], "path": []}

        self.show()
        self.axes.autoscale()
        self.axes.axis("equal")

    def draw_signal_callback(self):
        """Replacement for the Qt signal execution."""
        if hasattr(self, "fig"):
            self.fig.canvas.draw_idle()

    def update(self):
        """Updates the world visualization in a loop."""
        self.show()
        self.fig.canvas.draw_idle()
        plt.pause(self.options.animation_dt)
