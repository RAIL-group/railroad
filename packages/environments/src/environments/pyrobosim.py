import threading
import functools
from pyrobosim.core.yaml_utils import WorldYamlLoader
from .environments import BaseEnvironment, SkillStatus
import time
import matplotlib.pyplot as plt
from pyrobosim.gui.world_canvas import WorldCanvas
from pyrobosim.gui.options import WorldCanvasOptions
from pyrobosim.utils.knowledge import query_to_entity, graph_node_from_entity
from pyrobosim.navigation.visualization import plot_path_planner


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
    def __init__(self, world_file: str, show_plot: bool = True):
        self.world = WorldYamlLoader().from_file(world_file)
        self.canvas = MatplotlibWorldCanvas(self.world, show_plot)
        self.initial_robot_locations = {
            f"{r.name}_loc": r.pose for r in self.world.robots
        }
        self.locations = {loc.name: loc.pose for loc in self.world.locations}
        self.locations.update(self.initial_robot_locations)
        self.robots = {robot.name: robot for robot in self.world.robots}
        self.is_robot_assigned = {robot: False for robot in self.robots}
        self.is_no_op_running = {robot: False for robot in self.robots}

    def get_objects_at_location(self, location):
        objects = {'object': set()}
        if location.endswith("_loc"):
            # Our custom robot locations (not defined in pyrobosim) has no objects
            return objects
        location = self.world.get_location_by_name(location)
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
        # No need to search for now since all objects are assumed to be known
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

        def isVisible(self):
            return True

    def __init__(self, world, show_plot=True):
        options = WorldCanvasOptions()
        main_window = self.MockMainWindow()
        options = WorldCanvasOptions()

        super().__init__(main_window, world, show=show_plot, options=options)
        self.fig = plt.figure(dpi=options.dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)

        self.path_artists_storage = {}

    def _show_all_paths(self):
        """Custom method to draw paths for EVERY robot in the world."""
        for robot in self.world.robots:
            self._draw_single_robot_path(robot)

    def _draw_single_robot_path(self, robot):
        """Helper to draw a specific robot's path without clearing others."""
        if not robot.path_planner:
            return

        # Get the path from the robot
        path = robot.path_planner.get_latest_path()
        if not path:
            return

        # Clear OLD artists for THIS specific robot only
        if robot.name in self.path_artists_storage:
            for artist in self.path_artists_storage[robot.name]:
                try:
                    artist.remove()
                except Exception:
                    pass

        # Note: plot_path_planner returns a dict of lists of artists
        new_artists_dict = plot_path_planner(
            self.axes,
            graphs=[],  # Set to graphs=robot.path_planner.get_graphs() if you want the RRT/PRM trees
            path=path,
            path_color=robot.color
        )

        # Store these artists so we can remove them in the next frame
        flat_artists = new_artists_dict.get("path", []) + new_artists_dict.get("graph", [])
        self.path_artists_storage[robot.name] = flat_artists

    def show(self) -> None:
        """Overriding show to ensure all robots are processed."""
        self.show_rooms()
        self.show_hallways()
        self.show_locations()
        self.show_objects()
        self.show_robots()
        self.update_robots_plot()

        self._show_all_paths()  # Show paths for all robots

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
