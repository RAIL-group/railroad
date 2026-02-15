import threading
import time as time_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple, Optional, Union, Type

import numpy as np
import matplotlib.pyplot as plt
from railroad._bindings import State, Fluent, Action
from railroad.environment import PhysicalEnvironment, PhysicalScene, SkillStatus, ActiveSkill
from railroad.core import Operator

from pyrobosim.core.yaml_utils import WorldYamlLoader
from pyrobosim.utils.knowledge import query_to_entity, graph_node_from_entity
from pyrobosim.gui.world_canvas import WorldCanvas
from pyrobosim.gui.options import WorldCanvasOptions
from pyrobosim.navigation.visualization import plot_path_planner


def get_default_pyrobosim_world_file_path() -> Path:
    """Get the packaged default world file path."""
    return Path(__file__).resolve().parent / "resources" / "worlds" / "test_world.yaml"


def _run_async(func: Callable) -> Callable:
    """Decorator to run a function in a daemon thread."""
    import functools

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




class PyRoboSimScene(PhysicalScene):
    """PyRoboSim scene data provider."""

    def __init__(self, world_file: str | Path) -> None:
        self._world = WorldYamlLoader().from_file(Path(world_file))
        self._initial_robot_locations = {
            f"{r.name}_loc": r.pose for r in self._world.robots
        }
        self._location_poses = {loc.name: loc.pose for loc in self._world.locations}
        self._location_poses.update(self._initial_robot_locations)
        self._robots = {robot.name: robot for robot in self._world.robots}

        self._objects = {obj.name for obj in self._world.objects}

        self._object_locations: Dict[str, Set[str]] = {}
        for loc in self._world.locations:
            self._object_locations[loc.name] = set()
            for spawn in loc.children:
                for obj in spawn.children:
                    self._object_locations[loc.name].add(obj.name)

    @property
    def world(self):
        return self._world

    @property
    def robots(self):
        return self._robots

    @property
    def locations(self) -> Dict[str, Any]:
        return {name: (pose.x, pose.y) for name, pose in self._location_poses.items()}

    @property
    def objects(self) -> Set[str]:
        return self._objects

    @property
    def object_locations(self) -> Dict[str, Set[str]]:
        return self._object_locations

    @property
    def initial_robot_locations(self) -> Dict[str, Any]:
        return self._initial_robot_locations

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

    def get_move_cost_fn(self):
        def get_move_time(robot, loc_from, loc_to):
            # Get feasible poses for the robot at the from and to locations
            from_pose = self._get_feasible_pose_from_location_for_robot(self.robots[robot], loc_from)
            to_pose = self._get_feasible_pose_from_location_for_robot(self.robots[robot], loc_to)

            plan = self.robots[robot].path_planner.plan(from_pose, to_pose)  # type: ignore[union-attr]

            # Clear the latest path to avoid showing in the plot
            self.robots[robot].path_planner.latest_path = None  # type: ignore[union-attr]

            if plan is None:
                return float('inf')
            return plan.length / 1.0  # robot velocity = 1.0
        return get_move_time


class PyRoboSimEnvironment(PhysicalEnvironment):
    """PyRoboSim environment implementing the PhysicalEnvironment protocol.

    Provides integration with PyRoboSim simulator for PDDL planning execution.
    Initializes world from YAML and manages robots, objects, and locations.
    """

    def __init__(
        self,
        scene: PyRoboSimScene,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        operators: List[Operator],
        show_plot: bool = True,
        record_plots: bool = False,
        skip_canvas: bool = False,
        skill_overrides: Optional[Dict[str, Type[ActiveSkill]]] = None,
        location_registry: Any = None,
    ) -> None:
        """Initialize PyRoboSim environment.

        Args:
            scene: PyRoboSimScene data provider.
            state: Initial planning state.
            objects_by_type: Objects organized by type.
            operators: Planning operators.
            show_plot: Whether to show matplotlib visualization.
            record_plots: Whether to record animation frames.
            skip_canvas: Whether to skip canvas initialization.
            skill_overrides: Optional mapping from action type prefix to skill class.
            location_registry: Optional LocationRegistry for coordinate tracking.
        """
        self._scene = scene
        self._world = scene.world
        self._robots = {robot.name: robot for robot in self._world.robots}
        self._is_no_op_running: Dict[str, bool] = {robot: False for robot in self._robots}

        super().__init__(
            scene=scene,
            state=state,
            objects_by_type=objects_by_type,
            operators=operators,
            skill_overrides=skill_overrides,
            location_registry=location_registry
        )

        self.canvas = None
        if not skip_canvas:
            self.canvas = MatplotlibWorldCanvas(self._world, show_plot, record_plots)

    @property
    def robots(self):
        return self._robots

    @property
    def locations(self):
        return self._scene.locations

    @property
    def objects(self):
        return {obj.name: obj for obj in self._world.objects}

    def _on_act_loop_iteration(self, dt: float) -> None:
        """Process GUI events on each loop iteration."""
        if self.canvas:
            self.canvas.update()

    # --- PhysicalEnvironment Abstract Methods Implementation ---

    def execute_skill(self, robot_name: str, skill_name: str, *args: Any, **kwargs: Any) -> None:
        """Execute a skill on a robot."""
        if skill_name == "pick":
            # args might be (loc, obj) or (robot, loc, obj) depending on caller
            obj_name = args[2] if len(args) > 2 else args[1]
            self._pick(robot_name, obj_name)
        elif skill_name == "place":
            self._place(robot_name)
        elif skill_name == "move":
            loc_to = args[2] if len(args) > 2 else args[1]
            self._move(robot_name, loc_to)
        elif skill_name == "no_op":
            self._no_op(robot_name)
        time_module.sleep(0.1)

    def get_executed_skill_status(self, robot_name: str, skill_name: str) -> SkillStatus:
        """Get the execution status of a skill."""
        if skill_name == 'no_op' and self._is_no_op_running.get(robot_name, False):
            return SkillStatus.RUNNING

        robot = self._robots.get(robot_name)
        is_busy = robot is not None and robot.is_busy()
        return SkillStatus.RUNNING if is_busy else SkillStatus.DONE

    def stop_robot(self, robot_name: str) -> None:
        """Stop robot's current physical action."""
        if self._robots[robot_name].is_busy():
            self._robots[robot_name].cancel_actions()

    # --- Utility Methods ---

    def get_skills_time_fn(self, skill_name: str) -> Callable[..., float]:
        """Get a time function for a skill."""
        if skill_name == 'move':
            return self._scene.get_move_cost_fn()
        else:
            def get_skill_time(robot_name, *args, **kwargs):
                return 1.0
            return get_skill_time

    # --- Private Implementation Methods ---

    @_run_async
    def _pick(self, robot_name: str, object_name: str) -> None:
        self._robots[robot_name].pick_object(object_name)

    @_run_async
    def _place(self, robot_name: str) -> None:
        self._robots[robot_name].place_object()

    @_run_async
    def _move(self, robot_name: str, loc_to: str) -> None:
        # If loc_to is a custom starting location, pass the raw pose
        if loc_to in self._scene.initial_robot_locations:
             self._robots[robot_name].navigate(goal=self._scene.initial_robot_locations[loc_to])
        else:
             self._robots[robot_name].navigate(goal=loc_to)

    @_run_async
    def _no_op(self, robot_name: str) -> None:
        self._is_no_op_running[robot_name] = True
        time_module.sleep(1.0)
        self._is_no_op_running[robot_name] = False


class MatplotlibWorldCanvas(WorldCanvas):
    """
    Matplotlib-based visualization canvas for PyRoboSim worlds.
    Replaces Qt-based WorldCanvas to support non-blocking plotting.
    """

    class MockSignal:
        def __init__(self, callback):
            self.callback = callback

        def emit(self, *args, **kwargs):
            self.callback()

    class MockMainWindow:
        def get_current_robot(self):
            return None

    def __init__(self, world, show_plot: bool = True, record_plots: bool = False):
        self.world = world
        self.show_plot = show_plot
        if not self.show_plot:
            plt.switch_backend("Agg")
        self.record_plots = record_plots
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
        self.show_planner_and_path_signal = self.MockSignal(self._show_all_paths)

        # Manually initialize the artist lists from WorldCanvas since we do not call its __init__
        self.path_artists_storage = {}
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
        self._plot_frames = []

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
        self._plot_frames.append(self._get_frame())
        if self.show_plot:
            plt.pause(self.options.animation_dt)

    def wait_for_close(self):
        """Blocks until the plot window is closed."""
        if not self.show_plot:
            return
        plt.ioff()
        plt.show()

    def _get_frame(self):
        """Captures the current frame as an image array."""
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        width = int(renderer.width)
        height = int(renderer.height)
        image = np.frombuffer(self.fig.canvas.tostring_argb(), dtype='uint8')
        image = image.reshape((height, width, 4))
        return image[..., 1:4]  # Convert ARGB to RGB

    def save_animation(self, filepath):
        """Saves the recorded frames as a video file."""
        if not self._plot_frames:
            import warnings
            warnings.warn("No frames recorded to save animation. Use 'record_plots=True' to record plot frames.")
            return

        import imageio
        from PIL import Image
        fps = int(round(1 / self.options.animation_dt))

        # Use the first frame's size as the target
        target_size = (self._plot_frames[0].shape[1], self._plot_frames[0].shape[0])

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Add ./ to relative paths, for easier CLI use
        filepath_str = filepath if filepath.as_posix().startswith(("/", "./", "../")) else f"./{filepath}"

        writer = imageio.get_writer(
            filepath,
            format="ffmpeg",  # type: ignore[arg-type]  # imageio accepts string format names
            mode="I",
            fps=fps,
            codec="libx264",
            macro_block_size=None
        )

        for frame in self._plot_frames:
            frame_uint8 = frame.astype("uint8")
            # Resize if dimensions don't match
            if (frame.shape[1], frame.shape[0]) != target_size:
                img = Image.fromarray(frame_uint8)
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                frame_uint8 = np.array(img)
            writer.append_data(frame_uint8)

        writer.close()
        print(f"Animation saved to {filepath_str}")
