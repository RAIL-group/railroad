import threading
import multiprocessing
import time as time_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple, Optional, Union, Type

import numpy as np
import matplotlib.pyplot as plt
from railroad._bindings import State, Fluent, Action
from railroad.environment import PhysicalEnvironment, PhysicalScene, SkillStatus, ActiveSkill
from railroad.core import Operator

from pyrobosim.core import Room, Location, ObjectSpawn
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


class PyRoboSimServer:
    """Simulator server that runs PyRoboSim in a separate process.

    Provides a service-like interface for executing skills and tracking their status
    via a shared memory dictionary.
    """

    def __init__(
        self,
        world_file: str | Path,
        shared_status: Dict,
        show_plot: bool = True,
        record_plots: bool = False,
        skip_canvas: bool = False,
    ):
        self.world_file = world_file
        self.show_plot = show_plot
        self._shared_status = shared_status
        self._world = WorldYamlLoader().from_file(Path(world_file))
        self._robots = {robot.name: robot for robot in self._world.robots}

        # Local task tracking (thread-safe for simple assignments)
        self._lock = threading.Lock()
        self._current_task_ids = {name: -1 for name in self._robots}
        self._last_completed_ids = {name: -1 for name in self._robots}

        # Initialize shared status for each robot
        for name in self._robots:
            self._shared_status[name] = {
                "current_task_id": -1,
                "last_completed_id": -1,
                "is_busy": False,
                "location": None,
                "holding": None,
                "is_saving_animation": False
            }

        self._is_saving_animation = False
        self.canvas = None
        if not skip_canvas:
            self.canvas = MatplotlibWorldCanvas(
                self._world, show_plot=show_plot, record_plots=record_plots
            )

    def run(self, command_queue: multiprocessing.Queue):
        """Main loop for the simulator process."""
        import queue
        try:
            dt = self.canvas.options.animation_dt if self.canvas else 0.01
            while True:
                start_time = time_module.time()

                # 1. Process commands
                while True:
                    try:
                        cmd = command_queue.get_nowait()
                        cmd_type = cmd[0]

                        if cmd_type == "execute":
                            _, robot_name, skill_name, args, cmd_id = cmd
                            self._handle_execute(robot_name, skill_name, args, cmd_id)
                        elif cmd_type == "stop":
                            _, robot_name = cmd
                            self._stop_robot(robot_name)
                        elif cmd_type == "save_animation":
                            _, filepath = cmd
                            if self.canvas:
                                self._is_saving_animation = True
                                self._update_global_status()
                                try:
                                    self.canvas.save_animation(filepath)
                                finally:
                                    self._is_saving_animation = False
                                    self._update_global_status()
                        elif cmd_type == "exit":
                            return
                    except queue.Empty:
                        break

                # 2. Update Shared Status Table
                self._update_global_status()

                # 3. Update GUI and capture frame
                if self.canvas:
                    self.canvas.update()

                # 4. Maintain timing
                elapsed = time_module.time() - start_time
                sleep_time = max(0, dt - elapsed)
                time_module.sleep(sleep_time)

        except Exception as e:
            print(f"CRITICAL: Simulator server crashed: {e}")
            import traceback
            traceback.print_exc()

    def _update_global_status(self):
        """Push current simulator state to the shared dictionary."""
        # Only the main loop should write to _shared_status to prevent race conditions
        with self._lock:
            for name, robot in self._robots.items():
                # Determine location name
                loc_name = None
                if robot.location:
                    if isinstance(robot.location, (Location, ObjectSpawn)):
                        loc_name = robot.location.parent.name if isinstance(robot.location, ObjectSpawn) else robot.location.name
                    else:
                        loc_name = robot.location.name # Room

                # Construct new status sub-dict
                new_status = {
                    "current_task_id": self._current_task_ids[name],
                    "last_completed_id": self._last_completed_ids[name],
                    "is_busy": robot.is_busy(),
                    "location": loc_name,
                    "holding": robot.manipulated_object.name if robot.manipulated_object else None,
                    "is_saving_animation": self._is_saving_animation
                }
                self._shared_status[name] = new_status

    def _handle_execute(self, robot_name: str, skill_name: str, args: Any, cmd_id: int):
        """Dispatch execution command and update task tracking."""
        with self._lock:
            self._current_task_ids[robot_name] = cmd_id

        if skill_name == "pick":
            obj_name = args[2] if len(args) > 2 else args[1]
            self._pick(robot_name, obj_name, cmd_id)
        elif skill_name == "place":
            self._place(robot_name, cmd_id)
        elif skill_name == "move":
            loc_to = args[2] if len(args) > 2 else args[1]
            # Need to handle initial locations
            initial_locs = {f"{r.name}_loc": r.pose for r in self._world.robots}
            if loc_to in initial_locs:
                self._move(robot_name, initial_locs[loc_to], cmd_id)
            else:
                self._move(robot_name, loc_to, cmd_id)
        elif skill_name == "no_op":
            self._no_op(robot_name, cmd_id)
        elif skill_name == "search":
            self._search(robot_name, cmd_id)

    def _on_task_complete(self, robot_name: str, cmd_id: int):
        """Callback when an async task completes."""
        with self._lock:
            if self._current_task_ids[robot_name] == cmd_id:
                self._current_task_ids[robot_name] = -1
                self._last_completed_ids[robot_name] = cmd_id

    def _stop_robot(self, robot_name: str) -> None:
        if self._robots[robot_name].is_busy():
            self._robots[robot_name].cancel_actions()
            with self._lock:
                self._current_task_ids[robot_name] = -1

    @_run_async
    def _pick(self, robot_name: str, object_name: str, cmd_id: int) -> None:
        self._robots[robot_name].pick_object(object_name)
        self._on_task_complete(robot_name, cmd_id)

    @_run_async
    def _place(self, robot_name: str, cmd_id: int) -> None:
        self._robots[robot_name].place_object()
        self._on_task_complete(robot_name, cmd_id)

    @_run_async
    def _move(self, robot_name: str, goal: Any, cmd_id: int) -> None:
        success = self._robots[robot_name].navigate(goal=goal)
        self._on_task_complete(robot_name, cmd_id)

    @_run_async
    def _no_op(self, robot_name: str, cmd_id: int) -> None:
        time_module.sleep(1.0)
        self._on_task_complete(robot_name, cmd_id)

    @_run_async
    def _search(self, robot_name: str, cmd_id: int) -> None:
        time_module.sleep(1.0)
        self._on_task_complete(robot_name, cmd_id)


class PyRoboSimClient:
    """Client for communicating with the PyRoboSimServer."""

    def __init__(
        self,
        world_file: str | Path,
        show_plot: bool = True,
        record_plots: bool = False,
        skip_canvas: bool = False,
    ):
        self._manager = multiprocessing.Manager()
        self._shared_status = self._manager.dict()
        self._command_queue = multiprocessing.Queue()

        self._process = multiprocessing.Process(
            target=self._run_server,
            args=(world_file, self._shared_status, self._command_queue, show_plot, record_plots, skip_canvas),
            daemon=True
        )
        self._process.start()

    @staticmethod
    def _run_server(world_file, shared_status, cmd_q, show_plot, record_plots, skip_canvas):
        server = PyRoboSimServer(world_file, shared_status, show_plot, record_plots, skip_canvas)
        server.run(cmd_q)

    def stop(self):
        try:
            self._command_queue.put(("exit",))
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
            self._manager.shutdown()
        except:
            pass

    def call_service(self, service_name: str, *args):
        self._command_queue.put((service_name, *args))

    def get_robot_status(self, robot_name: str) -> Dict:
        return self._shared_status.get(robot_name, {})


class PyRoboSimScene(PhysicalScene):
    """PyRoboSim scene data provider."""

    def __init__(self, world_file: str | Path) -> None:
        self.world_file = world_file
        self._world = WorldYamlLoader().from_file(Path(world_file))
        self._initial_robot_locations = {
            f"{r.name}_loc": r.pose for r in self._world.robots}
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
        return goal_node.pose

    def get_move_cost_fn(self):
        def get_move_time(robot, loc_from, loc_to):
            from_pose = self._get_feasible_pose_from_location_for_robot(self.robots[robot], loc_from)
            to_pose = self._get_feasible_pose_from_location_for_robot(self.robots[robot], loc_to)
            plan = self.robots[robot].path_planner.plan(from_pose, to_pose)
            self.robots[robot].path_planner.latest_path = None
            if plan is None:
                return float('inf')
            return plan.length / 1.0
        return get_move_time


class PyRoboSimEnvironment(PhysicalEnvironment):
    """Undecoupled PyRoboSim environment."""

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
        if self.canvas:
            self.canvas.update()

    def execute_skill(self, robot_name: str, skill_name: str, *args: Any, **kwargs: Any) -> None:
        if skill_name == "pick":
            obj_name = args[2] if len(args) > 2 else args[1]
            self._pick(robot_name, obj_name)
        elif skill_name == "place":
            self._place(robot_name)
        elif skill_name == "move":
            loc_to = args[2] if len(args) > 2 else args[1]
            self._move(robot_name, loc_to)
        elif skill_name == "no_op":
            self._no_op(robot_name)

    def get_executed_skill_status(self, robot_name: str, skill_name: str) -> SkillStatus:
        if skill_name == 'no_op' and self._is_no_op_running.get(robot_name, False):
            return SkillStatus.RUNNING
        robot = self._robots.get(robot_name)
        is_busy = robot is not None and robot.is_busy()
        return SkillStatus.RUNNING if is_busy else SkillStatus.DONE

    def stop_robot(self, robot_name: str) -> None:
        if self._robots[robot_name].is_busy():
            self._robots[robot_name].cancel_actions()

    @_run_async
    def _pick(self, robot_name: str, object_name: str) -> None:
        self._robots[robot_name].pick_object(object_name)

    @_run_async
    def _place(self, robot_name: str) -> None:
        self._robots[robot_name].place_object()

    @_run_async
    def _move(self, robot_name: str, loc_to: str) -> None:
        if loc_to in self._scene.initial_robot_locations:
             self._robots[robot_name].navigate(goal=self._scene.initial_robot_locations[loc_to])
        else:
             self._robots[robot_name].navigate(goal=loc_to)

    @_run_async
    def _no_op(self, robot_name: str) -> None:
        self._is_no_op_running[robot_name] = True
        time_module.sleep(1.0)
        self._is_no_op_running[robot_name] = False


class DecoupledPyRoboSimEnvironment(PhysicalEnvironment):
    """Decoupled PyRoboSim environment with a single simulator process.

    All robots are managed in a single simulator process to ensure
    synchronized visualization while maintaining non-blocking concurrent execution.
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
        super().__init__(
            scene=scene,
            state=state,
            objects_by_type=objects_by_type,
            operators=operators,
            skill_overrides=skill_overrides,
            location_registry=location_registry,
        )
        self._scene = scene
        self._client = PyRoboSimClient(
            scene.world_file, show_plot=show_plot, record_plots=record_plots, skip_canvas=skip_canvas
        )
        self._last_dispatched_ids = {name: -1 for name in objects_by_type.get("robot", [])}
        self._next_id = 0

        # Wait for simulator to be ready and provide initial status
        start_wait = time_module.time()
        ready = False
        while time_module.time() - start_wait < 15.0:
            # Check if we have status for at least one robot
            for robot_name in self._last_dispatched_ids:
                if self._client.get_robot_status(robot_name):
                    ready = True
                    break
            if ready:
                break
            time_module.sleep(0.1)

    def __del__(self):
        if hasattr(self, "_client"):
            self._client.stop()

    @property
    def robots(self):
        return self._scene.robots

    @property
    def locations(self):
        return self._scene.locations

    @property
    def objects(self):
        return {obj.name: obj for obj in self._scene.world.objects}

    def execute_skill(self, robot_name: str, skill_name: str, *args: Any, **kwargs: Any) -> None:
        cmd_id = self._next_id
        self._next_id += 1
        self._last_dispatched_ids[robot_name] = cmd_id
        self._client.call_service("execute", robot_name, skill_name, args, cmd_id)

    def get_executed_skill_status(self, robot_name: str, skill_name: str) -> SkillStatus:
        target_id = self._last_dispatched_ids.get(robot_name, -1)
        if target_id == -1:
            return SkillStatus.DONE

        status = self._client.get_robot_status(robot_name)
        if not status:
            return SkillStatus.RUNNING

        # Reliable handshake via IDs
        current_id = status.get("current_task_id", -1)
        last_completed = status.get("last_completed_id", -1)

        if current_id == target_id:
            return SkillStatus.RUNNING
        if last_completed >= target_id:
            return SkillStatus.DONE

        return SkillStatus.RUNNING

    def stop_robot(self, robot_name: str) -> None:
        self._client.call_service("stop", robot_name)

    def save_animation(self, filepath: str | Path) -> None:
        self._client.call_service("save_animation", str(filepath))

        # Wait for handshake
        start_wait = time_module.time()
        robot_name = next(iter(self._last_dispatched_ids.keys()))
        while time_module.time() - start_wait < 60.0:
            status = self._client.get_robot_status(robot_name)
            if status.get("is_saving_animation"):
                break
            time_module.sleep(0.5)

        while time_module.time() - start_wait < 120.0:
            status = self._client.get_robot_status(robot_name)
            if not status.get("is_saving_animation"):
                print("Animation save complete.")
                return
            time_module.sleep(1.0)


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
        self.draw_signal = self.MockSignal(self.draw_signal_callback)
        self.show_robots_signal = self.MockSignal(self.show_robots)
        self.show_planner_and_path_signal = self.MockSignal(self._show_all_paths)

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
        self._plot_frames = [] if record_plots else None

    def _show_all_paths(self):
        for robot in self.world.robots:
            self._draw_single_robot_path(robot)

    def _draw_single_robot_path(self, robot):
        if not robot.path_planner:
            return
        path = robot.path_planner.get_latest_path()
        if not path:
            return
        if robot.name in self.path_artists_storage:
            for artist in self.path_artists_storage[robot.name]:
                try:
                    artist.remove()
                except Exception:
                    pass
        new_artists_dict = plot_path_planner(
            self.axes,
            graphs=[],
            path=path,
            path_color=robot.color
        )
        flat_artists = new_artists_dict.get("path", []) + new_artists_dict.get("graph", [])
        self.path_artists_storage[robot.name] = flat_artists

    def show(self) -> None:
        self.show_rooms()
        self.show_hallways()
        self.show_locations()
        self.show_objects()
        self.show_robots()
        self.update_robots_plot()
        self._show_all_paths()
        self.axes.autoscale()
        self.axes.axis("equal")

    def draw_signal_callback(self):
        if hasattr(self, "fig"):
            self.fig.canvas.draw_idle()

    def update(self):
        self.show()
        self.fig.canvas.draw_idle()
        if self._plot_frames is not None:
            self._plot_frames.append(self._get_frame())
        if self.show_plot:
            plt.pause(self.options.animation_dt)

    def wait_for_close(self):
        if not self.show_plot:
            return
        plt.ioff()
        plt.show()

    def _get_frame(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        width = int(renderer.width)
        height = int(renderer.height)
        image = np.frombuffer(self.fig.canvas.tostring_argb(), dtype='uint8')
        image = image.reshape((height, width, 4))
        return image[..., 1:4]

    def save_animation(self, filepath):
        if not self._plot_frames:
            import warnings
            warnings.warn("No frames recorded to save animation.")
            return
        import imageio
        from PIL import Image
        fps = int(round(1 / self.options.animation_dt))
        target_size = (self._plot_frames[0].shape[1], self._plot_frames[0].shape[0])
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath_str = filepath if filepath.as_posix().startswith(("/", "./", "../")) else f"./{filepath}"
        writer = imageio.get_writer(filepath, format="ffmpeg", mode="I", fps=fps, codec="libx264", macro_block_size=None)
        for frame in self._plot_frames:
            frame_uint8 = frame.astype("uint8")
            if (frame.shape[1], frame.shape[0]) != target_size:
                img = Image.fromarray(frame_uint8)
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                frame_uint8 = np.array(img)
            writer.append_data(frame_uint8)
        writer.close()
        print(f"Animation saved to {filepath_str}")
