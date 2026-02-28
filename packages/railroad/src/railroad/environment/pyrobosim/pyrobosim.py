import functools
import threading
import multiprocessing
import time as time_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from railroad._bindings import State, Fluent, Action
from railroad.environment import PhysicalEnvironment, PhysicalScene, SkillStatus, ActiveSkill
from railroad.core import Operator

from pyrobosim.core import Location, ObjectSpawn
from pyrobosim.core.yaml_utils import WorldYamlLoader
from pyrobosim.utils.knowledge import query_to_entity, graph_node_from_entity

from .canvas import MatplotlibWorldCanvas


def get_default_pyrobosim_world_file_path() -> Path:
    """Get the packaged default world file path."""
    return Path(__file__).resolve().parent / "resources" / "worlds" / "test_world.yaml"


def _run_async(func: Callable) -> Callable:
    """Decorator to run a function in a daemon thread with error handling."""

    @functools.wraps(func)
    def wrapper(self, robot_name: str, *args, **kwargs):
        def _thread_body():
            try:
                func(self, robot_name, *args, **kwargs)
            except Exception as e:
                # Store failure info so the client can detect it
                cmd_id = args[-1] if args else -1
                with self._lock:
                    self._failed_task_ids[robot_name] = cmd_id
                    self._failed_errors[robot_name] = str(e)
                    self._current_task_ids[robot_name] = -1
                import traceback
                traceback.print_exc()

        thread = threading.Thread(target=_thread_body, daemon=True)
        thread.start()
        return thread
    return wrapper


class PyRoboSimServer:
    """Simulator server that runs PyRoboSim in a separate process.

    Provides a service-like interface for executing skills and tracking their status
    via a shared memory dictionary. Computes and sends metadata to the client on startup.
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
        self._current_task_ids: Dict[str, int] = {name: -1 for name in self._robots}
        self._last_completed_ids: Dict[str, int] = {name: -1 for name in self._robots}
        self._failed_task_ids: Dict[str, int] = {name: -1 for name in self._robots}
        self._failed_errors: Dict[str, str] = {}

        # Initialize shared status for each robot
        for name in self._robots:
            self._shared_status[name] = {
                "current_task_id": -1,
                "last_completed_id": -1,
                "failed_task_id": -1,
                "failed_error": "",
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

    def _compute_metadata(self) -> Dict[str, Any]:
        """Compute world metadata for the client."""
        robot_names = list(self._robots.keys())

        # Location poses (actual locations + robot initial locations)
        location_poses: Dict[str, Tuple[float, float]] = {}
        for loc in self._world.locations:
            location_poses[loc.name] = (loc.pose.x, loc.pose.y)
        for robot in self._world.robots:
            location_poses[f"{robot.name}_loc"] = (robot.pose.x, robot.pose.y)

        # Room poses (separate from locations, used for move cost precomputation)
        room_poses: Dict[str, Tuple[float, float]] = {}
        for room in self._world.rooms:
            nav_pose = room.nav_poses[0] if room.nav_poses else room.centroid
            room_poses[room.name] = (nav_pose.x, nav_pose.y)

        # Objects and their locations
        objects: Set[str] = {obj.name for obj in self._world.objects}
        object_locations: Dict[str, Set[str]] = {}
        for loc in self._world.locations:
            object_locations[loc.name] = set()
            for spawn in loc.children:
                for obj in spawn.children:
                    object_locations[loc.name].add(obj.name)

        # Precompute move costs for all (robot, from, to) pairs
        move_costs: Dict[Tuple[str, str, str], float] = {}
        room_names = [room.name for room in self._world.rooms]
        real_location_names = [loc.name for loc in self._world.locations]
        # Include rooms, locations, and robot initial locations
        all_loc_names = room_names + real_location_names + [f"{r.name}_loc" for r in self._world.robots]

        for robot_name, robot in self._robots.items():
            for from_loc in all_loc_names:
                for to_loc in all_loc_names:
                    if from_loc == to_loc:
                        move_costs[(robot_name, from_loc, to_loc)] = 0.0
                        continue
                    try:
                        planner = robot.path_planner
                        if planner is None:
                            move_costs[(robot_name, from_loc, to_loc)] = float('inf')
                            continue
                        from_pose = self._get_pose_for_location(robot, from_loc)
                        to_pose = self._get_pose_for_location(robot, to_loc)
                        plan = planner.plan(from_pose, to_pose)
                        planner.latest_path = None
                        if plan is None:
                            move_costs[(robot_name, from_loc, to_loc)] = float('inf')
                        else:
                            move_costs[(robot_name, from_loc, to_loc)] = plan.length / 1.0
                    except Exception:
                        move_costs[(robot_name, from_loc, to_loc)] = float('inf')

        return {
            "robot_names": robot_names,
            "location_poses": location_poses,
            "room_poses": room_poses,
            "objects": objects,
            "object_locations": object_locations,
            "move_costs": move_costs,
        }

    def _get_pose_for_location(self, robot, location_name: str):
        """Get a feasible pose for a location name, used for path planning."""
        # Check if it's a robot initial location
        initial_locs = {f"{r.name}_loc": r.pose for r in self._world.robots}
        if location_name in initial_locs:
            return initial_locs[location_name]

        # Check if it's a room (use nav pose)
        for room in self._world.rooms:
            if room.name == location_name:
                return room.nav_poses[0] if room.nav_poses else room.centroid

        entity = query_to_entity(
            self._world,
            location_name,
            mode="location",
            robot=robot,
            resolution_strategy="nearest",
        )
        if entity is None:
            raise ValueError(f"Could not find entity for location '{location_name}'.")

        goal_node = graph_node_from_entity(self._world, entity, robot=robot)
        if goal_node is None:
            raise ValueError(f"Could not find graph node for location '{location_name}'.")
        return goal_node.pose

    def run(self, command_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue):
        """Main loop for the simulator process."""
        import queue
        try:
            # Send metadata to client before entering main loop
            metadata = self._compute_metadata()
            response_queue.put(("metadata", metadata))

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
                                    response_queue.put(("save_animation_done", filepath))
                                except Exception as e:
                                    response_queue.put(("error", str(e)))
                                finally:
                                    self._is_saving_animation = False
                                    self._update_global_status()
                            else:
                                response_queue.put(("error", "No canvas available for animation"))
                        elif cmd_type == "query_move_cost":
                            _, robot_name, from_pose_xy, to_location_name = cmd
                            try:
                                robot = self._robots[robot_name]
                                planner = robot.path_planner
                                if planner is None:
                                    response_queue.put(("move_cost", float('inf')))
                                else:
                                    from pyrobosim.utils.pose import Pose
                                    from_pose = Pose(x=from_pose_xy[0], y=from_pose_xy[1])
                                    to_pose = self._get_pose_for_location(robot, to_location_name)
                                    plan = planner.plan(from_pose, to_pose)
                                    planner.latest_path = None
                                    if plan is None:
                                        response_queue.put(("move_cost", float('inf')))
                                    else:
                                        response_queue.put(("move_cost", plan.length / 1.0))
                            except Exception as e:
                                response_queue.put(("move_cost", float('inf')))
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
        with self._lock:
            for name, robot in self._robots.items():
                # Determine location name
                loc_name = None
                if robot.location:
                    if isinstance(robot.location, ObjectSpawn):
                        loc_name = robot.location.parent.name if robot.location.parent else None
                    elif isinstance(robot.location, Location):
                        loc_name = robot.location.name
                    else:
                        loc_name = robot.location.name  # Room

                new_status = {
                    "current_task_id": self._current_task_ids[name],
                    "last_completed_id": self._last_completed_ids[name],
                    "failed_task_id": self._failed_task_ids.get(name, -1),
                    "failed_error": self._failed_errors.get(name, ""),
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
            # Handle initial robot locations and *_loc intermediate locations
            initial_locs = {f"{r.name}_loc": r.pose for r in self._world.robots}
            if loc_to in initial_locs:
                self._move(robot_name, initial_locs[loc_to], cmd_id)
            elif "_loc" in loc_to:
                # Intermediate location — coordinates will be passed as extra arg
                if len(args) > 3:
                    # Coordinates passed as (x, y) tuple
                    coords = args[3]
                    from pyrobosim.utils.pose import Pose
                    self._move(robot_name, Pose(x=coords[0], y=coords[1]), cmd_id)
                else:
                    self._move(robot_name, loc_to, cmd_id)
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
        self._robots[robot_name].navigate(goal=goal)
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
    """Client for communicating with the PyRoboSimServer.

    Creates a server process and provides methods for sending commands
    and receiving responses. Blocks on startup to receive world metadata.
    """

    def __init__(
        self,
        world_file: str | Path,
        show_plot: bool = True,
        record_plots: bool = False,
        skip_canvas: bool = False,
    ):
        self._manager = multiprocessing.Manager()
        self._shared_status = self._manager.dict()
        self._command_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._response_queue: multiprocessing.Queue = multiprocessing.Queue()

        self._process = multiprocessing.Process(
            target=self._run_server,
            args=(world_file, self._shared_status, self._command_queue,
                  self._response_queue, show_plot, record_plots, skip_canvas),
            daemon=True
        )
        self._process.start()

        # Block until metadata arrives from server
        self.metadata = self._wait_for_response(timeout=30.0)
        if self.metadata is None or self.metadata[0] != "metadata":
            raise RuntimeError("Failed to receive metadata from PyRoboSim server")
        self.metadata = self.metadata[1]

    @staticmethod
    def _run_server(world_file, shared_status, cmd_q, resp_q, show_plot, record_plots, skip_canvas):
        server = PyRoboSimServer(world_file, shared_status, show_plot, record_plots, skip_canvas)
        server.run(cmd_q, resp_q)

    def _wait_for_response(self, timeout: float = 60.0) -> Any:
        """Block on response queue until a message arrives or timeout."""
        try:
            return self._response_queue.get(timeout=timeout)
        except Exception:
            return None

    def stop(self):
        try:
            self._command_queue.put(("exit",))
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
            self._manager.shutdown()
        except Exception:
            pass

    def call_service(self, service_name: str, *args):
        self._command_queue.put((service_name, *args))

    def get_robot_status(self, robot_name: str) -> Dict:
        return self._shared_status.get(robot_name, {})


class PyRoboSimScene(PhysicalScene):
    """PyRoboSim scene data provider.

    Creates a PyRoboSimClient (which spawns the server process) and caches
    world metadata. The world YAML is loaded only in the server process.
    """

    def __init__(
        self,
        world_file: str | Path,
        show_plot: bool = True,
        record_plots: bool = False,
        skip_canvas: bool = False,
    ) -> None:
        import atexit

        self.world_file = world_file
        self._client = PyRoboSimClient(
            world_file,
            show_plot=show_plot,
            record_plots=record_plots,
            skip_canvas=skip_canvas,
        )
        atexit.register(self.close)

        # Cache metadata from server
        metadata = self._client.metadata
        self._robot_names: List[str] = metadata["robot_names"]
        self._location_poses: Dict[str, Tuple[float, float]] = metadata["location_poses"]
        self._room_poses: Dict[str, Tuple[float, float]] = metadata["room_poses"]
        self._objects: Set[str] = metadata["objects"]
        self._object_locations: Dict[str, Set[str]] = metadata["object_locations"]
        self._move_costs: Dict[Tuple[str, str, str], float] = metadata["move_costs"]
        self._location_registry: Any = None

    @property
    def client(self) -> PyRoboSimClient:
        return self._client

    @property
    def robot_names(self) -> List[str]:
        return self._robot_names

    @property
    def locations(self) -> Dict[str, Tuple[float, float]]:
        return dict(self._location_poses)

    @property
    def objects(self) -> Set[str]:
        return self._objects

    @property
    def object_locations(self) -> Dict[str, Set[str]]:
        return self._object_locations

    def set_location_registry(self, registry: Any) -> None:
        """Set the location registry for resolving *_loc intermediate locations."""
        self._location_registry = registry

    def get_move_cost_fn(self) -> Callable[[str, str, str], float]:
        """Get move cost function using precomputed costs with RPC fallback for *_loc."""
        precomputed = self._move_costs
        client = self._client
        registry_ref = self  # reference self to access _location_registry dynamically

        def get_move_time(robot: str, loc_from: str, loc_to: str) -> float:
            # Try precomputed cache first
            key = (robot, loc_from, loc_to)
            if key in precomputed:
                return precomputed[key]

            # For *_loc intermediate locations, use RPC to server
            registry = registry_ref._location_registry
            if registry is not None and loc_from.endswith("_loc"):
                coords = registry.get(loc_from)
                if coords is not None:
                    from_xy = (float(coords[0]), float(coords[1]))
                    client.call_service("query_move_cost", robot, from_xy, loc_to)
                    response = client._wait_for_response(timeout=10.0)
                    if response is not None and response[0] == "move_cost":
                        return response[1]

            return float('inf')

        return get_move_time

    def close(self) -> None:
        """Shut down the server process."""
        self._client.stop()


class PyRoboSimEnvironment(PhysicalEnvironment):
    """PyRoboSim environment using a subprocess-based simulator.

    All robots are managed in a single simulator process to ensure
    synchronized visualization while maintaining non-blocking concurrent execution.
    """

    def __init__(
        self,
        scene: PyRoboSimScene,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        operators: List[Operator],
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
        self._client = scene.client
        self._last_dispatched_ids: Dict[str, int] = {name: -1 for name in objects_by_type.get("robot", [])}
        self._next_id = 0

        # Connect location registry to scene for move cost lookups
        if location_registry is not None:
            scene.set_location_registry(location_registry)

        # Wait for simulator to be ready
        start_wait = time_module.time()
        ready = False
        while time_module.time() - start_wait < 15.0:
            for robot_name in self._last_dispatched_ids:
                if self._client.get_robot_status(robot_name):
                    ready = True
                    break
            if ready:
                break
            time_module.sleep(0.1)

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

        # Check for failure
        failed_id = status.get("failed_task_id", -1)
        if failed_id == target_id:
            error = status.get("failed_error", "Unknown error")
            raise RuntimeError(f"Skill execution failed for {robot_name}: {error}")

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
        """Save recorded animation frames to video file."""
        self._client.call_service("save_animation", str(filepath))
        response = self._client._wait_for_response(timeout=120.0)
        if response is None:
            print("Warning: Animation save timed out.")
        elif response[0] == "error":
            print(f"Warning: Animation save failed: {response[1]}")
        else:
            print("Animation save complete.")
