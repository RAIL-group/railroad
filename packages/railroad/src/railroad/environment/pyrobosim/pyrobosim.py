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
    """Agnostic simulator server that runs PyRoboSim in a separate process.

    Tracks command execution state using IDs to ensure robust synchronization.
    """

    def __init__(
        self,
        world_file: str | Path,
        show_plot: bool = True,
        record_plots: bool = False,
        skip_canvas: bool = False,
    ):
        self.world_file = world_file
        self.show_plot = show_plot
        self._world = WorldYamlLoader().from_file(Path(world_file))
        self._robots = {robot.name: robot for robot in self._world.robots}

        # Robot State Tracking
        # Maps robot_name -> { "current_id": int|None, "last_completed_id": int }
        self._robot_metadata = {
            name: {"current_id": None, "last_completed_id": -1}
            for name in self._robots
        }

        self._is_saving_animation = False
        self.canvas = None
        if not skip_canvas:
            self.canvas = MatplotlibWorldCanvas(
                self._world, show_plot=show_plot, record_plots=record_plots
            )

    def run(self, command_queue: multiprocessing.Queue, status_queue: multiprocessing.Queue):
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
                            # print(f"DEBUG: Simulator server dispatching {skill_name} for {robot_name} (ID: {cmd_id})")
                            self._handle_execute(robot_name, skill_name, args, cmd_id)
                        elif cmd_type == "stop":
                            _, robot_name = cmd
                            self._stop_robot(robot_name)
                        elif cmd_type == "save_animation":
                            _, filepath = cmd
                            if self.canvas:
                                self._is_saving_animation = True
                                try:
                                    self.canvas.save_animation(filepath)
                                finally:
                                    self._is_saving_animation = False
                        elif cmd_type == "exit":
                            return
                    except queue.Empty:
                        break

                # 2. Publish Status
                status_update = self._collect_status()
                status_queue.put(status_update)

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

    def _handle_execute(self, robot_name: str, skill_name: str, args: Any, cmd_id: int):
        """Dispatch execution command."""
        # Update metadata to show we are working on this ID
        self._robot_metadata[robot_name]["current_id"] = cmd_id

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
        meta = self._robot_metadata[robot_name]
        # Only update if this was the current task (handle cancellations/overwrites)
        if meta["current_id"] == cmd_id:
            meta["current_id"] = None
            meta["last_completed_id"] = cmd_id

    def _collect_status(self) -> Dict:
        """Snapshot the world state."""
        robot_states = {}
        for name, robot in self._robots.items():
            meta = self._robot_metadata[name]

            # Determine location info
            loc_name = None
            loc_type = None
            if robot.location:
                if isinstance(robot.location, (Location, ObjectSpawn)):
                    loc_type = "location"
                    if isinstance(robot.location, ObjectSpawn):
                        loc_name = robot.location.parent.name
                    else:
                        loc_name = robot.location.name
                elif isinstance(robot.location, Room):
                    loc_type = "room"
                    loc_name = robot.location.name

            robot_states[name] = {
                "pose": (robot.pose.x, robot.pose.y),
                "holding": robot.manipulated_object.name if robot.manipulated_object else None,
                "location": loc_name,
                "location_type": loc_type,
                # Command tracking
                "current_id": meta["current_id"],
                "last_completed_id": meta["last_completed_id"]
            }

        object_states = {}
        for obj in self._world.objects:
            loc_name = None
            loc_type = None
            if obj.parent:
                if isinstance(obj.parent, (Location, ObjectSpawn)):
                    loc_type = "location"
                    if isinstance(obj.parent, ObjectSpawn):
                        loc_name = obj.parent.parent.name
                    else:
                        loc_name = obj.parent.name
                else:
                    loc_name = obj.parent.name
                    loc_type = "room"

            object_states[obj.name] = {
                "location": loc_name,
                "location_type": loc_type
            }

        return {
            "time": time_module.time(),
            "is_saving_animation": self._is_saving_animation,
            "robots": robot_states,
            "objects": object_states
        }

    def _stop_robot(self, robot_name: str) -> None:
        if self._robots[robot_name].is_busy():
            self._robots[robot_name].cancel_actions()
            # Note: We don't mark task complete here, the async task will exit and
            # might not call complete, or we can handle "cancelled" state.
            # For simplicity, if we stop, we just clear current_id.
            self._robot_metadata[robot_name]["current_id"] = None

    @_run_async
    def _pick(self, robot_name: str, object_name: str, cmd_id: int) -> None:
        success = self._robots[robot_name].pick_object(object_name)
        self._on_task_complete(robot_name, cmd_id)

    @_run_async
    def _place(self, robot_name: str, cmd_id: int) -> None:
        success = self._robots[robot_name].place_object()
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
        # Search is essentially a delay in pyrobosim
        time_module.sleep(1.0)
        self._on_task_complete(robot_name, cmd_id)


class PyRoboSimBridge:
    """Manages the simulator process and IPC queues."""

    def __init__(
        self,
        world_file: str | Path,
        show_plot: bool = True,
        record_plots: bool = False,
        skip_canvas: bool = False,
    ):
        self.world_file = world_file
        self.show_plot = show_plot
        self.record_plots = record_plots
        self.skip_canvas = skip_canvas
        self.command_queue = multiprocessing.Queue()
        self.status_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(
            target=self._simulator_worker,
            args=(
                self.world_file,
                self.show_plot,
                self.record_plots,
                self.skip_canvas,
                self.command_queue,
                self.status_queue,
            ),
            daemon=True,
        )
        self._latest_status = None

    @staticmethod
    def _simulator_worker(world_file, show_plot, record_plots, skip_canvas, cmd_q, status_q):
        server = PyRoboSimServer(
            world_file, show_plot=show_plot, record_plots=record_plots, skip_canvas=skip_canvas
        )
        server.run(cmd_q, status_q)

    def start(self):
        self.process.start()

    def stop(self):
        self.command_queue.put(("exit",))
        self.process.join(timeout=2.0)
        if self.process.is_alive():
            self.process.terminate()

    def send_command(self, cmd_type: str, *args):
        try:
            self.command_queue.put((cmd_type, *args))
        except (BrokenPipeError, EOFError, ConnectionResetError):
            print("Error: Failed to send command to simulator process (Broken Pipe).")

    def get_latest_status(self) -> Optional[Dict]:
        """Drain status queue and return latest status."""
        if not self.process.is_alive():
            return self._latest_status

        try:
            # Drain the queue to get the absolute latest update
            while not self.status_queue.empty():
                self._latest_status = self.status_queue.get_nowait()
        except (BrokenPipeError, EOFError, ConnectionResetError, Exception):
            pass
        return self._latest_status


class PyRoboSimScene(PhysicalScene):
    """PyRoboSim scene data provider."""

    def __init__(self, world_file: str | Path) -> None:
        self.world_file = world_file
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

            plan = self.robots[robot].path_planner.plan(from_pose, to_pose)

            # Clear the latest path to avoid showing in the plot
            self.robots[robot].path_planner.latest_path = None

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


class DecoupledPyRoboSimEnvironment(PhysicalEnvironment):
    """PyRoboSim environment that runs the simulator in a separate process.

    Uses PyRoboSimBridge for IPC. Maintains state synchronization by mapping
    simulator status snapshots back to PDDL fluents.
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
        self._bridge = PyRoboSimBridge(
            scene.world_file,
            show_plot=show_plot,
            record_plots=record_plots,
            skip_canvas=skip_canvas,
        )
        self._bridge.start()

        # Maps robot_name -> last dispatched command ID
        self._last_dispatched_id: Dict[str, int] = {}
        self._next_command_id = 0

        # Build room mapping for consistency checks
        self._loc_to_room = {}
        for room in self._scene.world.rooms:
            for loc in room.locations:
                self._loc_to_room[loc.name] = room.name
                for spawn in loc.children:
                    self._loc_to_room[spawn.name] = room.name

        # Give simulator a moment to start
        time_module.sleep(1.0)

        # Initial synchronization
        self._latest_status: Optional[Dict] = None
        start_wait = time_module.time()
        while self._latest_status is None and time_module.time() - start_wait < 5.0:
            self._latest_status = self._bridge.get_latest_status()
            time_module.sleep(0.1)

        if self._latest_status:
            self._sync_state(self._latest_status)

    def __del__(self):
        if hasattr(self, "_bridge"):
            self._bridge.stop()

    @property
    def robots(self):
        return self._scene.robots

    @property
    def locations(self):
        return self._scene.locations

    @property
    def objects(self):
        return {obj.name: obj for obj in self._scene.world.objects}

    def save_animation(self, filepath: str | Path) -> None:
        """Saves animation and blocks until simulator process reports completion."""
        print(f"Requesting animation save to {filepath}...")
        self._bridge.send_command("save_animation", str(filepath))

        # 1. Wait for simulator to acknowledge and start saving
        start_wait = time_module.time()
        started = False
        while time_module.time() - start_wait < 10.0:
            status = self._bridge.get_latest_status()
            if status and status.get("is_saving_animation"):
                started = True
                break
            time_module.sleep(0.1)

        if not started:
            # Maybe it finished extremely fast or failed to start
            pass

        # 2. Wait for simulator to finish saving
        start_save = time_module.time()
        while time_module.time() - start_save < 60.0:
            status = self._bridge.get_latest_status()
            if status and not status.get("is_saving_animation"):
                print("Animation save completed.")
                return
            time_module.sleep(0.5)

        print("Warning: Animation save timed out or simulator process unresponsive.")

    def _on_skill_completed(self, skill: ActiveSkill) -> None:
        pass

    def _on_act_loop_iteration(self, dt: float) -> None:
        status = self._bridge.get_latest_status()
        if status:
            self._latest_status = status
            self._sync_state(status)

    def _sync_state(self, status: Dict):
        """Map simulator status back to PDDL fluents."""
        known_robots = self.objects_by_type.get("robot", set())
        for name, robot in status["robots"].items():
            if name not in known_robots:
                continue

            loc = robot["location"]
            loc_type = robot["location_type"]

            # Check execution status using IDs
            last_dispatched = self._last_dispatched_id.get(name, -1)
            completed_id = robot["last_completed_id"]
            current_id = robot["current_id"]

            # We consider the robot "busy" if we sent a command that isn't finished.
            # Finished means: last_completed_id >= last_dispatched
            is_busy = last_dispatched > completed_id

            # Update free status based on our robust ID check
            if is_busy:
                self.fluents.discard(Fluent("free", name))
            else:
                self.fluents.add(Fluent("free", name))

            # Update location - ONLY if not busy (to avoid clobbering planned moves)
            # OR if we are busy but the simulator reports a location update that matches our goal?
            # Actually, safe bet: if not busy, trust simulator. If busy, trust planner/simulator transition.

            if not is_busy:
                if loc_type == "location":
                    # Trust specific locations
                    for f in list(self.fluents):
                        if f.name == "at" and f.args[0] == name and f.args[1] != loc:
                            self.fluents.discard(f)
                    self.fluents.add(Fluent("at", name, loc))
                elif loc_type == "room":
                    # Trust room, but don't clobber specific location if it matches room
                    # If existing loc is in a DIFFERENT room, clobber it.
                    for f in list(self.fluents):
                        if f.name == "at" and f.args[0] == name:
                            existing = f.args[1]

                            # Check room consistency
                            if existing in self._loc_to_room:
                                if self._loc_to_room[existing] != loc:
                                    self.fluents.discard(f)
                            elif existing.endswith("_loc"):
                                # If it's a proxy loc NOT in our mapping (should not happen),
                                # we can't verify room. But if it WAS in mapping, we already checked.
                                pass
                            elif existing not in self.scene.locations:
                                # existing is likely a room name itself
                                if existing != loc:
                                    self.fluents.discard(f)
                            else:
                                # Specific location but not in _loc_to_room?
                                # This shouldn't happen with current init.
                                # Discard to be safe since Room report is authoritative.
                                self.fluents.discard(f)

                    # Add room fluent if no specific one exists
                    has_at = any(f.name == "at" and f.args[0] == name for f in self.fluents)
                    if not has_at:
                        self.fluents.add(Fluent("at", name, loc))

            # 3. Update holding status
            # Always trust simulator for holding
            holding = robot["holding"]
            for f in list(self.fluents):
                if f.name == "holding" and f.args[0] == name:
                    if f.args[1] != holding:
                        self.fluents.discard(f)

            if holding:
                self.fluents.add(Fluent("holding", name, holding))
                self.fluents.add(Fluent("hand-full", name))
            else:
                self.fluents.discard(Fluent("hand-full", name))
                for f in list(self.fluents):
                    if f.name == "holding" and f.args[0] == name:
                        self.fluents.discard(f)

        # Update object fluents
        known_objects = self.objects_by_type.get("object", set())
        for name, obj in status["objects"].items():
            if name not in known_objects:
                continue

            loc = obj["location"]
            loc_type = obj["location_type"]

            if loc_type == "location":
                for f in list(self.fluents):
                    if f.name == "at" and f.args[0] == name and f.args[1] != loc:
                        self.fluents.discard(f)
                self.fluents.add(Fluent("at", name, loc))
            else:
                # If object isn't at a specific location (held, room, moving), remove 'at'
                for f in list(self.fluents):
                    if f.name == "at" and f.args[0] == name:
                        self.fluents.discard(f)

    def execute_skill(self, robot_name: str, skill_name: str, *args: Any, **kwargs: Any) -> None:
        cmd_id = self._next_command_id
        self._next_command_id += 1
        self._last_dispatched_id[robot_name] = cmd_id

        # Optimistic update: mark busy immediately so subsequent logic sees it
        self.fluents.discard(Fluent("free", robot_name))

        # If moving, clear current location immediately to reflect transition
        if skill_name == "move":
            for f in list(self.fluents):
                if f.name == "at" and f.args[0] == robot_name:
                    self.fluents.discard(f)

        self._bridge.send_command("execute", robot_name, skill_name, args, cmd_id)
        # No sleep needed really, local state is updated

    def get_executed_skill_status(self, robot_name: str, skill_name: str) -> SkillStatus:
        # Check our local tracking vs remote status
        if self._latest_status:
            robot = self._latest_status["robots"].get(robot_name)
            if robot:
                last_dispatched = self._last_dispatched_id.get(robot_name, -1)
                last_completed = robot["last_completed_id"]

                # If we dispatched something that isn't complete yet -> RUNNING
                if last_dispatched > last_completed:
                    return SkillStatus.RUNNING

                return SkillStatus.DONE

        # Default fallback
        return SkillStatus.DONE

    def stop_robot(self, robot_name: str) -> None:
        self._bridge.send_command("stop", robot_name)

    def act(
        self,
        action: Action,
        loop_callback_fn: Optional[Callable[[], None]] = None,
        do_interrupt: bool = True,
    ) -> State:
        """Execute action and ensure final state sync before returning."""
        state = super().act(action, loop_callback_fn, do_interrupt)

        # Wait for the action to be fully reflected in the status
        # We need to wait until the simulator confirms the action we just finished IS done.
        # This prevents act() from returning before the final state update (e.g. arrival at goal) reaches us.

        # Identify the robot involved in this action
        parts = action.name.split()
        robot_name = parts[1] if len(parts) > 1 else None

        if robot_name:
            target_id = self._last_dispatched_id.get(robot_name, -1)
            start_wait = time_module.time()
            while time_module.time() - start_wait < 5.0:
                status = self._bridge.get_latest_status()
                if status:
                    self._latest_status = status
                    self._sync_state(status)

                    robot_info = status["robots"].get(robot_name)
                    if robot_info:
                        finished = robot_info["last_completed_id"] >= target_id
                        started = robot_info["current_id"] == target_id
                        if finished or started:
                            break
                time_module.sleep(0.05)

        return self.state


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
        self._plot_frames = [] if record_plots else None

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
        if self._plot_frames is not None:
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
