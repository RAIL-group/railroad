"""Self-contained PyRoboSim environment implementing Environment protocol."""

from __future__ import annotations

import threading
import time as time_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Set, Tuple

from railroad._bindings import Action, Fluent, GroundedEffect
from railroad.environment.skill import ActiveSkill, Environment, SymbolicSkill

if TYPE_CHECKING:
    import pyrobosim


class PhysicalSkill:
    """Skill that executes on a physical robot and polls for completion.

    Implements ActiveSkill protocol for physical robot execution where:
    - Physical actions run asynchronously via pyrobosim
    - Completion is detected by polling robot.is_busy()
    - Actual duration may differ from planned duration
    """

    def __init__(
        self,
        action: Action,
        start_time: float,
        robot: str,
        physical_env: "PyRoboSimEnvironment",
        skill_name: str,
        skill_args: tuple,
    ) -> None:
        self._action = action
        self._start_time = start_time
        self._robot = robot
        self._physical_env = physical_env
        self._skill_name = skill_name
        self._is_done = False
        self._is_interruptible = skill_name == "move"
        self._completion_time: float | None = None

        # Track wall-clock time for actual duration
        self._wall_start_time = time_module.time()

        # Compute upcoming effects from action (using planned times initially)
        self._upcoming_effects: List[Tuple[float, GroundedEffect]] = sorted(
            [(start_time + eff.time, eff) for eff in action.effects],
            key=lambda el: el[0]
        )

        # Start physical execution
        physical_env._execute_skill(robot, skill_name, *skill_args)

    @property
    def robot(self) -> str:
        return self._robot

    @property
    def is_done(self) -> bool:
        return self._is_done

    @property
    def is_interruptible(self) -> bool:
        return self._is_interruptible

    @property
    def upcoming_effects(self) -> List[Tuple[float, GroundedEffect]]:
        return self._upcoming_effects

    def _is_physical_action_done(self) -> bool:
        """Check if the physical robot action is complete."""
        # For no_op, check our own flag
        if self._skill_name == "no_op":
            done = not self._physical_env._is_no_op_running.get(self._robot, False)
        else:
            # For other skills, check if robot is busy
            robot = self._physical_env._robots.get(self._robot)
            done = robot is None or not robot.is_busy()

        # Record completion time when first detected
        if done and self._completion_time is None:
            actual_duration = time_module.time() - self._wall_start_time
            self._completion_time = self._start_time + actual_duration

        return done

    @property
    def completion_time(self) -> float | None:
        """Return the actual completion time (start_time + actual_duration)."""
        return self._completion_time

    @property
    def time_to_next_event(self) -> float:
        if not self._upcoming_effects:
            return float("inf")

        # Check for immediate effects (time=0 or time <= start_time)
        next_effect_time = self._upcoming_effects[0][0]
        if next_effect_time <= self._start_time + 1e-9:
            return next_effect_time

        # For completion effects, wait for physical action to be done
        if self._is_physical_action_done():
            # Return actual completion time instead of planned time
            return self._completion_time if self._completion_time else next_effect_time

        # Physical action still running - poll again soon
        # Return a time slightly after start to keep polling
        return self._start_time + 0.01

    def advance(self, time: float, env: Environment) -> None:
        """Advance skill, applying effects based on time and physical completion.

        - Immediate effects (time <= start_time): Apply based on time
        - Completion effects (time > start_time): Apply only when physical action done
        """
        if not self._upcoming_effects:
            return

        # Apply immediate effects (time=0) based on time alone
        immediate_effects = [
            (t, eff) for t, eff in self._upcoming_effects
            if t <= self._start_time + 1e-9 and t <= time + 1e-9
        ]
        for _, effect in immediate_effects:
            env.apply_effect(effect)

        # Remove applied immediate effects
        if immediate_effects:
            self._upcoming_effects = [
                (t, eff) for t, eff in self._upcoming_effects
                if t > self._start_time + 1e-9 or t > time + 1e-9
            ]

        # For completion effects, wait for physical action to be done
        if self._is_physical_action_done() and self._upcoming_effects:
            # Physical action complete - apply remaining effects
            for _, effect in self._upcoming_effects:
                env.apply_effect(effect)
            self._upcoming_effects = []

        # Mark done when no more effects and physical action complete
        if not self._upcoming_effects and self._is_physical_action_done():
            self._is_done = True

    def interrupt(self, env: Environment) -> None:
        """Interrupt this skill by stopping the physical robot."""
        if self._is_interruptible and not self._is_done:
            self._physical_env._stop_robot(self._robot)
            # Mark as done without applying remaining effects
            self._upcoming_effects = []
            self._is_done = True


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


class PyRoboSimEnvironment:
    """Self-contained PyRoboSim environment implementing Environment protocol.

    Loads world from YAML and derives initial state automatically:
    - Robot fluents: at(robot, robot_loc), free(robot), revealed(robot_loc)
    - Objects by type derived from world entities
    """

    def __init__(
        self,
        world_file: str,
        show_plot: bool = True,
        record_plots: bool = False,
    ) -> None:
        """Initialize the environment from a world file.

        Args:
            world_file: Path to the PyRoboSim world YAML file.
            show_plot: Whether to show the matplotlib plot.
            record_plots: Whether to record frames for video export.
        """
        from pyrobosim.core.yaml_utils import WorldYamlLoader
        from pyrobosim.utils.knowledge import query_to_entity, graph_node_from_entity

        self._world = WorldYamlLoader().from_file(Path(world_file))

        # Store initial robot locations (before they move)
        self._initial_robot_locations = {
            f"{r.name}_loc": r.pose for r in self._world.robots
        }

        # Build location dictionary
        self._locations = {loc.name: loc.pose for loc in self._world.locations}
        self._locations.update(self._initial_robot_locations)

        # Build robot dictionary
        self._robots = {robot.name: robot for robot in self._world.robots}
        self._is_robot_assigned = {robot: False for robot in self._robots}
        self._is_no_op_running: Dict[str, bool] = {robot: False for robot in self._robots}

        # Derive objects_by_type from world
        self._objects_by_type: Dict[str, Set[str]] = {
            "robot": set(self._robots.keys()),
            "location": set(self._locations.keys()),
            "object": {obj.name for obj in self._world.objects},
        }

        # Derive initial fluents from world state
        self._fluents: Set[Fluent] = set()
        for robot_name in self._robots:
            robot_loc = f"{robot_name}_loc"
            self._fluents.add(Fluent("at", robot_name, robot_loc))
            self._fluents.add(Fluent("free", robot_name))
            self._fluents.add(Fluent("revealed", robot_loc))

        # Initialize canvas for visualization
        self.canvas = MatplotlibWorldCanvas(self._world, show_plot, record_plots)

    @property
    def fluents(self) -> Set[Fluent]:
        """Current ground truth fluents."""
        return self._fluents

    @property
    def objects_by_type(self) -> Dict[str, Set[str]]:
        """All known objects, organized by type."""
        return self._objects_by_type

    def create_skill(self, action: Action, time: float) -> ActiveSkill:
        """Create a PhysicalSkill that executes on the robot."""
        parts = action.name.split()
        action_type = parts[0]
        robot = parts[1]

        if action_type in {'move', 'search', 'pick', 'place', 'no_op'}:
            return PhysicalSkill(
                action=action,
                start_time=time,
                robot=robot,
                physical_env=self,
                skill_name=action_type,
                skill_args=parts[1:],
            )
        else:
            raise NotImplementedError("Action type not found; unsupported behavior.")
            # # Default: use symbolic skill
            # return SymbolicSkill(
            #     action=action,
            #     start_time=time,
            #     robot=robot,
            # )

    def apply_effect(self, effect: GroundedEffect) -> None:
        """Apply effect fluents to the state."""
        for fluent in effect.resulting_fluents:
            if fluent.negated:
                self._fluents.discard(~fluent)
            else:
                self._fluents.add(fluent)

        # Handle probabilistic effects
        if effect.is_probabilistic:
            nested_effects, _ = self.resolve_probabilistic_effect(effect, self._fluents)
            for nested in nested_effects:
                self.apply_effect(nested)

        # Handle revelation from search
        self._handle_revelation()

    def _handle_revelation(self) -> None:
        """Reveal objects when locations are searched."""
        for fluent in list(self._fluents):
            if fluent.name == "searched":
                location = fluent.args[0]
                revealed_fluent = Fluent("revealed", location)

                if revealed_fluent not in self._fluents:
                    self._fluents.add(revealed_fluent)

                    # Get objects from physical environment
                    objects_at_loc = self.get_objects_at_location(location)
                    for obj in objects_at_loc.get("object", set()):
                        self._fluents.add(Fluent("found", obj))
                        self._fluents.add(Fluent("at", obj, location))
                        self._objects_by_type.setdefault("object", set()).add(obj)

    def resolve_probabilistic_effect(
        self,
        effect: GroundedEffect,
        current_fluents: Set[Fluent],
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        """Resolve probabilistic effects using physical environment ground truth."""
        if not effect.is_probabilistic:
            return [effect], current_fluents

        branches = effect.prob_effects
        if not branches:
            return [], current_fluents

        # For search, check physical environment for object presence
        for _, branch_effects in branches:
            for branch_eff in branch_effects:
                for fluent in branch_eff.resulting_fluents:
                    if fluent.name == "found" and not fluent.negated:
                        target_object = fluent.args[0]
                        # Find location from "at" fluent
                        location = self._find_search_location(branch_eff, target_object)
                        if location:
                            objects_at_loc = self.get_objects_at_location(location)
                            if target_object in objects_at_loc.get("object", set()):
                                return list(branch_effects), current_fluents

        # Failure branch (last one)
        _, last_branch = branches[-1]
        return list(last_branch), current_fluents

    def _find_search_location(self, effect: GroundedEffect, target_object: str) -> str | None:
        """Find location from 'at object location' fluent."""
        for fluent in effect.resulting_fluents:
            if fluent.name == "at" and len(fluent.args) >= 2:
                if fluent.args[0] == target_object:
                    return fluent.args[1]
        return None

    def get_objects_at_location(self, location: str) -> Dict[str, Set[str]]:
        """Get objects at a location (ground truth for search resolution)."""
        objects: Dict[str, Set[str]] = {"object": set()}
        if location.endswith("_loc"):
            # Our custom robot locations (not defined in pyrobosim) have no objects
            return objects
        loc = self._world.get_location_by_name(location)
        if loc:
            for spawn in loc.children:
                for obj in spawn.children:
                    objects["object"].add(obj.name)
        return objects

    def get_move_cost_fn(self) -> Callable[[str, str, str], float]:
        """Get a function that computes move costs based on path length.

        Returns:
            A function (robot, loc_from, loc_to) -> float that returns the
            estimated time to move based on path length and robot velocity.
        """
        from pyrobosim.utils.knowledge import query_to_entity, graph_node_from_entity

        def get_move_time(robot: str, loc_from: str, loc_to: str) -> float:
            # Get feasible poses for the robot at the from and to locations
            from_pose = self._get_feasible_pose_from_location_for_robot(
                self._robots[robot], loc_from
            )
            to_pose = self._get_feasible_pose_from_location_for_robot(
                self._robots[robot], loc_to
            )

            plan = self._robots[robot].path_planner.plan(from_pose, to_pose)  # type: ignore[union-attr]

            # Clear the latest path to avoid showing in the plot
            self._robots[robot].path_planner.latest_path = None  # type: ignore[union-attr]

            if plan is None:
                return float("inf")
            return plan.length / 1.0  # robot velocity = 1.0

        return get_move_time

    def _get_feasible_pose_from_location_for_robot(
        self, robot: "pyrobosim.core.robot.Robot", location_name: str
    ) -> "pyrobosim.utils.pose.Pose":
        """Get a feasible pose for a robot at a given location."""
        from pyrobosim.utils.knowledge import query_to_entity, graph_node_from_entity

        if location_name in self._initial_robot_locations:
            return self._initial_robot_locations[location_name]

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
            raise ValueError(
                f"Could not find graph node associated with location '{location_name}'."
            )
        return goal_node.pose

    def _execute_skill(self, robot_name: str, skill_name: str, *args: Any) -> None:
        """Execute a skill on a robot (internal method called by PhysicalSkill)."""
        self._is_robot_assigned[robot_name] = True
        if skill_name == "pick":
            object_name = args[1]
            self._pick(robot_name, object_name)
        elif skill_name == "place":
            self._place(robot_name)
        elif skill_name == "move":
            loc_to = args[1]
            self._move(robot_name, loc_to)
        elif skill_name == "search":
            self._search(robot_name)
        elif skill_name == "no_op":
            self._no_op(robot_name)
        else:
            raise ValueError(f"Skill '{skill_name}' not recognized.")
        time_module.sleep(0.1)  # Give some time for the skill to start

    def _stop_robot(self, robot_name: str) -> None:
        """Stop a robot's current action."""
        self._is_robot_assigned[robot_name] = False
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
        self._robots[robot_name].navigate(goal=loc_to)

    @_run_async
    def _search(self, robot_name: str) -> None:
        # No need to search for now since all objects are assumed to be known
        pass

    @_run_async
    def _no_op(self, robot_name: str) -> None:
        self._is_no_op_running[robot_name] = True
        time_module.sleep(1.0)  # Default no-op duration
        self._is_no_op_running[robot_name] = False


class MatplotlibWorldCanvas:
    """Matplotlib-based visualization canvas for PyRoboSim worlds.

    Replaces Qt-based WorldCanvas to support non-blocking plotting.
    Provides multi-robot path drawing and frame recording for video export.

    Note: This class uses duck-typing to work with WorldCanvas methods without
    inheriting from it, to avoid Qt signal initialization issues.
    """

    # Class attributes required by WorldCanvas methods
    robot_zorder = 3
    robot_dir_line_factor = 3.0

    class MockSignal:
        def __init__(self, callback: Callable[[], None]) -> None:
            self.callback = callback

        def emit(self, *args: Any, **kwargs: Any) -> None:
            self.callback()

    class MockMainWindow:
        def get_current_robot(self) -> None:
            return None

    def __init__(
        self,
        world: Any,  # pyrobosim.core.world.World
        show_plot: bool = True,
        record_plots: bool = False,
    ) -> None:
        import matplotlib.pyplot as plt
        from pyrobosim.gui.options import WorldCanvasOptions

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
        # Required by WorldCanvas methods
        self.draw_lock = threading.RLock()
        plt.ion()

        # Hijack the signals BEFORE calling any methods like show()
        # This prevents the "Signal source has been deleted" error
        self.draw_signal = self.MockSignal(self.draw_signal_callback)
        # Add other signals as needed to prevent attribute errors
        self.show_robots_signal = self.MockSignal(self.show_robots)
        self.show_planner_and_path_signal = self.MockSignal(self._show_all_paths)

        # Manually initialize the artist lists from WorldCanvas since we do not call its __init__
        self.path_artists_storage: Dict[str, List[Any]] = {}
        self.robot_bodies: List[Any] = []
        self.robot_dirs: List[Any] = []
        self.robot_lengths: List[Any] = []
        self.robot_texts: List[Any] = []
        self.robot_sensor_artists: List[Any] = []
        self.obj_patches: List[Any] = []
        self.obj_texts: List[Any] = []
        self.hallway_patches: List[Any] = []
        self.room_patches: List[Any] = []
        self.room_texts: List[Any] = []
        self.location_patches: List[Any] = []
        self.location_texts: List[Any] = []
        self.path_planner_artists: Dict[str, List[Any]] = {"graph": [], "path": []}

        self.show()
        self.axes.autoscale()
        self.axes.axis("equal")
        self._plot_frames: List[Any] = []

    def show_robots(self) -> None:
        """Show robots using WorldCanvas method."""
        from pyrobosim.gui.world_canvas import WorldCanvas
        WorldCanvas.show_robots(self)  # type: ignore[arg-type]

    def show_rooms(self) -> None:
        """Show rooms using WorldCanvas method."""
        from pyrobosim.gui.world_canvas import WorldCanvas
        WorldCanvas.show_rooms(self)  # type: ignore[arg-type]

    def show_hallways(self) -> None:
        """Show hallways using WorldCanvas method."""
        from pyrobosim.gui.world_canvas import WorldCanvas
        WorldCanvas.show_hallways(self)  # type: ignore[arg-type]

    def show_locations(self) -> None:
        """Show locations using WorldCanvas method."""
        from pyrobosim.gui.world_canvas import WorldCanvas
        WorldCanvas.show_locations(self)  # type: ignore[arg-type]

    def show_objects(self) -> None:
        """Show objects using WorldCanvas method."""
        from pyrobosim.gui.world_canvas import WorldCanvas
        WorldCanvas.show_objects(self)  # type: ignore[arg-type]

    def update_robots_plot(self) -> None:
        """Update robots plot using WorldCanvas method."""
        from pyrobosim.gui.world_canvas import WorldCanvas
        WorldCanvas.update_robots_plot(self)  # type: ignore[arg-type]

    def update_object_plot(self, obj: Any) -> None:
        """Update object plot using WorldCanvas method."""
        from pyrobosim.gui.world_canvas import WorldCanvas
        WorldCanvas.update_object_plot(self, obj)  # type: ignore[arg-type]

    def _show_all_paths(self) -> None:
        """Custom method to draw paths for EVERY robot in the world."""
        for robot in self.world.robots:
            self._draw_single_robot_path(robot)

    def _draw_single_robot_path(self, robot: Any) -> None:
        """Helper to draw a specific robot's path without clearing others."""
        from pyrobosim.navigation.visualization import plot_path_planner

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
        # Call wrapper methods
        self.show_rooms()
        self.show_hallways()
        self.show_locations()
        self.show_objects()
        self.show_robots()
        self.update_robots_plot()

        self._show_all_paths()  # Show paths for all robots

        self.axes.autoscale()
        self.axes.axis("equal")

    def draw_signal_callback(self) -> None:
        """Replacement for the Qt signal execution."""
        if hasattr(self, "fig"):
            self.fig.canvas.draw_idle()

    def update(self) -> None:
        """Updates the world visualization in a loop."""
        import matplotlib.pyplot as plt

        self.show()
        self.fig.canvas.draw_idle()
        self._plot_frames.append(self._get_frame())
        if self.show_plot:
            plt.pause(self.options.animation_dt)

    def wait_for_close(self) -> None:
        """Blocks until the plot window is closed."""
        import matplotlib.pyplot as plt

        if not self.show_plot:
            return
        plt.ioff()
        plt.show()

    def _get_frame(self) -> Any:
        """Captures the current frame as an image array."""
        import numpy as np

        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        width = int(renderer.width)
        height = int(renderer.height)
        image = np.frombuffer(self.fig.canvas.tostring_argb(), dtype="uint8")
        image = image.reshape((height, width, 4))
        return image[..., 1:4]  # Convert ARGB to RGB

    def save_animation(self, filepath: str) -> None:
        """Saves the recorded frames as a video file."""
        import warnings
        import numpy as np

        if not self._plot_frames:
            warnings.warn(
                "No frames recorded to save animation. Use 'record_plots=True' to record plot frames."
            )
            return

        import imageio
        from PIL import Image

        fps = int(round(1 / self.options.animation_dt))

        # Use the first frame's size as the target
        target_size = (self._plot_frames[0].shape[1], self._plot_frames[0].shape[0])

        filepath_path = Path(filepath)
        filepath_path.parent.mkdir(parents=True, exist_ok=True)

        # Add ./ to relative paths, for easier CLI use
        filepath_str = (
            str(filepath_path)
            if filepath_path.as_posix().startswith(("/", "./", "../"))
            else f"./{filepath_path}"
        )

        writer = imageio.get_writer(
            filepath,
            format="ffmpeg",  # type: ignore[arg-type]  # imageio accepts string format names
            mode="I",
            fps=fps,
            codec="libx264",
            macro_block_size=None,
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
