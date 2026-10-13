"""PyRoboSimScene: a PyRoboSim world described in railroad's terms."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import numpy as np
from pyrobosim.core.locations import ObjectSpawn
from pyrobosim.core.robot import Robot
from pyrobosim.core.yaml_utils import WorldYamlLoader
from pyrobosim.utils.knowledge import graph_node_from_entity, query_to_entity
from pyrobosim.utils.pose import Pose as SimPose
from pyrobosim.utils.trajectory import get_constant_speed_trajectory

from railroad._bindings import Fluent, State
from ..symbolic import LocationRegistry

DEFAULT_WORLD_FILE = Path(__file__).parent / "resources" / "worlds" / "test_world.yaml"


class PyRoboSimScene:
    """A PyRoboSim world, described in railroad's terms.

    Locations are the world's object locations (``table0``, ``my_desk``,
    ...) plus one ``{robot}_loc`` per robot at its starting pose -- the same
    name railroad uses for a robot stopped mid-move, so the
    :attr:`location_registry` coordinates of ``{robot}_loc`` always track
    where that robot was last left.
    """

    def __init__(self, world_file: str | Path = DEFAULT_WORLD_FILE, seed: int | None = None) -> None:
        """Load the world.

        Args:
            world_file: PyRoboSim world YAML.
            seed: Seeds the layout. Objects without a pose in the world file
                are placed at random on load (PyRoboSim samples with NumPy's
                global RNG), and some layouts leave too little room to put
                anything down at a location. None keeps it random.
        """
        if seed is not None:
            np.random.seed(seed)
        self.world = WorldYamlLoader().from_file(Path(world_file))
        self.robots: Dict[str, Robot] = {r.name: r for r in self.world.robots}
        self._planner_locks = {name: threading.Lock() for name in self.robots}
        self._move_time_cache: Dict[Tuple[str, float, float, float, float], float] = {}

        self._location_coords: Dict[str, np.ndarray] = {}
        for location in self.world.locations:
            nav = self._nav_pose_for_entity(next(iter(self.robots.values())), location.name)
            if nav is not None:
                self._location_coords[location.name] = np.array([nav.x, nav.y])
        coords = dict(self._location_coords)
        for name, robot in self.robots.items():
            pose = robot.get_pose()
            coords[f"{name}_loc"] = np.array([pose.x, pose.y])
        self.location_registry = LocationRegistry(coords)

    @property
    def locations(self) -> Dict[str, np.ndarray]:
        """World object locations -> where a robot stands to reach them
        (excludes ``{robot}_loc``). Same shape as ``ProcTHORScene.locations``."""
        return dict(self._location_coords)

    @property
    def objects(self) -> Set[str]:
        return {obj.name for obj in self.world.objects}

    @property
    def object_locations(self) -> Dict[str, Set[str]]:
        """Ground truth: location -> the objects currently on any of its spawns."""
        result: Dict[str, Set[str]] = {}
        for obj in self.world.objects:
            spawn = obj.parent
            if isinstance(spawn, ObjectSpawn) and spawn.parent is not None:
                result.setdefault(spawn.parent.name, set()).add(obj.name)
        return result

    def initial_state(self, robots: Iterable[str] | None = None) -> State:
        """Every robot (or just ``robots``) free at its ``{robot}_loc``."""
        fluents = set()
        for name in self.robots if robots is None else robots:
            fluents |= {
                Fluent("at", name, f"{name}_loc"),
                Fluent("free", name),
                Fluent("revealed", f"{name}_loc"),
            }
        return State(0.0, fluents, [])

    def objects_by_type(self, robots: Iterable[str] | None = None) -> Dict[str, Set[str]]:
        """Objects for planning with every robot (or just ``robots``)."""
        names = set(self.robots if robots is None else robots)
        return {
            "robot": names,
            "location": set(self.locations) | {f"{name}_loc" for name in names},
            "object": self.objects,
        }

    def planner_lock(self, robot: str) -> threading.Lock:
        """Serializes use of a robot's path planner across threads."""
        return self._planner_locks[robot]

    def _nav_pose_for_entity(self, robot: Robot, name: str) -> SimPose | None:
        entity = query_to_entity(
            self.world, name, mode="location", robot=robot, resolution_strategy="nearest"
        )
        if entity is None:
            return None
        node = graph_node_from_entity(self.world, entity, robot=robot)
        return None if node is None else node.pose

    def nav_pose(self, robot: str, location: str) -> SimPose:
        """Where ``robot`` should stand to be "at" ``location``."""
        if location in self.locations:
            pose = self._nav_pose_for_entity(self.robots[robot], location)
            if pose is None:
                raise ValueError(f"No navigable pose for location {location!r}")
            return pose
        coords = self.location_registry.get(location)
        if coords is None:
            raise ValueError(f"Unknown location {location!r}")
        xy = np.asarray(coords, dtype=float).reshape(-1)
        return SimPose(x=float(xy[0]), y=float(xy[1]))

    def move_time(self, robot: str, loc_from: str, loc_to: str) -> float:
        """Travel time (seconds) along the robot's planned trajectory;
        ``inf`` if unreachable."""
        try:
            start = self.nav_pose(robot, loc_from)
            goal = self.nav_pose(robot, loc_to)
        except ValueError:
            return float("inf")
        return self._move_time(robot, start.x, start.y, goal.x, goal.y)

    def _move_time(self, robot: str, x0: float, y0: float, x1: float, y1: float) -> float:
        key = (robot, x0, y0, x1, y1)
        if key not in self._move_time_cache:
            self._move_time_cache[key] = self._plan_move_time(*key)
        return self._move_time_cache[key]

    def _plan_move_time(self, robot: str, x0: float, y0: float, x1: float, y1: float) -> float:
        sim_robot = self.robots[robot]
        planner = sim_robot.path_planner
        if planner is None:
            return float("inf")
        with self.planner_lock(robot):
            path = planner.plan(SimPose(x=x0, y=y0), SimPose(x=x1, y=y1))
            planner.latest_path = None  # ty: ignore[invalid-assignment]
        if path is None or path.num_poses == 0:
            return float("inf")
        executor = sim_robot.path_executor
        # The duration of the trajectory the robot will actually follow --
        # including turning in place, which path length alone misses.
        trajectory = get_constant_speed_trajectory(
            path,
            linear_velocity=getattr(executor, "linear_velocity", 1.0),
            max_angular_velocity=getattr(executor, "max_angular_velocity", None),
        )
        if trajectory is None or trajectory.num_points() == 0:
            return float("inf")
        return float(trajectory.t_pts[-1])
