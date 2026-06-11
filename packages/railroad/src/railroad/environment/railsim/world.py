"""World models: shapely-polygon worlds built from occupancy grids."""

from __future__ import annotations

import math
import random
import warnings

import numpy as np
import scipy.ndimage
import shapely.geometry

from .environments.base import MapData
from .geometry import obstacles_and_boundary_from_occupancy_grid
from .pose import Pose


class World:
    """Stores the shapely polygons that define the world.

    Attributes:
        obstacles: list of shapely polygons (includes the boundary).
        boundary: polygon defining the outer wall of the world.
        known_space_poly: polygon of free space (boundary minus obstacles).
    """

    def __init__(self,
                 obstacles: list[shapely.geometry.Polygon],
                 boundary: shapely.geometry.Polygon):
        self.boundary = boundary
        self._internal_obstacles = list(obstacles)
        self.obstacles = list(obstacles) + [boundary]
        self.known_space_poly = boundary
        for obs in self._internal_obstacles:
            self.known_space_poly = self.known_space_poly.difference(obs)
        self.area = self.known_space_poly.area

    @property
    def map_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """((xmin, xmax), (ymin, ymax)) of the boundary polygon."""
        xs, ys = self.boundary.exterior.xy
        return (min(xs), max(xs)), (min(ys), max(ys))

    def get_signed_dist(self, pose: Pose) -> float:
        """Signed distance from a point to the nearest wall (positive =
        inside free space)."""
        point = shapely.geometry.Point([pose.x, pose.y])
        distance = 1e10
        for obstacle in self._internal_obstacles:
            if obstacle.contains(point):
                obs_dist = -obstacle.exterior.distance(point)
            else:
                obs_dist = obstacle.distance(point)
            distance = min(distance, obs_dist)

        if self.boundary.contains(point):
            boundary_dist = self.boundary.exterior.distance(point)
        else:
            boundary_dist = -self.boundary.distance(point)
        return min(distance, boundary_dist)

    def get_random_pose(self,
                        rng: random.Random,
                        min_signed_dist: float = 0,
                        num_attempts: int = 10000) -> Pose:
        """Random pose in free space, at least ``min_signed_dist`` from walls."""
        (xmin, xmax), (ymin, ymax) = self.map_bounds
        for _ in range(num_attempts):
            pose = Pose(rng.uniform(xmin, xmax), rng.uniform(ymin, ymax),
                        yaw=2 * math.pi * rng.random())
            if self.get_signed_dist(pose) >= min_signed_dist:
                return pose
        raise ValueError("Could not find random pose within bounds")


class OccupancyGridWorld(World):
    """World built from a generated map, with breadcrumb guidance markers
    sampled along the goal path."""

    def __init__(self,
                 map_data: MapData,
                 num_breadcrumb_elements: int = 2000,
                 min_breadcrumb_signed_distance: float | None = None,
                 min_interlight_distance: float | None = None,
                 min_light_to_wall_distance: float = 2.0,
                 seed: int | None = None):
        self.map_data = map_data
        self.resolution = map_data.resolution
        self.palette = map_data.palette
        if min_breadcrumb_signed_distance is None:
            min_breadcrumb_signed_distance = 4.0 * self.resolution
        if min_interlight_distance is None:
            # Maps carry their own default (3.0 m for the maze, matching
            # the old sim; 6.0 m for the sparser office).
            min_interlight_distance = map_data.min_interlight_distance

        free_space = (map_data.occ_grid < 0.5).astype(float)
        obstacles, boundary = obstacles_and_boundary_from_occupancy_grid(
            free_space, self.resolution)
        super().__init__(obstacles=obstacles, boundary=boundary)

        # Tables are obstacles for collision/pose sampling (occ_grid marks
        # them occupied) but render as table boxes, not walls -- and lights
        # may hang above them. Both use table-free occupancy, as the old sim.
        self.table_poses_sizes = list(map_data.tables)
        self.wall_obstacles = self._internal_obstacles
        self.wall_boundary = self.boundary
        light_grid = map_data.occ_grid
        if self.table_poses_sizes:
            clutter = map_data.semantic_labels['clutter']
            no_tables = map_data.occ_grid.copy()
            no_tables[map_data.semantic_grid == clutter] = 0.0
            light_grid = no_tables
            self.wall_obstacles, self.wall_boundary = (
                obstacles_and_boundary_from_occupancy_grid(
                    (no_tables < 0.5).astype(float), self.resolution))

        rng = random.Random(seed)
        self.breadcrumb_element_poses: list[Pose] = []
        if (num_breadcrumb_elements > 0
                and 'goal_path' not in map_data.semantic_labels):
            warnings.warn("This map has no 'goal_path' semantic label "
                          "(office-style maps have no breadcrumbs); "
                          "skipping breadcrumb sampling.")
            num_breadcrumb_elements = 0
        while len(self.breadcrumb_element_poses) < num_breadcrumb_elements:
            pose = self.get_random_pose_on_label(
                rng, semantic_label='goal_path',
                min_signed_dist=min_breadcrumb_signed_distance)
            self.breadcrumb_element_poses.append(pose)

        self.light_poses = _generate_light_poses(
            light_grid, self.resolution,
            min_wall_distance=min_light_to_wall_distance,
            min_interlight_distance=min_interlight_distance,
            rng=np.random.default_rng(seed))

    def semantic_label_at(self, pose: Pose) -> int:
        i = int(round(pose.x / self.resolution))
        j = int(round(pose.y / self.resolution))
        grid = self.map_data.semantic_grid
        i = min(max(i, 0), grid.shape[0] - 1)
        j = min(max(j, 0), grid.shape[1] - 1)
        return int(grid[i, j])

    def get_random_pose_on_label(self,
                                 rng: random.Random,
                                 semantic_label: str,
                                 min_signed_dist: float = 0,
                                 num_attempts: int = 10000) -> Pose:
        """Random pose constrained to cells with a given semantic label."""
        label_value = self.map_data.semantic_labels[semantic_label]
        for _ in range(num_attempts):
            pose = self.get_random_pose(rng, min_signed_dist=min_signed_dist)
            if self.semantic_label_at(pose) == label_value:
                return pose
        raise ValueError(
            f"Could not find random pose with semantic label '{semantic_label}'")


def _generate_light_poses(occ_grid: np.ndarray,
                          resolution: float,
                          min_wall_distance: float,
                          min_interlight_distance: float,
                          rng: np.random.Generator,
                          max_samples: int = 10000) -> list[tuple[float, float]]:
    """Ceiling light positions, replicating the old sim's dart throwing:
    ``max_samples`` random poses with enough wall clearance, keeping those
    far enough from every previously accepted light. The throw count is
    part of the look -- exhaustively saturating the space instead yields a
    noticeably denser light field on large (office-scale) maps."""
    free = occ_grid < 0.5
    # Distance from each free cell center to the nearest wall *face*,
    # matching the old shapely signed distance (cell centers sit half a
    # cell from the face).
    wall_dist = (scipy.ndimage.distance_transform_edt(free) - 0.5) * resolution
    eligible = np.argwhere(wall_dist >= min_wall_distance)
    if len(eligible) == 0:
        return []
    idx = rng.integers(0, len(eligible), size=max_samples)
    points = (eligible[idx] + rng.uniform(-0.5, 0.5, (max_samples, 2))) * resolution

    lights: list[tuple[float, float]] = []
    accepted = np.empty((0, 2))
    for xy in points:
        if len(lights) == 0 or np.all(
                np.hypot(*(accepted - xy).T) >= min_interlight_distance):
            lights.append((float(xy[0]), float(xy[1])))
            accepted = np.vstack([accepted, xy])
    return lights


def world_from_occupancy_grid(occ_grid: np.ndarray, resolution: float) -> World:
    """Build a plain `World` (no breadcrumbs/semantics) from an occupancy
    grid where 1 = occupied, 0 = free."""
    free_space = (occ_grid < 0.5).astype(float)
    obstacles, boundary = obstacles_and_boundary_from_occupancy_grid(
        free_space, resolution)
    return World(obstacles=obstacles, boundary=boundary)
