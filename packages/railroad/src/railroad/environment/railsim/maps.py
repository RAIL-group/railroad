"""Scene wrappers exposing railsim worlds to railroad environments.

``RailsimScene`` mirrors the ``ProcTHORScene`` data-provider surface
(``.grid``, ``.locations``, ``.object_locations``) so railsim's maze and
office worlds plug into the unknown-space exploration stack, and adds a
lazily-created :class:`Simulator` for rendering panoramic images.

Unit conventions: railroad environments work in grid-cell coordinates
(``Pose.x`` is the row index); railsim renders in meters where
``grid[i, j] <-> (x = i * resolution, y = j * resolution)``. The conversion
is a pure scaling by ``resolution`` with yaw unchanged
(:meth:`RailsimScene.cell_pose_to_meters`).

Navigation-vs-rendering tradeoff: the navigation grid (``.grid``) is the raw
occupancy grid inflated by ``inflation_radius_m``, so the laser senses and
paths against inflated walls while panoramas render the raw geometry.
Observed free space therefore ends ~``inflation_radius_m`` short of the true
walls; pass ``inflation_radius_m=0`` to sense the exact geometry instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Set, Tuple

import numpy as np

from railroad.environment.types import PoseLike, TopDownView
from railroad.navigation.pathing import inflate_grid

from .environments import (
    GuidedMazeConfig,
    MapData,
    OfficeConfig,
    make_guided_maze,
    make_office,
)
from .pose import Pose as SimPose
from .simulator import Simulator, SimulatorConfig
from .world import OccupancyGridWorld

if TYPE_CHECKING:
    pass

# Default inflation matches the generators' connectivity guarantee: both the
# maze and office samplers ensure start/goal remain connected after inflating
# by their config's inflation_radius_m (0.75 m by default).
_DEFAULT_INFLATION_RADIUS_M = 0.75


class RailsimScene:
    """Data provider for railsim worlds (maze/office) with pano rendering.

    Exposes the same surface the frontier-search stack consumes from
    ``ProcTHORScene``: ``grid`` (navigation occupancy grid), ``locations``
    (named grid coordinates including ``'start_loc'``), and
    ``object_locations`` (empty; railsim scenes are exploration-only).
    """

    def __init__(
        self,
        map_data: MapData,
        *,
        inflation_radius_m: float | None = None,
        simulator_config: SimulatorConfig | None = None,
        num_breadcrumb_elements: int = 0,
        seed: int | None = None,
    ) -> None:
        self._map_data = map_data
        self._simulator_config = simulator_config
        self._num_breadcrumb_elements = num_breadcrumb_elements
        self._seed = seed
        self._simulator: Simulator | None = None

        if inflation_radius_m is None:
            inflation_radius_m = _DEFAULT_INFLATION_RADIUS_M
        self._inflation_radius_m = float(inflation_radius_m)

        self._raw_grid = np.asarray(map_data.occ_grid, dtype=float)
        inflation_radius_cells = self._inflation_radius_m / map_data.resolution
        if inflation_radius_cells > 0:
            self._grid = inflate_grid(self._raw_grid, inflation_radius_cells)
        else:
            self._grid = self._raw_grid.copy()

        self._locations: Dict[str, Tuple[int, int]] = {
            "start_loc": (int(map_data.start_cell[0]), int(map_data.start_cell[1])),
            "goal_loc": (int(map_data.end_cell[0]), int(map_data.end_cell[1])),
        }

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def maze(
        cls,
        seed: int | None = None,
        config: GuidedMazeConfig | None = None,
        *,
        inflation_radius_m: float | None = None,
        simulator_config: SimulatorConfig | None = None,
        num_breadcrumb_elements: int = 2000,
    ) -> "RailsimScene":
        """Generate a guided-maze scene (breadcrumbs mark the goal path)."""
        config = config or GuidedMazeConfig()
        map_data, _, _ = make_guided_maze(seed=seed, config=config)
        return cls(
            map_data,
            inflation_radius_m=inflation_radius_m,
            simulator_config=simulator_config,
            num_breadcrumb_elements=num_breadcrumb_elements,
            seed=seed,
        )

    @classmethod
    def office(
        cls,
        seed: int | None = None,
        config: OfficeConfig | None = None,
        *,
        inflation_radius_m: float | None = None,
        simulator_config: SimulatorConfig | None = None,
    ) -> "RailsimScene":
        """Generate an office scene (no breadcrumbs; offices have no goal path)."""
        config = config or OfficeConfig()
        map_data, _, _ = make_office(seed=seed, config=config)
        return cls(
            map_data,
            inflation_radius_m=inflation_radius_m,
            simulator_config=simulator_config,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Scene-provider surface (mirrors ProcTHORScene)
    # ------------------------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        """Navigation occupancy grid (inflated; 0=free, 1=occupied)."""
        return self._grid

    @property
    def raw_grid(self) -> np.ndarray:
        """Raw (uninflated) occupancy grid matching the rendered geometry."""
        return self._raw_grid

    @property
    def locations(self) -> Dict[str, Tuple[int, int]]:
        """Named grid coordinates: ``start_loc`` and ``goal_loc``."""
        return self._locations

    @property
    def object_locations(self) -> Dict[str, Set[str]]:
        """Ground-truth object placements (empty; exploration-only scenes)."""
        return {}

    @property
    def map_data(self) -> MapData:
        return self._map_data

    @property
    def resolution(self) -> float:
        """Meters per grid cell."""
        return float(self._map_data.resolution)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @property
    def simulator(self) -> Simulator:
        """The visual simulator, created lazily on first access.

        The underlying OpenGL context is created on the first render call,
        so merely constructing the scene never requires a GPU/display.
        """
        if self._simulator is None:
            world = OccupancyGridWorld(
                self._map_data,
                num_breadcrumb_elements=self._num_breadcrumb_elements,
                seed=self._seed,
            )
            self._simulator = Simulator(world, self._simulator_config)
        return self._simulator

    def cell_pose_to_meters(self, pose: PoseLike) -> SimPose:
        """Convert a grid-cell pose (railroad) to a meter pose (railsim)."""
        res = self.resolution
        return SimPose(x=float(pose.x) * res, y=float(pose.y) * res, yaw=float(pose.yaw))

    def get_pano_image(self, pose_cells: PoseLike) -> np.ndarray:
        """Render an RGB panorama at a grid-cell pose.

        Returns a robot-aligned equirectangular image of shape
        ``(pano_height, pano_width, 3)`` uint8; the center column looks
        along the robot's heading.
        """
        return self.simulator.get_pano_image(self.cell_pose_to_meters(pose_cells))

    def get_top_down_image(self, orthographic: bool = True) -> np.ndarray:
        """Semantic top-down map image (for dashboard overhead panels)."""
        del orthographic  # always orthographic; kwarg matches ProcTHORScene
        palette = self._map_data.palette
        semantic = self._map_data.semantic_grid
        occupied = self._raw_grid >= 0.5
        floor_color = np.array(palette.get("floor", (0.7, 0.8, 0.87)))
        wall_color = np.array(palette.get("wall", (0.58, 0.58, 0.58)))

        image = np.empty((*semantic.shape, 3), dtype=float)
        image[...] = floor_color
        # Tint free regions toward their class wall color (hallway/room/...).
        for name, label in self._map_data.semantic_labels.items():
            tint = palette.get(f"wall_{name}")
            if tint is None:
                continue
            mask = (semantic == label) & ~occupied
            image[mask] = 0.5 * floor_color + 0.5 * np.array(tint)
        # Occupied cells: walls, with tables in their own color.
        image[occupied] = wall_color
        clutter_label = self._map_data.semantic_labels.get("clutter")
        if clutter_label is not None:
            table_color = np.array(palette.get("table", (0.36, 0.21, 0.06)))
            image[occupied & (semantic == clutter_label)] = table_color
        return (image * 255).astype(np.uint8)

    def get_top_down_view(self) -> TopDownView:
        """Semantic top-down map, positioned on the occupancy grid."""
        # get_top_down_image is indexed [x, y] like the grid itself, but the
        # trajectory plot draws grid.T -- row = y, col = x. Transposing only
        # here leaves get_top_down_image's own shape contract untouched.
        image = np.transpose(self.get_top_down_image(), (1, 0, 2))
        n_y, n_x = image.shape[:2]
        # The image is exactly one pixel per cell, so these are the same
        # bounds imshow would infer for the grid itself.
        return TopDownView(
            image=image, min_x=-0.5, max_x=n_x - 0.5, min_y=-0.5, max_y=n_y - 0.5,
        )

    def release(self) -> None:
        """Release the simulator's GPU resources (safe if never created)."""
        if self._simulator is not None:
            self._simulator.release()
            self._simulator = None
