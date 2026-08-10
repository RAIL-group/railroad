"""Collision-aware `move` paths for the MolmoSpaces executor.

Straight-line stand-to-stand interpolation (the executor's original
approach) cuts diagonally across the 2x2 table grid on some moves (e.g.
workstation1 -> box_station) and can clip straight through the *other*
table sitting in between. This module builds a small static occupancy grid
from the scene's table footprints and routes `move` through it with
Theta* (`railroad.navigation.pathing`, the same any-angle planner already
used for ProcTHOR navigation elsewhere in this repo) instead.

Kinematic only -- no dynamics here, just a smarter path for the executor's
direct qpos writes to follow. Real physics (weld-constraint grasping,
contact/gravity for held/placed objects) lives in molmospace_executor.py.
"""

from typing import Optional, Tuple

import numpy as np

from railroad.navigation.constants import COLLISION_VAL, FREE_VAL
from railroad.navigation.pathing import get_cost_and_path_theta

from molmospace_scene import (  # pyrefly: ignore [missing-import]
    STAND_OFFSET_Y,
    TABLE_TOP_HALF,
    WORKSTATION_LOCATIONS,
)

RESOLUTION = 0.05         # meters per grid cell
MARGIN = 1.5              # meters of clearance around the outermost tables/stands
TABLE_CLEARANCE = 0.4     # extra clearance beyond each table's top footprint
# MobileFrankaRobotConfig's base is a 0.25m-half-width box; blocking a
# 0.35m radius around the other robot's spot only accounts for its own
# half-width plus a thin margin, not the routing robot's *own* half-width
# too -- 0.25 (other robot) + 0.25 (self) + margin.
ROBOT_BLOCK_RADIUS = 0.6


class OccupancyGrid:
    """Static (table-footprint-only) occupancy grid covering the scene's
    2x2 table layout, plus a per-query overlay for the other robot's
    current position.
    """

    def __init__(self) -> None:
        xs = [p[0] for p in WORKSTATION_LOCATIONS.values()]
        ys = [p[1] for p in WORKSTATION_LOCATIONS.values()]
        stand_ys = [ty - STAND_OFFSET_Y for _tx, ty in WORKSTATION_LOCATIONS.values()]
        self.x_min = min(xs) - MARGIN
        self.x_max = max(xs) + MARGIN
        self.y_min = min(stand_ys) - MARGIN
        self.y_max = max(ys) + MARGIN
        self.n_cols = int((self.x_max - self.x_min) / RESOLUTION) + 1
        self.n_rows = int((self.y_max - self.y_min) / RESOLUTION) + 1

        self.static_grid = np.full((self.n_rows, self.n_cols), FREE_VAL)
        half_x = TABLE_TOP_HALF[0] + TABLE_CLEARANCE
        half_y = TABLE_TOP_HALF[1] + TABLE_CLEARANCE
        for tx, ty in WORKSTATION_LOCATIONS.values():
            self._mark_rect(self.static_grid, tx, ty, half_x, half_y)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        row = int(round((y - self.y_min) / RESOLUTION))
        col = int(round((x - self.x_min) / RESOLUTION))
        return (min(max(row, 0), self.n_rows - 1), min(max(col, 0), self.n_cols - 1))

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        return (self.x_min + col * RESOLUTION, self.y_min + row * RESOLUTION)

    def _mark_rect(self, grid: np.ndarray, cx: float, cy: float, half_x: float, half_y: float) -> None:
        r0, c0 = self.world_to_grid(cx - half_x, cy - half_y)
        r1, c1 = self.world_to_grid(cx + half_x, cy + half_y)
        grid[min(r0, r1):max(r0, r1) + 1, min(c0, c1):max(c0, c1) + 1] = COLLISION_VAL

    def path(
        self,
        start_xy: Tuple[float, float],
        end_xy: Tuple[float, float],
        blocked_xy: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """World-space Theta* path from `start_xy` to `end_xy` as a (2, N)
        array of (x, y) waypoints, optionally treating `blocked_xy` (the
        other robot's current position) as an extra temporary obstacle.

        If `blocked_xy`'s block radius seals off every route to the
        destination (e.g. the other robot is standing right at a shared
        table this robot is also headed to), retries against the static
        (tables-only) grid rather than jumping straight to the raw
        straight-line fallback below -- that fallback ignores every
        obstacle, not just the other robot, so without this retry a
        Theta* failure caused solely by the other robot's position could
        still render as this robot's path cutting straight through a
        table. Only truly falls back to a direct 2-point line if even the
        tables-only grid has no path.
        """
        start = self.world_to_grid(*start_xy)
        end = self.world_to_grid(*end_xy)

        if blocked_xy is not None:
            grid = self.static_grid.copy()
            self._mark_rect(grid, blocked_xy[0], blocked_xy[1], ROBOT_BLOCK_RADIUS, ROBOT_BLOCK_RADIUS)
            _cost, cells = get_cost_and_path_theta(grid, start, end)
            if cells.size == 0:
                _cost, cells = get_cost_and_path_theta(self.static_grid, start, end)
        else:
            _cost, cells = get_cost_and_path_theta(self.static_grid, start, end)

        if cells.size == 0:
            return np.array([[start_xy[0], end_xy[0]], [start_xy[1], end_xy[1]]])

        rows, cols = cells[0], cells[1]
        xs = self.x_min + cols * RESOLUTION
        ys = self.y_min + rows * RESOLUTION
        return np.vstack([xs, ys])
