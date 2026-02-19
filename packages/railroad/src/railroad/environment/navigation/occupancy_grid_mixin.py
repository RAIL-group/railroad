"""Reusable occupancy-grid pathing mixin for symbolic environments."""

from __future__ import annotations

from typing import cast

import numpy as np

from . import pathing


class OccupancyGridPathingMixin:
    """Shared symbolic-location path planning over occupancy grids."""

    @property
    def _pathing_unknown_as_obstacle(self) -> bool:
        return False

    @property
    def _pathing_use_soft_cost(self) -> bool:
        return True

    @property
    def _pathing_soft_cost_scale(self) -> float:
        return 12.0

    @property
    def _pathing_generation(self) -> int:
        return 0

    @property
    def _pathing_speed_cells_per_sec(self) -> float:
        return 1.0

    @property
    def occupancy_grid(self) -> np.ndarray:
        raise NotImplementedError

    def _lookup_location_xy(self, location: str) -> tuple[int, int] | None:
        registry = getattr(self, "location_registry", None)
        if registry is None:
            return None
        coords = registry.get(location)
        if coords is None:
            return None
        return (
            int(round(float(coords[0]))),
            int(round(float(coords[1]))),
        )

    def _get_cost_grid(self, loc: str) -> tuple[np.ndarray, tuple[int, int]] | None:
        """Return a cached Dijkstra cost grid from *loc*, recomputing if stale."""
        start = self._lookup_location_xy(loc)
        if start is None:
            return None

        cache = getattr(self, "_cost_grid_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_cost_grid_cache", cache)

        generation = self._pathing_generation
        use_soft_cost = self._pathing_use_soft_cost
        unknown_as_obstacle = self._pathing_unknown_as_obstacle
        soft_cost_scale = self._pathing_soft_cost_scale

        cached = cache.get(loc)
        if (
            cached is not None
            and cached[0] == generation
            and cached[1] == start[0]
            and cached[2] == start[1]
            and cached[3] == unknown_as_obstacle
            and cached[4] == use_soft_cost
            and cached[5] == soft_cost_scale
        ):
            return cached[6], start

        cost_grid = cast(
            np.ndarray,
            pathing.compute_cost_grid_from_position(
                self.occupancy_grid,
                start=[start[0], start[1]],
                use_soft_cost=use_soft_cost,
                unknown_as_obstacle=unknown_as_obstacle,
                soft_cost_scale=soft_cost_scale,
                only_return_cost_grid=True,
            ),
        )
        cache[loc] = (
            generation,
            start[0],
            start[1],
            unknown_as_obstacle,
            use_soft_cost,
            soft_cost_scale,
            cost_grid,
        )
        return cost_grid, start

    def estimate_move_time(self, robot: str, loc_from: str, loc_to: str) -> float:
        """Estimate move duration via cached Dijkstra cost grid lookup."""
        del robot
        end = self._lookup_location_xy(loc_to)
        if end is None:
            return float("inf")

        cost_grid_payload = self._get_cost_grid(loc_from)
        if cost_grid_payload is None:
            return float("inf")
        cost_grid, start = cost_grid_payload

        r = max(0, min(end[0], cost_grid.shape[0] - 1))
        c = max(0, min(end[1], cost_grid.shape[1] - 1))
        cost = float(cost_grid[r, c])
        if np.isinf(cost) or np.isnan(cost):
            return float("inf")
        if (r, c) == start:
            return 0.01

        speed = float(self._pathing_speed_cells_per_sec)
        if speed <= 1e-9:
            return float("inf")
        return max(0.01, cost / speed)

    def compute_move_path(
        self,
        loc_from: str,
        loc_to: str,
        *,
        use_theta: bool = True,
    ) -> np.ndarray:
        """Compute 2xN path between two symbolic locations."""
        start = self._lookup_location_xy(loc_from)
        end = self._lookup_location_xy(loc_to)
        if start is None or end is None:
            return np.array([[]], dtype=int)

        occupancy_grid = self.occupancy_grid
        if use_theta:
            _cost, path = pathing.get_cost_and_path_theta(
                occupancy_grid,
                start,
                end,
                use_soft_cost=self._pathing_use_soft_cost,
                unknown_as_obstacle=self._pathing_unknown_as_obstacle,
                soft_cost_scale=self._pathing_soft_cost_scale,
            )
            if path.size == 0 or path.shape[0] != 2:
                _cost, path = pathing.get_cost_and_path(
                    occupancy_grid,
                    start,
                    end,
                    use_soft_cost=self._pathing_use_soft_cost,
                    unknown_as_obstacle=self._pathing_unknown_as_obstacle,
                    soft_cost_scale=self._pathing_soft_cost_scale,
                )
        else:
            _cost, path = pathing.get_cost_and_path(
                occupancy_grid,
                start,
                end,
                use_soft_cost=self._pathing_use_soft_cost,
                unknown_as_obstacle=self._pathing_unknown_as_obstacle,
                soft_cost_scale=self._pathing_soft_cost_scale,
            )
        return path
