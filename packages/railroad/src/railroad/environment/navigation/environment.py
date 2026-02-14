"""Unknown-space environment with frontier-based exploration."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from railroad._bindings import Action, Fluent, GroundedEffect, State
from railroad.core import Operator

from ..skill import ActiveSkill
from ..symbolic import LocationRegistry, SymbolicEnvironment
from . import laser, mapping, pathing
from .constants import COLLISION_VAL, FREE_VAL, UNOBSERVED_VAL
from .frontiers import extract_frontiers, filter_reachable_frontiers
from .skill import NavigationMoveSkill
from .types import Frontier, NavigationConfig, Pose


class UnknownSpaceEnvironment(SymbolicEnvironment):
    """Environment for multi-robot exploration of an unknown occupancy grid.

    Extends :class:`SymbolicEnvironment` with:

    - Observed/true occupancy grids and laser-based sensing.
    - Frontier extraction and dynamic symbolic synchronisation.
    - Hidden-site unlocking when map cells become observed.
    - Interrupt-on-new-information policy for interruptible moves.
    """

    def __init__(
        self,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        operators: List[Operator],
        true_grid: np.ndarray,
        robot_initial_poses: Dict[str, Pose],
        location_registry: LocationRegistry,
        hidden_sites: Dict[str, Tuple[int, int]] | None = None,
        true_object_locations: Dict[str, Set[str]] | None = None,
        config: NavigationConfig | None = None,
    ) -> None:
        self._config = config or NavigationConfig()
        self._true_grid = true_grid
        self._observed_grid = UNOBSERVED_VAL * np.ones_like(true_grid)
        self._robot_poses: Dict[str, Pose] = dict(robot_initial_poses)
        self._hidden_sites: Dict[str, Tuple[int, int]] = dict(hidden_sites or {})
        self._frontiers: Dict[str, Frontier] = {}
        self._frames: list[np.ndarray] = []
        self._grid_generation: int = 0  # bumped when observed_grid changes
        self._cost_grid_cache: Dict[str, Tuple[int, np.ndarray]] = {}  # robot -> (gen, cost_grid)

        # Interrupt tracking
        self._interrupt_requested = False
        self._new_cells_since_interrupt = 0
        self._last_interrupt_time = state.time

        # Remember base locations (everything except dynamic frontiers/robot locs)
        self._base_locations: Set[str] = set(objects_by_type.get("location", set()))

        # Laser scanner directions (computed once)
        self._laser_directions = laser.get_laser_scanner_directions(
            self._config.sensor_num_rays, self._config.sensor_fov_rad
        )

        # Initialize parent (which calls _create_initial_effects_skill)
        super().__init__(
            state=state,
            objects_by_type=objects_by_type,
            operators=operators,
            true_object_locations=true_object_locations,
            location_registry=location_registry,
        )

        # Initial observation from all robot poses (no interrupt during init)
        self.observe_all_robots(time=state.time, allow_interrupt=False)
        self.refresh_frontiers()
        self.sync_dynamic_navigable_targets()

    @property
    def config(self) -> NavigationConfig:
        return self._config

    @property
    def robot_poses(self) -> Dict[str, Pose]:
        return self._robot_poses

    @property
    def true_grid(self) -> np.ndarray:
        return self._true_grid

    @property
    def observed_grid(self) -> np.ndarray:
        return self._observed_grid

    @property
    def frontiers(self) -> Dict[str, Frontier]:
        return self._frontiers

    @property
    def frames(self) -> list[np.ndarray]:
        return self._frames

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe_from_pose(
        self,
        robot: str,
        pose: Pose,
        time: float,
        allow_interrupt: bool = True,
    ) -> int:
        """Simulate laser scan from *pose* and fuse into observed grid.

        Returns the number of newly observed cells.
        """
        laser_ranges = laser.simulate_sensor_measurement(
            self._true_grid,
            self._laser_directions,
            self._config.sensor_range,
            pose,
        )

        self._observed_grid, newly_observed = mapping.insert_scan(
            occupancy_grid=self._observed_grid,
            laser_scanner_directions=self._laser_directions,
            laser_ranges=laser_ranges,
            max_range=self._config.sensor_range,
            sensor_pose=pose,
            connect_neighbor_distance=self._config.connect_neighbor_distance,
            occupied_prob=self._config.occupied_prob,
            unoccupied_prob=self._config.unoccupied_prob,
        )

        # Optional: correct observed cells with true-grid values
        if self._config.correct_with_known_map:
            observed_mask = self._observed_grid != UNOBSERVED_VAL
            self._observed_grid[observed_mask] = self._true_grid[observed_mask]

        self._new_cells_since_interrupt += newly_observed
        if newly_observed > 0:
            self._grid_generation += 1

        # Check interrupt thresholds
        if (
            allow_interrupt
            and not self._interrupt_requested
            and not self._any_robot_free()
            and self._new_cells_since_interrupt >= self._config.interrupt_min_new_cells
            and (time - self._last_interrupt_time) >= self._config.interrupt_min_dt
        ):
            self._interrupt_requested = True

        # Record frame snapshot
        if self._config.record_frames and newly_observed > 0:
            self._frames.append(self._observed_grid.copy())

        return newly_observed

    def observe_all_robots(
        self,
        time: float | None = None,
        allow_interrupt: bool = True,
    ) -> int:
        """Observe from every robot's current pose. Returns total new cells."""
        t = time if time is not None else self._time
        total = 0
        for robot, pose in self._robot_poses.items():
            total += self.observe_from_pose(robot, pose, t, allow_interrupt=allow_interrupt)
        self.refresh_frontiers()
        return total

    # ------------------------------------------------------------------
    # Frontier management
    # ------------------------------------------------------------------

    def refresh_frontiers(self) -> None:
        """Re-extract frontiers and update symbolic object sets."""
        raw_frontiers = extract_frontiers(self._observed_grid)

        robot_positions = [
            (int(round(p.x)), int(round(p.y)))
            for p in self._robot_poses.values()
        ]
        reachable = filter_reachable_frontiers(
            raw_frontiers, self._observed_grid, robot_positions
        )

        self._frontiers = {f.id: f for f in reachable}
        frontier_ids = set(self._frontiers.keys())

        # Update objects_by_type
        self._objects_by_type["frontier"] = set(frontier_ids)

        # Stabilize robot locations: if a robot is at a stale frontier that
        # no longer exists, remap it to a stable {robot}_loc at its current pose.
        stable_locs = self._base_locations | frontier_ids
        robot_locs = set()
        for rob in self._objects_by_type.get("robot", set()):
            for fluent in list(self._fluents):
                if (fluent.name == "at" and not fluent.negated
                        and len(fluent.args) >= 2 and fluent.args[0] == rob):
                    loc = fluent.args[1]
                    if loc not in stable_locs:
                        # Robot is at a stale location — remap to {robot}_loc
                        new_loc = f"{rob}_loc"
                        pose = self._robot_poses.get(rob)
                        if pose is not None and self._location_registry is not None:
                            self._location_registry.register(
                                new_loc, np.array([pose.x, pose.y])
                            )
                        self._fluents.discard(fluent)
                        self._fluents.add(Fluent("at", rob, new_loc))
                        robot_locs.add(new_loc)
                    else:
                        robot_locs.add(loc)
                    break

        self._objects_by_type["location"] = (
            self._base_locations | frontier_ids | robot_locs
        )

        # Register frontier centroids in location registry
        registry = self._location_registry
        if registry is not None:
            for f in reachable:
                registry.register(f.id, np.array([f.centroid_row, f.centroid_col]))

        # exploration-complete fluent
        ec_fluent = Fluent("exploration-complete")
        if not frontier_ids:
            self._fluents.add(ec_fluent)
        else:
            self._fluents.discard(ec_fluent)

    # ------------------------------------------------------------------
    # Dynamic navigable / hidden-site sync
    # ------------------------------------------------------------------

    def sync_dynamic_navigable_targets(self) -> None:
        """Synchronise (navigable *) fluents with current frontiers and sites."""
        desired_navigable: set[str] = set()

        # All current frontiers are navigable
        desired_navigable.update(self._objects_by_type.get("frontier", set()))

        # Previously unlocked hidden sites persist as navigable
        for fluent in list(self._fluents):
            if fluent.name == "navigable" and not fluent.negated and len(fluent.args) >= 1:
                site = fluent.args[0]
                if site in self._hidden_sites:
                    desired_navigable.add(site)

        # Unlock hidden sites whose cells are now observed
        for site, (r, c) in self._hidden_sites.items():
            if self.is_cell_observed(r, c):
                desired_navigable.add(site)

        # Replace all (navigable *) fluents
        stale_nav = [
            f for f in self._fluents
            if f.name == "navigable" and not f.negated
        ]
        for f in stale_nav:
            self._fluents.discard(f)
        for loc in desired_navigable:
            self._fluents.add(Fluent("navigable", loc))

        # Prune stale claims
        stale_claims = [
            f for f in self._fluents
            if f.name == "claimed" and not f.negated
            and len(f.args) >= 1 and f.args[0] not in desired_navigable
        ]
        for f in stale_claims:
            self._fluents.discard(f)

    # ------------------------------------------------------------------
    # Path / move-time helpers
    # ------------------------------------------------------------------

    def compute_move_path(self, loc_from: str, loc_to: str) -> np.ndarray:
        """Compute 2xN grid path between two symbolic locations."""
        registry = self._location_registry
        if registry is None:
            return np.array([[0], [0]])
        start_coords = registry.get(loc_from)
        end_coords = registry.get(loc_to)
        if start_coords is None or end_coords is None:
            return np.array([[0], [0]])

        start = (int(round(float(start_coords[0]))), int(round(float(start_coords[1]))))
        end = (int(round(float(end_coords[0]))), int(round(float(end_coords[1]))))
        _cost, path = pathing.get_cost_and_path(self._observed_grid, start, end)
        if path.size == 0:
            # Fallback: straight line
            return np.array([[start[0], end[0]], [start[1], end[1]]])
        return path

    def _get_cost_grid(self, loc: str) -> np.ndarray | None:
        """Return a cached Dijkstra cost grid from *loc*, recomputing if stale."""
        registry = self._location_registry
        if registry is None:
            return None
        coords = registry.get(loc)
        if coords is None:
            return None

        cached = self._cost_grid_cache.get(loc)
        if cached is not None and cached[0] == self._grid_generation:
            return cached[1]

        start = (int(round(float(coords[0]))), int(round(float(coords[1]))))
        cost_grid: np.ndarray = pathing.compute_cost_grid_from_position(  # type: ignore[assignment]
            self._observed_grid, start=[start[0], start[1]], only_return_cost_grid=True,
        )
        self._cost_grid_cache[loc] = (self._grid_generation, cost_grid)
        return cost_grid

    def estimate_move_time(self, robot: str, loc_from: str, loc_to: str) -> float:
        """Estimate move duration via cached Dijkstra cost grid lookup."""
        registry = self._location_registry
        if registry is None:
            return 0.01
        end_coords = registry.get(loc_to)
        if end_coords is None:
            return 0.01

        cost_grid = self._get_cost_grid(loc_from)
        if cost_grid is None:
            return 0.01

        r, c = int(round(float(end_coords[0]))), int(round(float(end_coords[1])))
        r = max(0, min(r, cost_grid.shape[0] - 1))
        c = max(0, min(c, cost_grid.shape[1] - 1))
        cost = float(cost_grid[r, c])
        if np.isinf(cost) or cost <= 0:
            return 0.01
        return cost / self._config.speed_cells_per_sec

    def is_cell_observed(self, row: int, col: int) -> bool:
        """Check whether a grid cell has been observed."""
        if not (0 <= row < self._observed_grid.shape[0]
                and 0 <= col < self._observed_grid.shape[1]):
            return False
        return float(self._observed_grid[row, col]) != UNOBSERVED_VAL

    # ------------------------------------------------------------------
    # Interrupt hooks (override base Environment)
    # ------------------------------------------------------------------

    def should_interrupt_active_skills(self) -> bool:
        return self._interrupt_requested

    def clear_interrupt_request(self) -> None:
        self._interrupt_requested = False
        self._new_cells_since_interrupt = 0
        self._last_interrupt_time = self._time

    # ------------------------------------------------------------------
    # Skill creation (override SymbolicEnvironment)
    # ------------------------------------------------------------------

    def create_skill(self, action: Action, time: float) -> ActiveSkill:
        """Create skill; use NavigationMoveSkill for move actions."""
        parts = action.name.split()
        if parts[0] == "move":
            return NavigationMoveSkill(action=action, start_time=time, env=self)
        return super().create_skill(action, time)

    # ------------------------------------------------------------------
    # Override act to sync navigable targets after each action
    # ------------------------------------------------------------------

    def act(self, action, loop_callback_fn=None):
        """Execute action, then sync frontiers and navigable targets."""
        result = super().act(action, loop_callback_fn=loop_callback_fn)
        self.refresh_frontiers()
        self.sync_dynamic_navigable_targets()
        return result
