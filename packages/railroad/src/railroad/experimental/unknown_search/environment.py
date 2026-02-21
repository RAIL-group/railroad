"""Unknown-space environment with frontier-based exploration."""

from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple, Type

import numpy as np
import scipy.ndimage

from railroad._bindings import Action, Fluent, State
from railroad.core import Operator
from railroad.environment.skill import (
    ActiveSkill,
    MotionSkill,
    NavigationMoveSkill,
)
from railroad.environment.symbolic import LocationRegistry, SymbolicEnvironment
from railroad.environment.types import Pose
from railroad.navigation.constants import OBSTACLE_THRESHOLD, UNOBSERVED_VAL
from railroad.navigation import OccupancyGridPathingMixin

from . import laser, mapping
from .frontiers import extract_frontiers, filter_reachable_frontiers
from .types import Frontier, NavigationConfig


class UnknownSpaceEnvironment(OccupancyGridPathingMixin, SymbolicEnvironment):
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
        skill_overrides: Dict[str, Type[ActiveSkill]] | None = None,
        hidden_sites: Dict[str, Tuple[int, int]] | None = None,
        true_object_locations: Dict[str, Set[str]] | None = None,
        config: NavigationConfig | None = None,
        validate_move_skill: bool = True,
    ) -> None:
        self._config = config or NavigationConfig()

        if validate_move_skill:
            has_nav_skill = any(
                issubclass(cls, NavigationMoveSkill)
                for cls in (skill_overrides or {}).values()
            )
            if not has_nav_skill:
                raise TypeError(
                    "UnknownSpaceEnvironment requires a NavigationMoveSkill (or "
                    "subclass such as InterruptibleNavigationMoveSkill) in "
                    "skill_overrides. Pass validate_move_skill=False to "
                    "suppress this check."
                )

        self._true_grid = true_grid
        self._observed_grid = UNOBSERVED_VAL * np.ones_like(true_grid)
        self._robot_poses: Dict[str, Pose] = dict(robot_initial_poses)
        self._hidden_sites: Dict[str, Tuple[int, int]] = dict(hidden_sites or {})
        self._frontiers: Dict[str, Frontier] = {}
        self._frames: list[np.ndarray] = []
        self._grid_generation: int = 0  # bumped when observed_grid changes
        self._cost_grid_cache: Dict[
            str,
            Tuple[int, int, int, bool, bool, float, np.ndarray],
        ] = {}

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
            skill_overrides=skill_overrides,
            location_registry=location_registry,
        )

        # Initial observation from all robot poses (no interrupt during init)
        self.observe_all_robots(time=state.time, allow_interrupt=False)
        self.refresh_frontiers()
        self.sync_dynamic_targets()

    @property
    def state(self) -> State:
        """Assemble current state from fluents and active skill effects."""
        return super().state

    @property
    def config(self) -> NavigationConfig:
        return self._config

    @property
    def robot_poses(self) -> Dict[str, Pose]:
        return self._robot_poses

    def set_robot_pose(self, robot: str, pose: object) -> None:
        """Store current robot pose for continuous-motion consumers."""
        if isinstance(pose, Pose):
            self._robot_poses[robot] = pose

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

    def register_discovered_location(
        self,
        name: str,
        coords: tuple[int, int] | np.ndarray | None = None,
    ) -> None:
        """Add a newly discovered map location to planner-visible locations."""
        self._base_locations.add(name)
        self._objects_by_type.setdefault("location", set()).add(name)
        if coords is not None and self._location_registry is not None:
            self._location_registry.register(name, np.asarray(coords, dtype=float))

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

        # Optional: correct observed cells with true-grid values, then
        # reveal obstacle cells adjacent to observed free space.
        if self._config.correct_with_known_map:
            # Correct all observed cells with true values
            observed_mask = self._observed_grid != UNOBSERVED_VAL
            self._observed_grid[observed_mask] = self._true_grid[observed_mask]
            # Inflate observed free cells with a 3x3 footprint, and reveal
            # obstacle cells touched by that inflated free-space region.
            known_free = observed_mask & (self._observed_grid < OBSTACLE_THRESHOLD)
            inflated_free = scipy.ndimage.binary_dilation(
                known_free,
                structure=np.ones((3, 3), dtype=bool),
            )
            inflated_obstacles = inflated_free & (self._true_grid >= OBSTACLE_THRESHOLD)
            new_obstacles = inflated_obstacles & (self._observed_grid == UNOBSERVED_VAL)
            self._observed_grid[new_obstacles] = self._true_grid[new_obstacles]
            newly_observed += int(np.count_nonzero(new_obstacles))

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
    # Dynamic target sync
    # ------------------------------------------------------------------

    def sync_dynamic_targets(self) -> None:
        """Prune stale target-tracking fluents after frontier/location updates."""
        valid_targets = set(self._objects_by_type.get("location", set()))

        # Remove stale claims whose target no longer exists.
        stale_claims = [
            f for f in self._fluents
            if f.name == "claimed" and not f.negated
            and len(f.args) >= 1 and f.args[0] not in valid_targets
        ]
        for f in stale_claims:
            self._fluents.discard(f)

        # Legacy cleanup: remove any leftover navigable fluents.
        stale_nav = [
            f for f in self._fluents
            if f.name == "navigable" and not f.negated
        ]
        for f in stale_nav:
            self._fluents.discard(f)

        # If a robot is currently anchored to its transient {robot}_loc,
        # clear just-moved so the next dispatch is not forced to no-op.
        for robot in self._objects_by_type.get("robot", set()):
            if Fluent("at", robot, f"{robot}_loc") in self._fluents:
                self._fluents.discard(Fluent("just-moved", robot))

    # ------------------------------------------------------------------
    # Path / move-time helpers
    # ------------------------------------------------------------------

    @property
    def occupancy_grid(self) -> np.ndarray:
        return self._observed_grid

    @property
    def _pathing_unknown_as_obstacle(self) -> bool:
        return True

    @property
    def _pathing_use_soft_cost(self) -> bool:
        return self._config.trajectory_use_soft_cost

    @property
    def _pathing_soft_cost_scale(self) -> float:
        return self._config.trajectory_soft_cost_scale

    @property
    def _pathing_generation(self) -> int:
        return self._grid_generation

    @property
    def _pathing_speed_cells_per_sec(self) -> float:
        return self._config.speed_cells_per_sec

    def is_cell_observed(self, row: int, col: int) -> bool:
        """Check whether a grid cell has been observed."""
        if not (0 <= row < self._observed_grid.shape[0]
                and 0 <= col < self._observed_grid.shape[1]):
            return False
        return float(self._observed_grid[row, col]) != UNOBSERVED_VAL

    # ------------------------------------------------------------------
    # Interrupt hooks (override base Environment)
    # ------------------------------------------------------------------

    def _clear_interrupt_request(self) -> None:
        self._interrupt_requested = False
        self._new_cells_since_interrupt = 0
        self._last_interrupt_time = self._time

    def _should_interrupt_skills(self) -> bool:
        """Interrupt only when new map info exists and an interruptible skill runs."""
        if not self._interrupt_requested:
            return False
        return any(
            skill.is_interruptible and not skill.is_done
            for skill in self._active_skills
        )

    def interrupt_skills(self) -> None:
        """Interrupt according to nav policy (new-info or robot-free)."""
        self.refresh_frontiers()
        self.sync_dynamic_targets()

        has_interruptible = any(
            skill.is_interruptible and not skill.is_done
            for skill in self._active_skills
        )

        if self._interrupt_requested:
            if not has_interruptible:
                # Avoid stale interrupt requests when all active moves are
                # non-interruptible.
                self._clear_interrupt_request()
                return
            interrupted = super().interrupt_skills()
            # Always clear request after an interrupt attempt.
            self._clear_interrupt_request()
            return interrupted

        super().interrupt_skills()

    def _iter_active_motion_skills(self) -> Iterable[MotionSkill]:
        """Yield active skills that expose continuous motion state."""
        for skill in self._active_skills:
            if isinstance(skill, MotionSkill):
                yield skill

    def _cap_next_advance_time(self, proposed_next_time: float) -> float:
        """Bound scheduler step so sensing occurs at regular cadence in motion."""
        sensor_dt = float(self._config.sensor_dt)
        if sensor_dt <= 1e-9:
            return proposed_next_time
        if any(ms.is_motion_active_at(self._time) for ms in self._iter_active_motion_skills()):
            return min(proposed_next_time, self._time + sensor_dt)
        return proposed_next_time

    def _after_skills_advanced(self, advanced_to_time: float) -> None:
        """Perform sensing for robots that are actively moving."""
        moving_robots = {
            motion_skill.controlled_robot
            for motion_skill in self._iter_active_motion_skills()
            if motion_skill.is_motion_active_at(advanced_to_time)
        }
        for robot in moving_robots:
            pose = self._robot_poses.get(robot)
            if pose is not None:
                self.observe_from_pose(robot, pose, advanced_to_time, allow_interrupt=True)

    # ------------------------------------------------------------------
    # Skill creation (override SymbolicEnvironment)
    # ------------------------------------------------------------------

    def _is_valid_action(self, action: Action) -> bool:
        """Filter base-invalid actions and oversized move durations."""
        if not super()._is_valid_action(action):
            return False

        parts = action.name.split()
        if parts and parts[0] == "move":
            if len(parts) > 3 and Fluent("at", parts[1], parts[2]) in self._fluents:
                # Fast reachability guard: use cached cost-grid lookup
                # instead of reconstructing a full path per action candidate.
                move_time = self.estimate_move_time(parts[1], parts[2], parts[3])
                if not np.isfinite(move_time):
                    return False
            delayed_times = [eff.time for eff in action.effects if eff.time > 1e-9]
            if delayed_times and min(delayed_times) > self._config.max_move_action_time:
                # Only filter from the robot's currently dispatchable source.
                # Keeping high-cost non-current actions preserves relaxed
                # reachability in heuristic computation.
                if len(parts) > 2 and Fluent("at", parts[1], parts[2]) in self._fluents:
                    return False

        return True

    # ------------------------------------------------------------------
    # Override act to sync dynamic targets after each action
    # ------------------------------------------------------------------

    def act(self, action, loop_callback_fn=None):
        """Execute action, then sync frontiers and dynamic targets."""
        result = super().act(action, loop_callback_fn=loop_callback_fn)
        self.refresh_frontiers()
        self.sync_dynamic_targets()
        return result
