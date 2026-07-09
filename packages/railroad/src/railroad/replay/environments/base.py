"""Shared machinery for the replay environments.

Two mixins, both mixed in *before* the concrete environment base so their
overrides win:

* :class:`ReplayConfinementMixin` — confinement sensing + pristine correction +
  net-motion + served panos. Used by the two *unknown-map* replays (navigation
  and unknown-map search); the known-map replay needs none of it.
* :class:`ReplayArenaMixin` — the policy/goal/finalize contract that lets
  :func:`~railroad.replay.driver.run_replay` drive any replay env uniformly. A
  replay env is *policy-agnostic* until :meth:`~ReplayArenaMixin.apply_policy`
  swaps in a candidate; its operators read the current policy through
  ``self._policy`` (so a new policy takes effect without rebuilding the arena).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence

import numpy as np
import scipy.ndimage

from railroad._bindings import Fluent, Goal, GroundedEffect
from railroad.environment.environment import Environment
from railroad.environment.types import Pose
from railroad.experimental.unknown_search import (
    NavigationConfig,
    UnknownSpaceEnvironment,
    laser,
    mapping,
)
from railroad.lsp.pano import roll_pano_to_bearing
from railroad.navigation.constants import (
    COLLISION_VAL,
    OBSTACLE_THRESHOLD,
    UNOBSERVED_VAL,
)

from ..cost import Commit, accumulate_bounds
from ..loop import MctsConfig
from ..policy import CandidatePolicy
from ..types import ReplayResult


def navigation_config_from_log(log_config: Mapping[str, Any]) -> NavigationConfig:
    """Rebuild the deployment's :class:`NavigationConfig` from a recorded log.

    The recorded ``config`` (a ``dataclasses.asdict`` of the deployment's
    config) is the **single source of truth**: replay senses and maps exactly
    as the deployment did. There is deliberately no default fallback — a
    separately-maintained default could silently drift from the deployment
    (e.g. a mismatched ``sensor_range`` would re-sense a different observed map
    than was recorded), breaking replay fidelity. Keys no longer on
    ``NavigationConfig`` (schema drift across versions) are ignored.

    Raises :class:`ValueError` if the log carries no config (it must come from
    the deployment; record it via ``build_rollout_log``, which captures
    ``env.config``, or pass an explicit ``config=`` to ``build_replay_env``).
    """
    if not log_config:
        raise ValueError(
            "RolloutLog has no recorded config; replay must use the "
            "deployment's NavigationConfig. Record it with build_rollout_log "
            "(which captures env.config), or pass an explicit config= override."
        )
    valid = {f.name for f in fields(NavigationConfig)}
    return NavigationConfig(**{k: v for k, v in log_config.items() if k in valid})


def robot_from_free(
    effects: Sequence[GroundedEffect], robots: Optional[Collection[str]] = None
) -> str:
    """The robot named by a ``free ?robot`` effect among *effects* (provenance).

    When *robots* is given, only a ``free`` on one of them counts and the
    lowest-id robot is the fallback; otherwise the first ``free`` wins and the
    fallback is ``""``.
    """
    for effect in effects:
        for fluent in effect.resulting_fluents:
            if fluent.name == "free" and not fluent.negated and fluent.args:
                if robots is None or fluent.args[0] in robots:
                    return fluent.args[0]
    return next(iter(sorted(robots)), "") if robots else ""


def require_goal(log: Any) -> Goal | Fluent:
    """The log's recorded planning goal, or a clear error if unset.

    Recorders capture it (pass ``goal=`` to ``build_rollout_log``); a search
    replay cannot plan without the goal the deployment pursued.
    """
    if log.goal is None:
        raise ValueError(
            f"{log.problem_class!r} replay needs a goal; record it by passing "
            "goal= to build_rollout_log."
        )
    return log.goal


def objects_in_goal(goal: Goal | Fluent, exclude: Collection[str]) -> set:
    """Object names a search *goal* references (its literal args minus *exclude*).

    Search goals are over the objects being found (``found ?object``), so the
    args of the goal's literals — minus the robots and locations in *exclude* —
    are exactly the objects the search operators must ground over. Handles
    compound goals uniformly (``get_all_literals`` flattens the tree).
    """
    excluded = set(exclude)
    return {arg for lit in goal.get_all_literals() for arg in lit.args} - excluded


@dataclass
class ServedPano:
    """A recorded panorama served as the onboard observation at a replay pose.

    Duck-types ``PanoRecord`` (so it feeds ``compute_frontier_views`` /
    best-vantage perception and the dashboard's onboard pane). It is the recorded
    observation nearest the robot's current pose, re-stamped with the replay time.
    """

    robot: str
    time: float
    pose_cells: Pose
    pose_meters: tuple
    image: np.ndarray
    visibility_polygon: Optional[np.ndarray] = None


class ReplayConfinementMixin(UnknownSpaceEnvironment):
    """Confinement sensing, pristine correction, net-motion, served panos."""

    _pristine_grid: np.ndarray
    _confinement_grid: np.ndarray
    _net_motion: Dict[str, float]
    # The recorded deployment panoramas (the buffer observations are served from).
    _recorded_panos: List
    # Per-robot key of the last served onboard pano: (recorded-pano id, roll shift).
    _last_served: Dict[str, tuple]
    # Onboard observations served along the replay trajectory (grown as the robot
    # moves) — what perception and the dashboard see. Mirrors a live visual env.
    pano_records: List

    def _setup_replay_grids(
        self, recorded_grid: np.ndarray, pano_records: Sequence
    ) -> np.ndarray:
        """Set up pristine/confinement grids + cost/pano state; return confinement.

        Call **before** ``super().__init__`` so the confinement grid is the
        ``true_grid`` and the served-pano buffer exists before the first
        sensing/refresh.
        """
        self._pristine_grid = np.asarray(recorded_grid, dtype=float).copy()
        confinement = self._pristine_grid.copy()
        confinement[confinement == UNOBSERVED_VAL] = COLLISION_VAL
        self._confinement_grid = confinement
        # The recorded buffer is the source; pano_records starts empty and is
        # filled with the observation served from each pose the robot reaches.
        self._recorded_panos = list(pano_records)
        self._last_served = {}
        self.pano_records = []
        self._net_motion = {}
        return confinement

    def _serve_pano(self, robot: str, pose: Pose, time: float) -> None:
        """Append the recorded panorama nearest *pose* as the onboard observation.

        This is replay's analogue of a visual env rendering a panorama at the
        robot's pose: it retrieves (does not render) the closest recorded view.
        Panoramas are heading-centered (the center column looks along the capture
        yaw), so the recorded image is rolled to the robot's *current* heading and
        re-stamped with that yaw — otherwise the onboard pano would face wherever
        the deployment looked here, not where the replay robot now faces. Rolling
        the image and the yaw *together* leaves best-vantage perception unchanged
        (``make_training_view``'s roll-to-frontier cancels the yaw); only the
        capture position stays recorded (the image really was taken there).
        De-duplicated so consecutive senses near the same recorded vantage don't
        pile up.
        """
        if not self._recorded_panos:
            return
        nearest = min(
            self._recorded_panos,
            key=lambda r: math.hypot(
                r.pose_cells.x - pose.x, r.pose_cells.y - pose.y
            ),
        )
        recorded_pose = nearest.pose_cells
        relative_yaw = float(pose.yaw) - float(recorded_pose.yaw)
        # De-dup on the rolled image, not just the recorded vantage: a robot that
        # turns (e.g. ~180 deg backtracking out of a dead end) while the nearest
        # recorded pano is unchanged must still re-serve a re-rolled view. Keying
        # on the column shift skips only pixel-identical frames, matching the live
        # visual env's (time, x, y, yaw) capture key.
        width = nearest.image.shape[1]
        shift = int(round(width * relative_yaw / (2.0 * math.pi))) % width
        served_key = (id(nearest), shift)
        if self._last_served.get(robot) == served_key:
            return
        self._last_served[robot] = served_key
        image = roll_pano_to_bearing(nearest.image, relative_yaw)
        recorded_meters = tuple(nearest.pose_meters)
        self.pano_records.append(
            ServedPano(
                robot=robot,
                time=float(time),
                pose_cells=Pose(recorded_pose.x, recorded_pose.y, float(pose.yaw)),
                pose_meters=(
                    recorded_meters[0],
                    recorded_meters[1],
                    float(pose.yaw),
                ),
                image=image,
                visibility_polygon=nearest.visibility_polygon,
            )
        )

    @property
    def net_motion(self) -> Dict[str, float]:
        """Cumulative travel distance (cells) per robot."""
        return self._net_motion

    def set_robot_pose(self, robot: str, pose: object) -> None:
        previous = self._robot_poses.get(robot)
        if isinstance(pose, Pose) and previous is not None:
            self._net_motion[robot] = self._net_motion.get(robot, 0.0) + math.hypot(
                pose.x - previous.x, pose.y - previous.y
            )
        super().set_robot_pose(robot, pose)

    def observe_from_pose(
        self, robot: str, pose: Pose, time: float, allow_interrupt: bool = True
    ) -> int:
        # Laser ranges against the confinement grid: occlusion only, so the robot
        # can never see past known space.
        laser_ranges = laser.simulate_sensor_measurement(
            self._confinement_grid,
            self._laser_directions,
            self._config.sensor_range,
            pose,
        )
        self._on_laser_scan(robot, pose, time, laser_ranges)

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

        # Correct observed cells against the PRISTINE recorded map (never the
        # confinement grid): masked/behind-frontier cells stay FREE/UNOBSERVED,
        # so the lidar never paints them as obstacles.
        observed_mask = self._observed_grid != UNOBSERVED_VAL
        self._observed_grid[observed_mask] = self._pristine_grid[observed_mask]
        known_free = observed_mask & (self._observed_grid < OBSTACLE_THRESHOLD)
        inflated_free = scipy.ndimage.binary_dilation(
            known_free, structure=np.ones((3, 3), dtype=bool)
        )
        inflated_obstacles = inflated_free & (self._pristine_grid >= OBSTACLE_THRESHOLD)
        new_obstacles = inflated_obstacles & (self._observed_grid == UNOBSERVED_VAL)
        self._observed_grid[new_obstacles] = self._pristine_grid[new_obstacles]
        newly_observed += int(np.count_nonzero(new_obstacles))

        self._new_cells_since_interrupt += newly_observed
        if newly_observed > 0:
            self._grid_generation += 1

        if (
            allow_interrupt
            and not self._interrupt_requested
            and not self._any_robot_free()
            and self._new_cells_since_interrupt
            >= self._config.interrupt_min_new_cells
            and (time - self._last_interrupt_time)
            >= self._config.interrupt_min_dt
        ):
            self._interrupt_requested = True

        if self._config.record_frames and newly_observed > 0:
            self._frames.append(self._observed_grid.copy())

        # Serve the onboard observation for this pose from the recorded panos.
        self._serve_pano(robot, pose, time)

        return newly_observed


class ReplayArenaMixin(Environment):
    """The policy / goal / finalize contract shared by every replay env.

    Bundles everything the driver needs beyond the env itself: default planner
    knobs, how to apply a candidate policy, what the planning goal is, and how the
    terminal state reduces to a :class:`~railroad.replay.types.ReplayResult`.
    Mixed in first so these win.
    """

    # Planner defaults for this flavor (overridden per env). run_replay reads
    # them when the caller does not pass mcts= / max_planning_iterations=.
    default_mcts: MctsConfig
    default_max_planning_iterations: int
    # Fluent-name substrings the dashboard shows for this flavor.
    dashboard_fluent_keywords: tuple

    # The current candidate policy; operators/estimators read through it, so
    # apply_policy() swaps behavior without rebuilding the arena.
    _policy: CandidatePolicy
    _refresh_estimators: list

    # Provided by each concrete env (annotated here so finalize() type-checks).
    replay_commits: List[Commit]

    def _init_policy(self) -> None:
        """Install the neutral (policy-agnostic) policy. Call first in __init__."""
        self._policy = CandidatePolicy()
        self._refresh_estimators = []

    def apply_policy(self, policy: CandidatePolicy) -> None:
        """Swap in *policy* so subsequent planning uses its probabilities.

        The base handles the fields every flavor may carry (the find-probability
        callables its operators read through ``self._policy``, and the
        served-vantage ``refresh_estimators``); the navigation env extends this
        to also install the frontier-statistics estimator.
        """
        self._policy = policy
        self._refresh_estimators = list(policy.refresh_estimators)

    @property
    def goal(self) -> Goal | Fluent:
        """The planning goal for this replay (implemented per flavor)."""
        raise NotImplementedError

    @property
    def search_log(self) -> List:
        """Per-search provenance; empty for navigation (search envs override)."""
        return []

    def finalize(self, termination: str) -> ReplayResult:
        """Reduce the terminal state + commits to a :class:`ReplayResult`.

        Uniform across flavors: the two bounds come from the recorded commits and
        the replay makespan, and ``goal_reached`` is read straight off the state.
        """
        total_cost = float(self.state.time)
        return ReplayResult(
            bounds=accumulate_bounds(self.replay_commits, total_cost),
            commits=list(self.replay_commits),
            termination=termination,
            total_cost=total_cost,
            sim_time=total_cost,
            goal_reached=self.goal.evaluate(self.state.fluents),
            search_log=list(self.search_log),
        )
