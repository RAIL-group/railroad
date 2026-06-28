"""Build a :class:`RolloutLog` from a live deployment environment.

Duck-typed against any unknown-space LSP environment: it needs
``observed_grid`` and ``frontiers`` (and, for the search extension, a
``robot_poses`` mapping). In production the live env is the railsim-backed
``LSPVisualEnvironment``; in tests it is a GL-free :class:`ReplayEnvironment`
or a synthetic stand-in — the recorder touches no GL/torch path.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from railroad.lsp.oracle import frontier_cells_hash

from .types import Pose3, RolloutLog, SubgoalRecord


def _pose_tuple(pose: Any) -> Pose3:
    return (float(pose.x), float(pose.y), float(getattr(pose, "yaw", 0.0)))


def _container_subgoals(env: Any, containers: list) -> list:
    """Revealed containers as subgoals: signature=name, centroid=coords,
    contents=true objects (so search replay can reconstruct hidden_sites +
    recorded_object_locations from the log alone)."""
    registry = getattr(env, "_location_registry", None)
    contents_map = getattr(env, "_objects_at_locations", {})
    out = []
    for name in containers:
        coord = registry.get(name) if registry is not None else None
        row, col = (int(coord[0]), int(coord[1])) if coord is not None else (0, 0)
        out.append(
            SubgoalRecord(
                signature=name,
                centroid=(row, col),
                cells=np.array([[row], [col]], dtype=int),
                contents=tuple(sorted(contents_map.get(name, set()))),
            )
        )
    return out


def build_rollout_log(
    env: Any,
    *,
    goal_cell: Tuple[int, int],
    robot_starts: Mapping[str, Any],
    env_name: str = "",
    seed: int | None = None,
    problem_class: str = "navigation",
) -> RolloutLog:
    """Snapshot *env*'s final observed map and frontiers into a log.

    *robot_starts* maps each robot to its initial pose (a ``Pose`` or any
    object with ``.x`` / ``.y`` / ``.yaw``).

    The deployed policy's realized cost (``actual_total_cost``) is always the
    deployment **makespan** — the final simulation time ``env.state.time``
    (seconds). It is taken from the env directly; there is no override, so the
    recorded cost is exactly the deployment cost that replay bounds (also in
    seconds) are compared against.
    """
    recorded_grid = np.asarray(env.observed_grid, dtype=float).copy()

    # For object search, record the revealed containers (coords + true contents)
    # so a search replay arena is self-contained; for navigation, record the
    # final frontiers. Replay re-derives frontiers by re-sensing, so a search log
    # need not store them.
    containers = sorted(env.objects_by_type.get("container", set()))
    if containers:
        subgoals = _container_subgoals(env, containers)
    else:
        subgoals = [
            SubgoalRecord(
                signature=frontier_cells_hash(frontier),
                centroid=(int(frontier.centroid_row), int(frontier.centroid_col)),
                cells=np.asarray(frontier.cells, dtype=int).copy(),
            )
            for frontier in env.frontiers.values()
        ]

    starts: Dict[str, Pose3] = {
        robot: _pose_tuple(pose) for robot, pose in robot_starts.items()
    }

    # Accumulated panoramas (empty for non-visual deployments); served to a
    # learned estimator during replay.
    pano_records = list(getattr(env, "pano_records", []))

    # Record the deployment's NavigationConfig so replay senses/maps exactly as
    # the deployment did (e.g. the same sensor_range) instead of a default.
    env_config = getattr(env, "config", None)
    config = dataclasses.asdict(env_config) if env_config is not None else {}

    # The deployed policy's realized cost = makespan (final sim time, seconds).
    state = getattr(env, "state", None)
    actual_total_cost = float(getattr(state, "time", 0.0))

    return RolloutLog(
        recorded_grid=recorded_grid,
        goal_cell=(int(goal_cell[0]), int(goal_cell[1])),
        robot_starts=starts,
        problem_class=problem_class,
        env_name=env_name,
        seed=seed,
        subgoals=subgoals,
        actual_total_cost=float(actual_total_cost),
        config=config,
        pano_records=pano_records,
    )
