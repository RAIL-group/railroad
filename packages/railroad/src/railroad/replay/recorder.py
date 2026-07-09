"""Build a :class:`RolloutLog` from a live deployment environment.

One entry point, :func:`build_rollout_log`, snapshots any deployment env into a
:class:`~railroad.replay.types.RolloutLog`, dispatching on ``problem_class``:
the unknown-map flavors are duck-typed against an unknown-space LSP env (needs
``observed_grid`` / ``frontiers``), the known-map flavor against a symbolic env
(needs ``occupancy_grid``). In production the live env is the railsim-backed
``LSPVisualEnvironment`` or a ProcTHOR env; in tests it is a GL-free replay env
or a synthetic stand-in — the recorder touches no GL/torch path.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Collection, Dict, Iterable, List, Mapping, Tuple

import numpy as np

from railroad.lsp.oracle import frontier_cells_hash

from .types import Pose3, RolloutLog, SubgoalRecord

START_NAME = "start_loc"
# Known-map search defaults (the recorder stores these into log.config so the
# replay env reconstructs the same travel speed and search time).
DEFAULT_SEARCH_TIME = 10.0
DEFAULT_SPEED = 1.0


def _pose_tuple(pose: Any) -> Pose3:
    """Normalize a pose (``Pose``-like or an ``(x, y[, yaw])`` tuple) to ``Pose3``."""
    if isinstance(pose, (tuple, list)):
        return (float(pose[0]), float(pose[1]), float(pose[2]) if len(pose) > 2 else 0.0)
    return (float(pose.x), float(pose.y), float(getattr(pose, "yaw", 0.0)))


def _searched_sites(env: Any) -> set:
    """Names carrying a truthy ``searched ?site`` fluent in *env*'s state.

    The search operator sets it deterministically on both found and not-found
    outcomes, so it marks exactly the sites the deployment actually inspected.
    """
    state = getattr(env, "state", None)
    fluents = getattr(state, "fluents", ()) if state is not None else ()
    return {
        f.args[0]
        for f in fluents
        if f.name == "searched" and not f.negated and f.args
    }


def _site_subgoals(
    sites: Iterable[str],
    registry: Any,
    contents_map: Mapping[str, Any],
    searched: Collection[str],
) -> List[SubgoalRecord]:
    """Revealed *sites* as subgoals: signature=name, centroid from *registry*, and
    — **only for the sites in *searched*** — their contents.

    A site (container / searchable location) is *revealed* the moment its map
    cell is observed, but its **contents are unknown until it is searched**.
    Offline replay's soundness rests on it seeing *only what the deployment
    observed*, so a revealed-but-uninspected site's true contents must never
    enter the log (they are ground truth the deployment never learned).
    Coordinates are fine — the cell was observed, so the candidate may navigate
    there; searching it then resolves not-found.
    """
    out: List[SubgoalRecord] = []
    for name in sites:
        coord = registry.get(name) if registry is not None else None
        row, col = (int(coord[0]), int(coord[1])) if coord is not None else (0, 0)
        was_searched = name in searched
        contents = (
            tuple(sorted(contents_map.get(name, set()))) if was_searched else ()
        )
        out.append(
            SubgoalRecord(
                signature=name,
                centroid=(row, col),
                cells=np.array([[row], [col]], dtype=int),
                contents=contents,
                searched=was_searched,
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
    goal: Any = None,
) -> RolloutLog:
    """Snapshot a deployment *env* into a :class:`RolloutLog` — the one recorder.

    Dispatches on *problem_class*:

    - ``"known-map-search"``: the map is fully known, so record ``env.occupancy_grid``,
      every searchable location's coordinates (contents only where searched), and
      the travel speed + search time into ``config``.
    - ``"navigation"`` / ``"object-search"``: record the final *observed* map, the
      deployment's ``NavigationConfig``, and any onboard panoramas; subgoals are
      the revealed containers (search) or the final frontiers (navigation).

    *robot_starts* maps each robot to its initial pose (a ``Pose`` or any object
    with ``.x`` / ``.y`` / ``.yaw``). *goal* is the deployment's planning goal (a
    ``Goal`` or bare ``Fluent``, possibly compound); it must be set for a search
    log so replay can plan toward the same goal.

    The deployed policy's realized cost (``actual_total_cost``) is always the
    deployment **makespan** — the final simulation time ``env.state.time``
    (seconds). It is taken from the env directly; there is no override, so the
    recorded cost is exactly the deployment cost that replay bounds (also in
    seconds) are compared against.

    Only deployment-observed information ever enters the log; in particular
    ``_site_subgoals`` withholds the contents of any revealed-but-unsearched
    container (ground truth the deployment never learned).
    """
    if problem_class == "known-map-search":
        recorded_grid = np.asarray(env.occupancy_grid, dtype=float).copy()
        searchable = sorted(env.objects_by_type.get("location", set()) - {START_NAME})
        subgoals = _site_subgoals(
            searchable,
            env.location_registry,
            getattr(env, "_objects_at_locations", {}),
            _searched_sites(env),
        )
        speed = float(getattr(env, "_pathing_speed_cells_per_sec", DEFAULT_SPEED))
        config: Dict[str, Any] = {
            "speed_cells_per_sec": speed,
            "search_time": DEFAULT_SEARCH_TIME,
        }
        pano_records: list = []
    else:
        recorded_grid = np.asarray(env.observed_grid, dtype=float).copy()
        # For object search, record the revealed containers (coords + searched
        # contents) so the arena is self-contained; for navigation, record the
        # final frontiers. Replay re-derives frontiers by re-sensing, so a search
        # log need not store them.
        containers = sorted(env.objects_by_type.get("container", set()))
        if containers:
            subgoals = _site_subgoals(
                containers,
                getattr(env, "_location_registry", None),
                getattr(env, "_objects_at_locations", {}),
                _searched_sites(env),
            )
        else:
            subgoals = [
                SubgoalRecord(
                    signature=frontier_cells_hash(frontier),
                    centroid=(int(frontier.centroid_row), int(frontier.centroid_col)),
                    cells=np.asarray(frontier.cells, dtype=int).copy(),
                )
                for frontier in env.frontiers.values()
            ]
        # Record the deployment's NavigationConfig so replay senses/maps exactly
        # as the deployment did (e.g. the same sensor_range) instead of a default.
        env_config = getattr(env, "config", None)
        config = dataclasses.asdict(env_config) if env_config is not None else {}
        # Accumulated panoramas (empty for non-visual deployments); served to a
        # learned estimator during replay.
        pano_records = list(getattr(env, "pano_records", []))

    starts: Dict[str, Pose3] = {
        robot: _pose_tuple(pose) for robot, pose in robot_starts.items()
    }

    # Normalize a bare Fluent goal to a Goal so the log always holds a Goal.
    if goal is not None:
        from railroad._bindings import Fluent, LiteralGoal

        if isinstance(goal, Fluent):
            goal = LiteralGoal(goal)

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
        goal=goal,
        subgoals=subgoals,
        actual_total_cost=float(actual_total_cost),
        config=config,
        pano_records=pano_records,
    )
