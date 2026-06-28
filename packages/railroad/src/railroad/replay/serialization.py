"""On-disk format for :class:`RolloutLog` (npz arrays + ``meta.json``).

Mirrors the ``lsp/data.py`` convention: bulky arrays go to a compressed
``.npz``, everything else to a human-readable ``meta.json``. The
navigation log carries no recorded ``State`` objects (replay re-runs the
policy rather than replaying states), so no binding types need pickling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from railroad.environment.types import Pose

from .types import RolloutLog, StepRecord, SubgoalRecord

_GRID_FILE = "grid.npz"
_META_FILE = "meta.json"
_PANOS_FILE = "panos.npz"


@dataclass
class LoadedPanoRecord:
    """A deserialized panorama record (duck-types railsim ``PanoRecord``).

    Carries exactly what best-vantage perception needs (``image``,
    ``pose_cells``, ``time``, ``visibility_polygon``); reconstructed without
    importing the railsim (GL) extra.
    """

    robot: str
    time: float
    pose_cells: Pose
    pose_meters: Tuple[float, float, float]
    image: np.ndarray
    visibility_polygon: Optional[np.ndarray] = None


def _save_panos(out: Path, pano_records: List[Any]) -> List[str]:
    """Write panos to ``panos.npz``; return the per-record robot names (for meta)."""
    if not pano_records:
        return []
    images = np.stack([np.asarray(r.image) for r in pano_records])
    pose_cells = np.array(
        [[r.pose_cells.x, r.pose_cells.y, r.pose_cells.yaw] for r in pano_records],
        dtype=float,
    )
    pose_meters = np.array([list(r.pose_meters) for r in pano_records], dtype=float)
    times = np.array([r.time for r in pano_records], dtype=float)

    polys: List[np.ndarray] = []
    lengths: List[int] = []
    for r in pano_records:
        vp = r.visibility_polygon
        if vp is None:
            lengths.append(0)
        else:
            vp = np.asarray(vp, dtype=float).reshape(2, -1)
            polys.append(vp)
            lengths.append(vp.shape[1])
    vis_polys = np.concatenate(polys, axis=1) if polys else np.zeros((2, 0), dtype=float)

    np.savez_compressed(
        out / _PANOS_FILE,
        images=images,
        pose_cells=pose_cells,
        pose_meters=pose_meters,
        times=times,
        vis_polys=vis_polys,
        vis_lengths=np.array(lengths, dtype=int),
    )
    return [str(r.robot) for r in pano_records]


def _load_panos(src: Path, robots: List[str]) -> List[LoadedPanoRecord]:
    path = src / _PANOS_FILE
    if not path.exists():
        return []
    with np.load(path) as a:
        images = a["images"]
        pose_cells = a["pose_cells"]
        pose_meters = a["pose_meters"]
        times = a["times"]
        vis_polys = np.asarray(a["vis_polys"], dtype=float)
        vis_lengths = np.asarray(a["vis_lengths"], dtype=int)
    records: List[LoadedPanoRecord] = []
    offset = 0
    for i in range(len(times)):
        length = int(vis_lengths[i])
        poly = vis_polys[:, offset : offset + length].copy() if length > 0 else None
        offset += length
        records.append(
            LoadedPanoRecord(
                robot=robots[i] if i < len(robots) else "",
                time=float(times[i]),
                pose_cells=Pose(
                    float(pose_cells[i][0]),
                    float(pose_cells[i][1]),
                    float(pose_cells[i][2]),
                ),
                pose_meters=(
                    float(pose_meters[i][0]),
                    float(pose_meters[i][1]),
                    float(pose_meters[i][2]),
                ),
                image=images[i].copy(),
                visibility_polygon=poly,
            )
        )
    return records


def save_rollout_log(log: RolloutLog, directory: str | Path) -> Path:
    """Write *log* to *directory* (created if needed); return the directory."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    # Stack variable-length subgoal cells into one array + a lengths vector
    # (fixed npz keys; reconstructed by splitting on load).
    if log.subgoals:
        subgoal_cells = np.concatenate(
            [np.asarray(s.cells, dtype=int).reshape(2, -1) for s in log.subgoals],
            axis=1,
        )
        subgoal_lengths = np.array(
            [np.asarray(s.cells, dtype=int).reshape(2, -1).shape[1] for s in log.subgoals],
            dtype=int,
        )
    else:
        subgoal_cells = np.zeros((2, 0), dtype=int)
        subgoal_lengths = np.zeros((0,), dtype=int)
    np.savez_compressed(
        out / _GRID_FILE,
        recorded_grid=np.asarray(log.recorded_grid),
        subgoal_cells=subgoal_cells,
        subgoal_lengths=subgoal_lengths,
    )

    pano_robots = _save_panos(out, log.pano_records)

    meta = {
        "problem_class": log.problem_class,
        "env_name": log.env_name,
        "seed": log.seed,
        "pano_robots": pano_robots,
        "goal_cell": [int(log.goal_cell[0]), int(log.goal_cell[1])],
        "robot_starts": {
            robot: [float(v) for v in pose]
            for robot, pose in log.robot_starts.items()
        },
        "actual_total_cost": float(log.actual_total_cost),
        "config": dict(log.config),
        "subgoals": [
            {
                "signature": s.signature,
                "centroid": [int(s.centroid[0]), int(s.centroid[1])],
                "contents": list(s.contents),
            }
            for s in log.subgoals
        ],
        "steps": [
            {
                "time": float(step.time),
                "robot_poses": {
                    robot: [float(v) for v in pose]
                    for robot, pose in step.robot_poses.items()
                },
                "chosen_action": step.chosen_action,
                "net_motion": {
                    robot: float(v) for robot, v in step.net_motion.items()
                },
            }
            for step in log.steps
        ],
    }
    (out / _META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True))
    return out


def load_rollout_log(directory: str | Path) -> RolloutLog:
    """Reconstruct a :class:`RolloutLog` written by :func:`save_rollout_log`."""
    src = Path(directory)
    meta = json.loads((src / _META_FILE).read_text())
    with np.load(src / _GRID_FILE) as arrays:
        recorded_grid = arrays["recorded_grid"]
        all_cells = np.asarray(arrays["subgoal_cells"], dtype=int)
        lengths = np.asarray(arrays["subgoal_lengths"], dtype=int)
        subgoals: List[SubgoalRecord] = []
        offset = 0
        for raw, length in zip(meta["subgoals"], lengths):
            cells = all_cells[:, offset : offset + int(length)]
            offset += int(length)
            subgoals.append(
                SubgoalRecord(
                    signature=raw["signature"],
                    centroid=(int(raw["centroid"][0]), int(raw["centroid"][1])),
                    cells=cells.copy(),
                    contents=tuple(raw.get("contents", ())),
                )
            )

    steps = [
        StepRecord(
            time=float(raw["time"]),
            robot_poses={
                robot: (float(p[0]), float(p[1]), float(p[2]))
                for robot, p in raw["robot_poses"].items()
            },
            chosen_action=raw["chosen_action"],
            net_motion={r: float(v) for r, v in raw["net_motion"].items()},
        )
        for raw in meta["steps"]
    ]

    pano_records = _load_panos(src, list(meta.get("pano_robots", [])))

    return RolloutLog(
        recorded_grid=recorded_grid,
        goal_cell=(int(meta["goal_cell"][0]), int(meta["goal_cell"][1])),
        robot_starts={
            robot: (float(p[0]), float(p[1]), float(p[2]))
            for robot, p in meta["robot_starts"].items()
        },
        problem_class=meta["problem_class"],
        env_name=meta["env_name"],
        seed=meta["seed"],
        subgoals=subgoals,
        steps=steps,
        actual_total_cost=float(meta["actual_total_cost"]),
        config=dict(meta.get("config", {})),
        pano_records=pano_records,
    )
