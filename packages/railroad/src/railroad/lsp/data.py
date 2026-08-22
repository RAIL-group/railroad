"""Training-data persistence: change detection, npz writing, and loading.

Data is emitted *on change only*: each frontier's label and chosen
vantage are hashed into a signature, and a datum is written only when
the signature differs from the last one emitted for that frontier id.
On disk, each datum is one compressed ``.npz`` plus a line in an
append-only ``index.jsonl``; run-level metadata lives in ``meta.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import weakref
from pathlib import Path
from typing import Any, Dict, Iterable, List, TextIO

import numpy as np

from .types import OracleFrontierLabel, TrainingDatum

VantageKey = tuple[str, float, float, float, float]


def vantage_key(record: Any) -> VantageKey:
    """Identity of a vantage: (robot, time, x, y) of the capture."""
    pose = record.pose_cells
    return (
        str(record.robot),
        float(record.time),
        float(pose.x),
        float(pose.y),
        float(pose.yaw),
    )


def _round_or_none(value: float | None, decimals: int) -> float | None:
    return None if value is None else round(float(value), decimals)


def frontier_signature(
    label: OracleFrontierLabel,
    vantage: VantageKey | None,
    *,
    cost_round_decimals: int = 1,
) -> str:
    """Hash of everything that makes a frontier's datum distinct."""
    payload = json.dumps(
        [
            label.cells_hash,
            label.prob_feasible,
            _round_or_none(label.success_cost, cost_round_decimals),
            _round_or_none(label.optimistic_cost, cost_round_decimals),
            _round_or_none(label.exploration_cost, cost_round_decimals),
            list(vantage) if vantage is not None else None,
        ]
    )
    return hashlib.sha1(payload.encode()).hexdigest()


class FrontierChangeTracker:
    """Tracks the last-emitted signature per frontier id."""

    def __init__(self) -> None:
        self._signatures: Dict[str, str] = {}

    def should_emit(self, frontier_id: str, signature: str) -> bool:
        """True (and record the signature) if it differs from the last emit."""
        if self._signatures.get(frontier_id) == signature:
            return False
        self._signatures[frontier_id] = signature
        return True

    def prune(self, live_ids: Iterable[str]) -> None:
        """Drop tracking state for frontiers that no longer exist."""
        live = set(live_ids)
        for stale in set(self._signatures) - live:
            del self._signatures[stale]


def _cost_to_array(value: float | None) -> float:
    return math.nan if value is None else float(value)


def _cost_from_array(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


class TrainingDataWriter:
    """Writes one compressed npz per datum plus a JSONL index."""

    def __init__(
        self,
        out_dir: str | Path,
        run_metadata: Dict[str, Any] | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.num_written = 0
        with open(self.out_dir / "meta.json", "w") as f:
            json.dump(run_metadata or {}, f, indent=2)
        self._index_file: TextIO | None = None
        self._finalizer: weakref.finalize | None = None

    def _index(self) -> TextIO:
        """Open ``index.jsonl`` on first write.

        Opening in ``__init__`` leaked a descriptor whenever the caller was
        abandoned between construction and the ``try`` that closes it --
        ``run_point_goal_rollout`` builds the writer at rollout.py:233 but only
        enters its ``try``/``finally`` at :243, and ``bulk.py`` swallows the
        exception and moves to the next seed. A sweep over failing seeds leaked
        one descriptor per failure, up to ``EMFILE``.

        A writer that never writes now never opens the file, and one that does
        is closed by the finalizer even if nobody calls ``close()``.
        """
        if self._index_file is None:
            self._index_file = open(self.out_dir / "index.jsonl", "a")
            self._finalizer = weakref.finalize(self, self._index_file.close)
        return self._index_file

    def write(self, datum: TrainingDatum) -> Path:
        path = self.out_dir / f"datum_{self.num_written:06d}.npz"
        np.savez_compressed(
            path,
            image=datum.image,
            frontier_xy_ego=np.asarray(datum.frontier_xy_ego, dtype=float),
            goal_xy_ego=np.asarray(datum.goal_xy_ego, dtype=float),
            label=np.asarray(datum.label, dtype=bool),
            success_cost=_cost_to_array(datum.success_cost),
            optimistic_cost=_cost_to_array(datum.optimistic_cost),
            exploration_cost=_cost_to_array(datum.exploration_cost),
        )
        index_entry = {"file": path.name, "label": bool(datum.label)}
        index_entry.update({
            k: v for k, v in datum.metadata.items()
            if isinstance(v, (str, int, float, bool)) or v is None
        })
        index_file = self._index()
        index_file.write(json.dumps(index_entry) + "\n")
        index_file.flush()
        self.num_written += 1
        return path

    def close(self) -> None:
        # Idempotent, and a no-op for a writer that never opened the index:
        # a finalize object runs at most once and detaches itself.
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        self._index_file = None

    def __enter__(self) -> "TrainingDataWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def load_datum(path: str | Path) -> TrainingDatum:
    """Load a datum written by :class:`TrainingDataWriter`."""
    with np.load(path) as data:
        frontier_xy = data["frontier_xy_ego"]
        goal_xy = data["goal_xy_ego"]
        return TrainingDatum(
            image=data["image"],
            frontier_xy_ego=(float(frontier_xy[0]), float(frontier_xy[1])),
            goal_xy_ego=(float(goal_xy[0]), float(goal_xy[1])),
            label=bool(data["label"]),
            success_cost=_cost_from_array(float(data["success_cost"])),
            optimistic_cost=_cost_from_array(float(data["optimistic_cost"])),
            exploration_cost=_cost_from_array(float(data["exploration_cost"])),
        )


def read_index(out_dir: str | Path) -> List[Dict[str, Any]]:
    """Read the JSONL index of a training-data directory."""
    index_path = Path(out_dir) / "index.jsonl"
    if not index_path.exists():
        return []
    with open(index_path) as f:
        return [json.loads(line) for line in f if line.strip()]
