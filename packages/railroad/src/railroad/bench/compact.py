"""
Lazy compaction cache for benchmark experiments.

MLflow stores each run in many small files / SQLite rows, which makes loading
a finished experiment slow. To speed up dashboard browsing, we materialize a
single Parquet (runs) + JSON (metadata + summary) cache per experiment in
``.benchmark_cache/<exp_name>/``.

The cache is a *staleness-checked* projection of the source data. We use the
mtime of the SQLite backend file plus the experiment's artifact directory as
a fingerprint: if either has been modified since the cache was written, we
treat the cache as stale and recompute. We also refuse to write a cache while
any run is still ``RUNNING``/``SCHEDULED`` — that way an in-progress run
never produces a cache that gets stuck.

Killing a run mid-way is fine: MLflow flips its status to ``FAILED`` /
``KILLED``, the run is no longer ``RUNNING``, and the experiment becomes
eligible for compaction on the next load.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import dataclasses

import mlflow
import pandas as pd
import plotly.io as pio
from plotly.graph_objects import Figure


CACHE_DIR_NAME = ".benchmark_cache"


def _cache_dir() -> Path:
    return Path(CACHE_DIR_NAME)


def _cache_paths(exp_name: str) -> tuple[Path, Path, Path]:
    d = _cache_dir() / exp_name
    return d / "runs.parquet", d / "meta.json", d / "figures.json"


def _serialize_figures(figures: dict) -> dict:
    """Serialize plotly Figures to JSON-safe dicts in-place-equivalent.

    Expects the shape produced by ``create_violin_plots_by_benchmark`` and
    ``create_all_sweep_plots``: violins are ``[{"benchmark": str, "figure": Figure}]``,
    sweeps are ``{benchmark: [{"title": str, "figure": Figure, ...}]}``.
    """
    def _encode(value):
        if isinstance(value, Figure):
            return {"__plotly__": True, "json": pio.to_json(value)}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            # Avoid dataclasses.asdict because it recursively converts nested
            # dataclasses to plain dicts before our encoder sees them.
            return {
                "__dataclass__": type(value).__name__,
                "fields": {
                    f.name: _encode(getattr(value, f.name))
                    for f in dataclasses.fields(value)
                },
            }
        if isinstance(value, dict):
            return {k: _encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_encode(v) for v in value]
        return value

    def _encode_item(item):
        return {k: _encode(v) for k, v in item.items()}

    out = {}
    if "violin" in figures:
        out["violin"] = [_encode_item(item) for item in figures["violin"]]
    if "sweep" in figures:
        out["sweep"] = {
            bench: [_encode_item(item) for item in sweep_list]
            for bench, sweep_list in figures["sweep"].items()
        }
    return out


# Registry of dataclasses we know how to round-trip. Imports are deferred to
# avoid circular imports during module load.
def _dataclass_registry() -> dict:
    from railroad.bench.dashboard.sweeps import SweepAnalysis, SweepGroup
    return {
        "SweepAnalysis": SweepAnalysis,
        "SweepGroup": SweepGroup,
    }


def _deserialize_figures(payload: dict) -> dict:
    registry = _dataclass_registry()

    def _decode(value):
        if isinstance(value, dict):
            if value.get("__plotly__"):
                return pio.from_json(value["json"])
            if "__dataclass__" in value:
                cls = registry.get(value["__dataclass__"])
                if cls is None:
                    return {k: _decode(v) for k, v in value["fields"].items()}
                return cls(**{k: _decode(v) for k, v in value["fields"].items()})
            return {k: _decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_decode(v) for v in value]
        return value

    def _decode_item(item):
        return {k: _decode(v) for k, v in item.items()}

    out = {}
    if "violin" in payload:
        out["violin"] = [_decode_item(item) for item in payload["violin"]]
    if "sweep" in payload:
        out["sweep"] = {
            bench: [_decode_item(item) for item in sweep_list]
            for bench, sweep_list in payload["sweep"].items()
        }
    return out


def _artifact_root(exp_name: str) -> Optional[Path]:
    """Return the experiment's artifact directory (best-effort)."""
    try:
        exp = mlflow.get_experiment_by_name(exp_name)  # type: ignore[possibly-missing-attribute]
    except Exception:
        return None
    if exp is None or not exp.artifact_location:
        return None
    loc = exp.artifact_location
    # Strip common URI prefixes
    if loc.startswith("file://"):
        loc = loc[len("file://"):]
    if loc.startswith("mlflow-artifacts:"):
        # Cannot resolve to a filesystem path without server context
        return None
    p = Path(loc)
    return p if p.exists() else None


def _source_fingerprint(exp_name: str) -> dict:
    """A cheap stamp that changes whenever the experiment data changes."""
    fp: dict = {}
    db = Path("mlflow.db")
    if db.exists():
        fp["db_mtime"] = db.stat().st_mtime
    art = _artifact_root(exp_name)
    if art is not None:
        try:
            mtimes = [f.stat().st_mtime for f in art.rglob("*") if f.is_file()]
            fp["artifact_mtime"] = max(mtimes) if mtimes else 0.0
        except Exception:
            pass
    return fp


def _has_in_progress_runs(df: pd.DataFrame) -> bool:
    if df.empty or "status" not in df.columns:
        return False
    return df["status"].isin(["RUNNING", "SCHEDULED"]).any()


def load(exp_name: str) -> Optional[tuple[pd.DataFrame, dict, dict]]:
    """
    Return ``(df, metadata, summary)`` if a fresh cache exists, else ``None``.
    """
    runs_path, meta_path, _figures_path = _cache_paths(exp_name)
    if not runs_path.exists() or not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            cached = json.load(f)
        if cached.get("fingerprint") != _source_fingerprint(exp_name):
            return None
        df = pd.read_parquet(runs_path)
        return df, cached["metadata"], cached["summary"]
    except Exception:
        return None


def load_figures(exp_name: str) -> Optional[dict]:
    """
    Return cached figures (deserialized) if a fresh cache exists, else ``None``.

    Validates the figures cache against its own embedded fingerprint so that
    figures stay correctly invalidated even when the runs/meta cache cannot be
    refreshed (e.g., while runs are still in progress).
    """
    _runs_path, _meta_path, figures_path = _cache_paths(exp_name)
    if not figures_path.exists():
        return None
    try:
        with open(figures_path) as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return None
        if payload.get("fingerprint") != _source_fingerprint(exp_name):
            return None
        return _deserialize_figures(payload.get("figures", {}))
    except Exception:
        return None


def save(
    exp_name: str,
    df: pd.DataFrame,
    metadata: dict,
    summary: dict,
    figures: Optional[dict] = None,
) -> bool:
    """
    Persist the cache for ``exp_name``. Returns True if written.

    Skips writing while any run is still in progress. If ``figures`` is
    provided, it is serialized to ``figures.json``.
    """
    if _has_in_progress_runs(df):
        return False
    runs_path, meta_path, figures_path = _cache_paths(exp_name)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(runs_path, index=False)
    payload = {
        "metadata": metadata,
        "summary": summary,
        "fingerprint": _source_fingerprint(exp_name),
        "cached_at": datetime.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(payload, f, default=str)
    if figures is not None:
        _write_figures(figures_path, figures)
    return True


def _write_figures(figures_path: Path, figures: dict) -> None:
    figures_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # Embed an independent fingerprint so figures stay correctly
        # invalidated even when runs/meta can't be refreshed.
        "fingerprint": _source_fingerprint(figures_path.parent.name),
        "cached_at": datetime.now().isoformat(),
        "figures": _serialize_figures(figures),
    }
    with open(figures_path, "w") as f:
        json.dump(payload, f)


def save_figures(exp_name: str, figures: dict) -> bool:
    """
    Persist the figures cache for ``exp_name``.

    Independent of the runs/meta cache: writes against the *current* source
    fingerprint, so the saved figures will be invalidated as soon as the
    underlying experiment data changes.
    """
    _runs_path, _meta_path, figures_path = _cache_paths(exp_name)
    _write_figures(figures_path, figures)
    return True


def invalidate(exp_name: str) -> None:
    """Remove the cache directory for ``exp_name`` if present."""
    d = _cache_dir() / exp_name
    if d.exists():
        shutil.rmtree(d)


def invalidate_all() -> None:
    """Remove the entire cache directory."""
    d = _cache_dir()
    if d.exists():
        shutil.rmtree(d)
