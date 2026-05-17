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
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import dataclasses

import mlflow
import pandas as pd
import plotly.io as pio
from plotly.graph_objects import Figure


CACHE_DIR_NAME = ".benchmark_cache"

# Bump whenever the on-disk cache layout (parquet columns, meta/figures JSON
# shape, figure serialization) changes in a way that makes old caches
# unreadable. A mismatch purges and rebuilds the cache transparently.
# v2: summary.success_by_benchmark gained avg_plan_cost / avg_wall_time.
CACHE_FORMAT_VERSION = 2

# Per-experiment stamp file, written into the experiment's artifact directory.
# Its contents change whenever the experiment's data changes, so a cache keyed
# on it is invalidated only by *that* experiment — unlike the shared
# ``mlflow.db`` mtime, which every run (for any experiment) bumps.
STAMP_FILENAME = ".railroad_cache_stamp"


# Set RAILROAD_CACHE_DEBUG=1 to print per-experiment cache hit/miss decisions
# (and, on a miss, exactly which fingerprint component changed). This is the
# fast way to confirm the per-experiment stamp keying is actually keeping
# caches valid across unrelated runs.
_CACHE_DEBUG = os.environ.get("RAILROAD_CACHE_DEBUG", "").lower() not in (
    "",
    "0",
    "false",
    "no",
)


def _log(exp_name: str, msg: str) -> None:
    if _CACHE_DEBUG:
        print(f"[cache] {exp_name}: {msg}")


def _fingerprint_diff(cached: object, current: dict) -> str:
    """Human-readable explanation of why two fingerprints differ."""
    if not isinstance(cached, dict):
        return f"cached fingerprint is not a dict ({type(cached).__name__})"
    keys = sorted(set(cached) | set(current))
    parts = []
    for k in keys:
        old = cached.get(k, "<absent>")
        new = current.get(k, "<absent>")
        if old != new:
            parts.append(f"{k}: {old!r} -> {new!r}")
    return "; ".join(parts) if parts else "(fingerprints equal?)"


def _cache_dir() -> Path:
    return Path(CACHE_DIR_NAME)


def _stamp_path(exp_name: str) -> Optional[Path]:
    """Path to the experiment's cache stamp file, if its artifact dir resolves."""
    art = _artifact_root(exp_name)
    if art is None:
        return None
    return art / STAMP_FILENAME


def touch_stamp(exp_name: str) -> None:
    """Best-effort: record that ``exp_name``'s data just changed.

    Writes the current wall-clock time into the experiment's stamp file. Called
    from the tracker whenever runs/metadata are written. Failures are swallowed:
    a missing stamp simply falls back to the legacy mtime fingerprint.
    """
    try:
        path = _stamp_path(exp_name)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(repr(time.time()))
        _log(exp_name, f"touched stamp {path}")
    except Exception as e:
        _log(exp_name, f"failed to touch stamp: {e}")


def _ensure_stamp(exp_name: str) -> bool:
    """Create a stamp for ``exp_name`` only if it has none yet (best-effort).

    Experiments last run before the stamp existed never get one from the
    tracker (``touch_stamp`` only fires on a new write). They are then stuck on
    the legacy *shared* ``mlflow.db`` mtime fallback, so any run for any other
    experiment invalidates their cache. Backfilling a stamp here — at cache
    save time, when the experiment's data is stable — adopts them onto
    per-experiment keying so unrelated runs stop reloading them.

    Returns True if a stamp now exists (created or already present).
    """
    try:
        path = _stamp_path(exp_name)
        if path is None:
            return False
        if path.exists():
            return True
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(repr(time.time()))
        _log(exp_name, f"backfilled stamp {path} (was on legacy db_mtime keying)")
        return True
    except Exception as e:
        _log(exp_name, f"failed to backfill stamp: {e}")
        return False


def remove_stamp(exp_name: str) -> None:
    """Remove the stamp file for ``exp_name`` if present (best-effort)."""
    try:
        path = _stamp_path(exp_name)
        if path is not None and path.exists():
            path.unlink()
    except Exception:
        pass


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
    """A cheap stamp that changes whenever the experiment data changes.

    Preferred: the per-experiment stamp file, which only changes when *this*
    experiment is written. This keeps unrelated benchmark runs from
    invalidating every experiment's cache (the shared ``mlflow.db`` mtime did).

    Fallback (no stamp yet — e.g. an experiment last run before this code, or
    an artifact dir that doesn't resolve to the filesystem): the legacy
    ``mlflow.db`` + artifact-dir mtime fingerprint, so old caches keep working
    until that experiment's next run writes a stamp.
    """
    fp: dict = {"cache_format": CACHE_FORMAT_VERSION}

    stamp = _stamp_path(exp_name)
    if stamp is not None and stamp.exists():
        try:
            fp["stamp"] = stamp.read_text()
            return fp
        except Exception:
            pass

    db = Path("mlflow.db")
    if db.exists():
        fp["db_mtime"] = db.stat().st_mtime
    art = _artifact_root(exp_name)
    if art is not None:
        # Use the directory's own mtime rather than walking the tree: artifact
        # writes update the containing run dir's mtime, and that's enough to
        # detect new/changed runs without an O(n) scan on every load.
        try:
            run_dirs = [p for p in art.iterdir() if p.is_dir()]
            mtimes = [art.stat().st_mtime] + [d.stat().st_mtime for d in run_dirs]
            fp["artifact_mtime"] = max(mtimes) if mtimes else 0.0
        except Exception:
            pass
    return fp


_TERMINAL_STATUSES = {"FINISHED", "FAILED", "KILLED"}


def _has_in_progress_runs(df: pd.DataFrame) -> bool:
    if df.empty or "status" not in df.columns:
        return False
    return (~df["status"].isin(_TERMINAL_STATUSES)).any()


def load(exp_name: str) -> Optional[tuple[pd.DataFrame, dict, dict]]:
    """
    Return ``(df, metadata, summary)`` if a fresh cache exists, else ``None``.
    """
    runs_path, meta_path, _figures_path = _cache_paths(exp_name)
    if not runs_path.exists() or not meta_path.exists():
        _log(exp_name, "MISS (no cache files on disk yet)")
        return None
    try:
        with open(meta_path) as f:
            cached = json.load(f)
        current_fp = _source_fingerprint(exp_name)
        cached_fp = cached.get("fingerprint")
        if cached_fp != current_fp:
            keyed = "stamp" if "stamp" in current_fp else "legacy db/artifact mtime"
            _log(
                exp_name,
                f"MISS (fingerprint changed, keyed on {keyed}) "
                f"[{_fingerprint_diff(cached_fp, current_fp)}]",
            )
            return None
        df = pd.read_parquet(runs_path)
        _log(
            exp_name,
            f"HIT ({len(df)} runs, cached_at={cached.get('cached_at', '?')})",
        )
        return df, cached["metadata"], cached["summary"]
    except Exception as e:
        # Corrupt / unreadable cache (e.g. truncated parquet, schema change):
        # purge it so this same load rebuilds it cleanly instead of failing
        # on every future load.
        _log(exp_name, f"MISS (corrupt cache, purging and rebuilding): {e}")
        invalidate(exp_name)
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
        _log(exp_name, "figures MISS (no figures.json yet)")
        return None
    try:
        with open(figures_path) as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            _log(exp_name, "figures MISS (payload not a dict)")
            return None
        current_fp = _source_fingerprint(exp_name)
        cached_fp = payload.get("fingerprint")
        if cached_fp != current_fp:
            _log(
                exp_name,
                "figures MISS (fingerprint changed) "
                f"[{_fingerprint_diff(cached_fp, current_fp)}]",
            )
            return None
        _log(exp_name, f"figures HIT (cached_at={payload.get('cached_at', '?')})")
        return _deserialize_figures(payload.get("figures", {}))
    except Exception as e:
        # Unreadable figures cache (e.g. a serialization-format change):
        # purge the whole experiment cache so it is rebuilt cleanly.
        _log(exp_name, f"figures MISS (corrupt, purging): {e}")
        invalidate(exp_name)
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
        _log(exp_name, "SAVE skipped (run still in progress)")
        return False
    runs_path, meta_path, figures_path = _cache_paths(exp_name)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(runs_path, index=False)
    # Adopt this experiment onto per-experiment stamp keying *before* taking
    # the fingerprint, so the cache we write here is immediately stamp-keyed
    # and a later unrelated run can't invalidate it via the shared db mtime.
    _ensure_stamp(exp_name)
    payload = {
        "metadata": metadata,
        "summary": summary,
        "fingerprint": _source_fingerprint(exp_name),
        "cached_at": datetime.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(payload, f, default=str)
    if figures is not None:
        _write_figures(exp_name, figures_path, figures)
    keyed = "stamp" if "stamp" in payload["fingerprint"] else "legacy db/artifact mtime"
    _log(
        exp_name,
        f"SAVED ({len(df)} runs, figures={'yes' if figures is not None else 'no'}, "
        f"fingerprint keyed on {keyed})",
    )
    return True


def _write_figures(exp_name: str, figures_path: Path, figures: dict) -> None:
    figures_path.parent.mkdir(parents=True, exist_ok=True)
    # Same rationale as save(): key the figures cache on the per-experiment
    # stamp, not the shared db mtime, so unrelated runs don't reload figures.
    _ensure_stamp(exp_name)
    payload = {
        # Embed an independent fingerprint so figures stay correctly
        # invalidated even when runs/meta can't be refreshed.
        "fingerprint": _source_fingerprint(exp_name),
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
    _write_figures(exp_name, figures_path, figures)
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


def remove_all_stamps() -> None:
    """Remove the stamp file from every railroad experiment (best-effort).

    Forces a clean fingerprint baseline so the next load fully rebuilds.
    """
    try:
        experiments = mlflow.search_experiments()  # type: ignore[possibly-missing-attribute]
    except Exception:
        return
    for exp in experiments:
        try:
            remove_stamp(exp.name)
        except Exception:
            pass
