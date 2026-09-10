"""Rewrite absolute artifact paths in an MLflow SQLite tracking DB.

MLflow's SQLite backend stores the artifact root as an *absolute* filesystem
path for every experiment and every run (``experiments.artifact_location`` and
``runs.artifact_uri``, plus ``logged_models.artifact_location`` on newer
schemas). When you copy ``mlflow.db`` + ``mlruns/`` to a different machine or a
different directory (e.g. rsync from the lab desktop), those stored paths still
point at the *old* location, so the benchmarks dashboard / ``mlflow`` server
can't resolve any artifact and the dashboard's ``/artifact/<run_id>/...`` route
404s. Metrics and params still load because they are plain DB columns.

This script rewrites the stored path prefix (``old_root`` -> ``new_root``).
Non-local URIs (``mlflow-artifacts:``, ``s3://``, ``http``...) are left alone.

Dry-run by default. ``--apply`` writes, making a timestamped copy of the DB
first (``mlflow.db.bak-YYYYmmdd-HHMMSS``) unless ``--no-backup`` is given.

Examples
--------
# auto-detect: old root from the DB, new root = the dir that contains ./mlruns
uv run python scripts/rewrite_mlflow_artifact_paths.py

# inspect, then apply
uv run python scripts/rewrite_mlflow_artifact_paths.py --db mlflow.db
uv run python scripts/rewrite_mlflow_artifact_paths.py --db mlflow.db --apply --verify

# be explicit about both ends
uv run python scripts/rewrite_mlflow_artifact_paths.py \
    --old-root /home/rail/smille20/railroad \
    --new-root /Users/sammiller/dev/interruption-project/railroad \
    --apply
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import time
from urllib.parse import urlparse

# (table, column) pairs that hold an absolute artifact path. Tables/columns that
# don't exist in a given DB's schema are skipped silently.
PATH_COLUMNS: list[tuple[str, str]] = [
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),  # MLflow >= 2.9 / 3.x
]


def _split_file_scheme(value: str) -> tuple[str, str]:
    """Return ``(prefix, bare_path)`` where prefix is ``""`` or ``"file://"``."""
    if value.startswith("file://"):
        return "file://", urlparse(value).path
    return "", value


def _is_local_abs(value: str) -> bool:
    """True if ``value`` is a local absolute path (bare or ``file://``)."""
    if not value:
        return False
    scheme = urlparse(value).scheme
    # A drive letter ("c:\...") parses as scheme "c" -- treat 1-char schemes as local.
    if scheme and scheme != "file" and len(scheme) > 1:
        return False
    _, bare = _split_file_scheme(value)
    return os.path.isabs(bare)


def _existing_columns(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for table, column in PATH_COLUMNS:
        try:
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        except sqlite3.OperationalError:
            continue
        if column in cols:
            found.append((table, column))
    return found


def _collect_values(
    conn: sqlite3.Connection, cols: list[tuple[str, str]]
) -> dict[tuple[str, str], list[str]]:
    values: dict[tuple[str, str], list[str]] = {}
    for table, column in cols:
        values[(table, column)] = [
            row[0]
            for row in conn.execute(
                f"SELECT {column} FROM {table} "
                f"WHERE {column} IS NOT NULL AND {column} != ''"
            )
        ]
    return values


def _detect_old_root(values: list[str], marker: str) -> str | None:
    """Longest common ancestor of the local paths, trimmed at ``/<marker>/``."""
    candidates: list[str] = []
    key = f"/{marker}/"
    for value in values:
        if not _is_local_abs(value):
            continue
        _, bare = _split_file_scheme(value)
        norm = bare.replace("\\", "/")
        idx = norm.rfind(key)
        candidates.append(norm[:idx] if idx != -1 else os.path.dirname(norm))
    if not candidates:
        return None
    try:
        return os.path.commonpath(candidates)
    except ValueError:  # mix of absolute/relative -- shouldn't happen
        return None


def _default_new_root(db_path: str, marker: str) -> str:
    db_dir = os.path.dirname(os.path.abspath(db_path)) or os.getcwd()
    for candidate in (db_dir, os.getcwd()):
        if os.path.isdir(os.path.join(candidate, marker)):
            return candidate
    return db_dir


def _rewrite_one(value: str, old_root: str, new_root: str) -> str | None:
    """Return the rewritten value, or ``None`` if it doesn't match ``old_root``."""
    if not _is_local_abs(value):
        return None
    prefix, bare = _split_file_scheme(value)
    norm = bare.replace("\\", "/")
    old = old_root.replace("\\", "/").rstrip("/")
    if norm != old and not norm.startswith(old + "/"):
        return None
    return prefix + new_root.replace("\\", "/").rstrip("/") + norm[len(old):]


def _plan(
    values: dict[tuple[str, str], list[str]], old_root: str, new_root: str
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    plan: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for key, rows in values.items():
        changes: list[tuple[str, str]] = []
        for value in rows:
            new_value = _rewrite_one(value, old_root, new_root)
            if new_value and new_value != value:
                changes.append((value, new_value))
        plan[key] = changes
    return plan


def _verify(conn: sqlite3.Connection, cols: list[tuple[str, str]]) -> None:
    checked = missing = 0
    for table, column in cols:
        for (value,) in conn.execute(
            f"SELECT {column} FROM {table} "
            f"WHERE {column} IS NOT NULL AND {column} != ''"
        ):
            if not _is_local_abs(value):
                continue
            _, bare = _split_file_scheme(value)
            checked += 1
            if not os.path.isdir(bare):
                missing += 1
                if missing <= 5:
                    print(f"  missing on disk: {bare}")
    print(f"verify: {checked - missing}/{checked} artifact dirs present on disk")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db", default="mlflow.db",
        help="path to the MLflow SQLite DB (default: ./mlflow.db)",
    )
    parser.add_argument(
        "--old-root",
        help="path prefix to replace (default: auto-detected from the DB)",
    )
    parser.add_argument(
        "--new-root",
        help="replacement prefix (default: the dir containing ./<marker>)",
    )
    parser.add_argument(
        "--marker", default="mlruns",
        help="artifact-root dir name used to anchor detection (default: mlruns)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="write the changes (without this flag it's a dry run)",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="skip the timestamped DB backup when applying",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="after rewriting, check that each artifact dir exists on disk",
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.db):
        print(f"error: no such DB: {args.db}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(args.db)
    try:
        cols = _existing_columns(conn)
        if not cols:
            print(
                "error: none of the expected path columns "
                f"({', '.join(f'{t}.{c}' for t, c in PATH_COLUMNS)}) exist here",
                file=sys.stderr,
            )
            return 2
        values = _collect_values(conn, cols)
    finally:
        conn.close()

    flat = [value for rows in values.values() for value in rows]
    new_root = os.path.abspath(args.new_root or _default_new_root(args.db, args.marker))
    old_root = args.old_root or _detect_old_root(flat, args.marker)

    if not old_root:
        print("Could not auto-detect --old-root (no local absolute paths in the DB).")
        print("Re-run with --old-root set explicitly. Sample stored values:")
        for value in flat[:5]:
            print(f"  {value}")
        return 1

    print(f"DB       : {os.path.abspath(args.db)}")
    print(f"old root : {old_root}")
    print(f"new root : {new_root}")
    marker_dir = os.path.join(new_root, args.marker)
    if not os.path.isdir(marker_dir):
        print(f"warning: {marker_dir} does not exist -- is --new-root correct?")
    if old_root.replace("\\", "/").rstrip("/") == new_root.replace("\\", "/").rstrip("/"):
        print("\nnothing to do: old root == new root")
        return 0
    print()

    plan = _plan(values, old_root, new_root)
    total = sum(len(changes) for changes in plan.values())

    for (table, column), changes in plan.items():
        print(f"{table}.{column}: {len(changes)}/{len(values[(table, column)])} rows to rewrite")
        for old, new in changes[:2]:
            print(f"    - {old}")
            print(f"    + {new}")
        if len(changes) > 2:
            print(f"    ... (+{len(changes) - 2} more)")

    skipped = [value for value in flat if not _is_local_abs(value)]
    if skipped:
        print(f"\nleaving {len(skipped)} non-local URI(s) untouched (e.g. {skipped[0]})")

    if total == 0:
        print("\nno rows matched the old root -- nothing to do")
        return 0

    if not args.apply:
        print(f"\ndry run: {total} row(s) would change. Re-run with --apply to write.")
        return 0

    if not args.no_backup:
        backup = f"{args.db}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(args.db, backup)
        print(f"\nbackup written: {backup}")

    conn = sqlite3.connect(args.db)
    try:
        with conn:  # single transaction; rolls back on error
            for (table, column), changes in plan.items():
                conn.executemany(
                    f"UPDATE {table} SET {column} = ? WHERE {column} = ?",
                    [(new, old) for old, new in changes],
                )
        print(f"applied {total} row update(s)")
        if args.verify:
            _verify(conn, cols)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
