"""Pointing MLflow at a tracking database, once, without two processes racing.

MLflow builds a brand-new SQLite database by creating the *initial* schema and
then replaying every migration up to head. Several of those migrations are
alembic batch operations, which SQLite cannot do in place: each one creates an
``_alembic_tmp_<table>`` copy, fills it, drops the original and renames.

Two processes doing that to the same fresh file at once interleave, and the
loser dies inside a batch step -- leaving its temp table behind. That does not
just fail the one command: it wedges the database permanently, because every
later open replays the same migration and trips over the table that is already
there. The way back is to drop the leftovers by hand.

A sweep and a dashboard page load are two processes. ``tutorial reset`` with a
dashboard still up is the reliable way to put them both onto a fresh file at
the same moment, which is how this was found.

So the first open goes under an exclusive lock beside the database: whoever
gets there first migrates, the rest wait and find the schema already at head.
MLflow caches the store per process, so every open after that costs a file
open and an uncontended lock.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Iterator, List, Optional
from urllib.parse import urlparse

import mlflow

try:
    import fcntl

    HAVE_FLOCK = True
except ImportError:  # pragma: no cover - not POSIX
    HAVE_FLOCK = False

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
"""SQLite in the working directory, which is what every entry point defaults to."""

_BATCH_PREFIX = "_alembic_tmp_"
"""What alembic calls the copy it builds when SQLite cannot alter in place."""


def use_tracking_uri(tracking_uri: Optional[str] = None) -> str:
    """Point MLflow at *tracking_uri* and make sure its schema is built.

    Returns the URI actually used, so callers can report it.

    A database wedged by an *earlier* interrupted migration -- the lock stops
    two processes colliding, not a sweep stopped with ctrl-c or a machine that
    slept mid-upgrade -- is repaired in place rather than left for the reader
    of a hundred-line traceback to fix by hand.
    """
    resolved = tracking_uri or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(resolved)
    database = sqlite_path(resolved)
    with _init_lock(database):
        # Constructing a client resolves the store, which is what creates the
        # tables and runs the migrations. Doing it here means it happens with
        # the lock held rather than at some arbitrary later call.
        try:
            _open_store()
        except Exception:
            if not _drop_stale_batch_tables(database):
                raise
            _open_store()
    return resolved


def _open_store() -> None:
    """Resolve the tracking store, building or migrating the schema as needed."""
    mlflow.tracking.MlflowClient()  # type: ignore[possibly-missing-attribute]


def _drop_stale_batch_tables(database: Optional[Path]) -> List[str]:
    """Clear ``_alembic_tmp_*`` tables left by a migration that died.

    Returns the names dropped, empty if there was nothing to do -- which is the
    caller's signal that the failure was something else and belongs upwards.

    Held under the initialisation lock, so a *live* migration is not one of the
    things this can walk into. The one temp table it must not touch is one
    whose real table is missing: alembic recreates a table by filling a copy,
    dropping the original and renaming, so in that window the copy is the only
    place the rows are. That is left alone and the error stands.
    """
    if database is None or not database.exists():
        return []
    dropped: List[str] = []
    with closing(sqlite3.connect(database)) as connection:
        names = {
            row[0] for row in connection.execute(
                "select name from sqlite_master where type = 'table'"
            )
        }
        for name in sorted(names):
            if not name.startswith(_BATCH_PREFIX):
                continue
            if name[len(_BATCH_PREFIX):] not in names:
                continue
            connection.execute(f'drop table "{name}"')
            dropped.append(name)
        connection.commit()
    return dropped


def sqlite_path(tracking_uri: str) -> Optional[Path]:
    """The file a ``sqlite:///`` URI names, or ``None`` for any other backend.

    SQLAlchemy spells a relative path with three slashes and an absolute one
    with four, which is one leading slash either way.
    """
    if not tracking_uri.startswith("sqlite:"):
        return None
    path = urlparse(tracking_uri).path
    return Path(path[1:]) if path.startswith("/") else None


def lock_path(database: Path) -> Path:
    """The lock beside *database*. Hidden, because nobody needs to see it."""
    return database.with_name(f".{database.name}.lock")


@contextmanager
def _init_lock(database: Optional[Path]) -> Iterator[None]:
    """Hold the initialisation lock for *database*, if there is one to hold.

    A remote tracking server does its own serialising, and a platform without
    ``flock`` gets the old behaviour rather than an import error.
    """
    if database is None or not HAVE_FLOCK:
        yield
        return
    database.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path(database), "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
