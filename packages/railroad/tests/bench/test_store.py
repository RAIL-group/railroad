"""Tests for opening the tracking database (railroad.bench.store).

MLflow builds a fresh SQLite database by creating the initial schema and then
replaying every migration to head, several of them alembic batch operations
that work by way of an ``_alembic_tmp_<table>`` copy. Two processes doing that
at once, or one interrupted partway, leaves a temp table behind -- and that
wedges the database for good, since every later open replays the migration and
finds the table already there.

These cover the lock that stops the collision, and the repair that survives one
happening anyway.
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from railroad.bench import store


def _uri(tmp_path):
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def _tables(database):
    with sqlite3.connect(database) as connection:
        return {row[0] for row in connection.execute(
            "select name from sqlite_master where type = 'table'"
        )}


def test_sqlite_path_reads_both_spellings():
    """Three slashes is relative, four is absolute; anything else is not a file."""
    assert store.sqlite_path("sqlite:///mlflow.db") == Path("mlflow.db")
    assert store.sqlite_path("sqlite:////tmp/x/mlflow.db") == Path("/tmp/x/mlflow.db")
    assert store.sqlite_path("http://localhost:5000") is None


def test_opening_a_fresh_database_builds_it_to_head(tmp_path):
    import mlflow

    store.use_tracking_uri(_uri(tmp_path))

    assert mlflow.get_tracking_uri() == _uri(tmp_path)
    with sqlite3.connect(tmp_path / "mlflow.db") as connection:
        assert connection.execute("select * from alembic_version").fetchone()
    assert not [name for name in _tables(tmp_path / "mlflow.db")
                if name.startswith("_alembic_tmp_")]


def test_the_lock_lives_beside_the_database_and_is_hidden(tmp_path):
    store.use_tracking_uri(_uri(tmp_path))

    lock = store.lock_path(tmp_path / "mlflow.db")
    assert lock.parent == tmp_path
    assert lock.name.startswith(".")
    assert lock.exists()


_OPEN_STORE = """
import sys
from railroad.bench import store
try:
    store.use_tracking_uri(sys.argv[1])
except Exception as error:
    sys.exit(f"{type(error).__name__}: {error}")
"""


@pytest.mark.slow
def test_concurrent_first_opens_do_not_collide(tmp_path):
    """The regression: a sweep and a dashboard reaching a fresh database at once.

    Separate processes, because the lock is between processes -- threads in one
    interpreter share MLflow's store cache and never race the way this did.
    Six of them because the collision is a real interleaving, not a certainty:
    unlocked, this failed on four runs in five.
    """
    processes = [
        subprocess.Popen([sys.executable, "-c", _OPEN_STORE, _uri(tmp_path)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(6)
    ]
    failures = []
    for process in processes:
        _, err = process.communicate(timeout=300)
        if process.returncode != 0:
            failures.append(err.strip().splitlines()[-1] if err.strip() else "?")

    assert not failures, f"{len(failures)} of 6 failed: {failures[:3]}"
    leftover = [name for name in _tables(tmp_path / "mlflow.db")
                if name.startswith("_alembic_tmp_")]
    assert not leftover, f"a migration died and left {leftover}"


def _wedged(database):
    """A database in the state an interrupted batch migration leaves behind."""
    with sqlite3.connect(database) as connection:
        connection.execute("create table params (x integer)")
        connection.execute("create table _alembic_tmp_params (x integer)")


def test_a_wedged_database_is_repaired_and_the_open_retried(tmp_path, monkeypatch):
    """The failure that wedged the tutorial playground, and the way back."""
    database = tmp_path / "mlflow.db"
    _wedged(database)
    attempts = []

    def flaky_open():
        attempts.append(_tables(database))
        if len(attempts) == 1:
            raise RuntimeError("table _alembic_tmp_params already exists")

    monkeypatch.setattr(store, "_open_store", flaky_open)
    store.use_tracking_uri(_uri(tmp_path))

    assert len(attempts) == 2, "the repaired database is opened again"
    assert "_alembic_tmp_params" in attempts[0]
    assert "_alembic_tmp_params" not in attempts[1]
    assert "params" in attempts[1], "the real table is not collateral"


def test_a_failure_with_nothing_to_repair_is_raised(tmp_path, monkeypatch):
    """Only the leftover-temp-table failure is ours to swallow."""
    def broken_open():
        raise RuntimeError("disk is on fire")

    monkeypatch.setattr(store, "_open_store", broken_open)
    with pytest.raises(RuntimeError, match="disk is on fire"):
        store.use_tracking_uri(_uri(tmp_path))


def test_a_temp_table_holding_the_only_rows_is_left_alone(tmp_path):
    """Dropped-original, not-yet-renamed: the copy is the data. Do not drop it."""
    database = tmp_path / "mlflow.db"
    with sqlite3.connect(database) as connection:
        connection.execute("create table _alembic_tmp_orphan (x integer)")

    assert store._drop_stale_batch_tables(database) == []
    assert "_alembic_tmp_orphan" in _tables(database)


def test_a_missing_database_has_nothing_to_repair(tmp_path):
    assert store._drop_stale_batch_tables(tmp_path / "absent.db") == []
    assert store._drop_stale_batch_tables(None) == []


def test_a_remote_uri_makes_no_lock_file(tmp_path, monkeypatch):
    """Only SQLite is ours to serialise; a tracking server does its own."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(store, "_open_store", lambda: None)

    store.use_tracking_uri("http://localhost:5000")

    assert list(tmp_path.iterdir()) == []
