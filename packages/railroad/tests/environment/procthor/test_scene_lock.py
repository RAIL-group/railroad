"""Tests for the cross-process ProcTHOR scene-generation lock.

Generating an uncached scene starts a Unity controller; several at once can
take a machine down, and benchmark workers are separate processes. These check
that the lock is real across processes and that it never becomes a way for a
run to fail.

Children report through append-only files rather than a multiprocessing
Manager, which needs a socket and is not always available in a sandbox.
CLOCK_MONOTONIC is system-wide on Linux, so the timestamps are comparable.
"""

import multiprocessing as mp
import time
from pathlib import Path

import pytest

from railroad.environment.procthor._scene_lock import scene_generation_lock


def _hold(lock_str: str, log_str: str, hold_seconds: float) -> None:
    with scene_generation_lock(path=Path(lock_str)) as held:
        with open(log_str, "a") as handle:
            handle.write(f"enter {held} {time.monotonic()}\n")
            handle.flush()
        time.sleep(hold_seconds)
        with open(log_str, "a") as handle:
            handle.write(f"exit {held} {time.monotonic()}\n")
            handle.flush()


def _wait_for(log: Path, token: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log.exists() and token in log.read_text():
            return
        time.sleep(0.05)
    raise AssertionError(f"{token!r} never appeared in {log}")


def _intervals(log: Path) -> list[tuple[float, float]]:
    """(enter, exit) pairs in the order they were written."""
    events = [line.split() for line in log.read_text().splitlines() if line]
    assert all(held == "True" for _kind, held, _t in events), events
    stack, spans = [], []
    for kind, _held, stamp in events:
        if kind == "enter":
            stack.append(float(stamp))
        else:
            spans.append((stack.pop(), float(stamp)))
    return spans


def test_lock_is_exclusive_across_processes(tmp_path):
    """The second process must wait for the first, not run alongside it."""
    lock, log = tmp_path / "lock", tmp_path / "log"
    ctx = mp.get_context("fork")

    first = ctx.Process(target=_hold, args=(str(lock), str(log), 1.0))
    first.start()
    _wait_for(log, "enter")

    second = ctx.Process(target=_hold, args=(str(lock), str(log), 0.0))
    second.start()
    first.join(timeout=30)
    second.join(timeout=30)
    assert first.exitcode == 0 and second.exitcode == 0

    spans = _intervals(log)
    assert len(spans) == 2, log.read_text()
    (a_start, a_end), (b_start, b_end) = sorted(spans)
    assert b_start >= a_end, (
        f"the two processes overlapped inside the lock: {spans}"
    )
    assert b_end >= b_start


def test_timeout_fails_open_rather_than_blocking_a_run(tmp_path):
    """A lock we cannot take should slow things down, never stop them."""
    lock, log = tmp_path / "lock", tmp_path / "log"
    ctx = mp.get_context("fork")
    holder = ctx.Process(target=_hold, args=(str(lock), str(log), 3.0))
    holder.start()
    _wait_for(log, "enter")

    began = time.monotonic()
    with scene_generation_lock(timeout=0.5, path=lock) as held:
        assert held is False
    assert time.monotonic() - began < 3.0, "should have given up, not waited it out"
    holder.join(timeout=30)


def test_an_unwritable_location_yields_false(tmp_path):
    """Failing open: a locking problem must not stop a run that would work."""
    (tmp_path / "not-a-dir").write_text("regular file")
    with scene_generation_lock(path=tmp_path / "not-a-dir" / "nested" / "lock") as held:
        assert held is False


@pytest.mark.parametrize("timeout", [0.0, 0.5])
def test_the_lock_is_released_on_exit(tmp_path, timeout):
    """Otherwise the second scene of a sweep would hang forever."""
    lock = tmp_path / "lock"
    for _ in range(3):
        with scene_generation_lock(timeout=timeout, path=lock) as held:
            assert held is True
