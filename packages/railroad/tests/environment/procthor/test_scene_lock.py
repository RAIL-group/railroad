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


def _log(log_str: str, *fields: object) -> None:
    with open(log_str, "a") as handle:
        handle.write(" ".join(str(field) for field in fields) + "\n")
        handle.flush()


def _hold(
    lock_str: str,
    log_str: str,
    tag: str,
    release_str: str = "",
    max_hold: float = 10.0,
) -> None:
    """Take the lock, then hold it until *release_str* appears on disk.

    Holding until the parent says so -- rather than for a fixed duration --
    is what keeps these tests honest under ``spawn``: the parent waits for
    the child to reach the lock instead of guessing how long a fresh
    interpreter takes to get there. ``max_hold`` is a backstop for a parent
    that dies, not a timing parameter; no assertion depends on it.

    ``trying`` is logged immediately before the attempt, so the parent can
    tell "about to contend" from "has not started yet" -- under ``spawn``
    those are a second or more apart.
    """
    release = Path(release_str) if release_str else None
    _log(log_str, "trying", tag, time.monotonic())
    with scene_generation_lock(path=Path(lock_str)) as held:
        _log(log_str, "enter", tag, held, time.monotonic())
        deadline = time.monotonic() + max_hold
        while release is not None and not release.exists():
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        _log(log_str, "exit", tag, held, time.monotonic())


def _wait_for(log: Path, token: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log.exists() and token in log.read_text():
            return
        time.sleep(0.05)
    raise AssertionError(f"{token!r} never appeared in {log}")


def _intervals(log: Path) -> list[tuple[float, float]]:
    """(enter, exit) pairs, one per process, earliest first."""
    enters: dict[str, float] = {}
    exits: dict[str, float] = {}
    for fields in (line.split() for line in log.read_text().splitlines() if line):
        kind, tag = fields[0], fields[1]
        if kind == "trying":
            continue
        held, stamp = fields[2], fields[3]
        assert held == "True", log.read_text()
        (enters if kind == "enter" else exits)[tag] = float(stamp)
    assert enters.keys() == exits.keys(), log.read_text()
    return sorted((enters[tag], exits[tag]) for tag in enters)


@pytest.mark.slow
def test_lock_is_exclusive_across_processes(tmp_path):
    """The second process must wait for the first, not run alongside it.

    The first process holds until we release it, and we do not release it
    until the second has announced it is about to contend -- so the overlap
    the assertion rules out is one that was actually possible. A fixed hold
    instead has to outlast the second process's ``spawn`` (measured at over
    a second on a loaded machine); when it does not, the first is gone
    before the second reaches the lock and ``b_start >= a_end`` holds
    vacuously -- a removed lock would still pass.

    Slow because it pays for two interpreter startups, not for a sleep.
    """
    lock, log = tmp_path / "lock", tmp_path / "log"
    release = tmp_path / "release"
    ctx = mp.get_context("spawn")

    first = ctx.Process(
        target=_hold, args=(str(lock), str(log), "first", str(release)),
    )
    first.start()
    _wait_for(log, "enter first")

    second = ctx.Process(target=_hold, args=(str(lock), str(log), "second"))
    second.start()
    _wait_for(log, "trying second")
    # `trying` is logged on the line before the attempt; give it room to
    # reach the flock itself while the first is demonstrably still inside.
    time.sleep(0.2)
    release.touch()

    first.join(timeout=30)
    second.join(timeout=30)
    assert first.exitcode == 0 and second.exitcode == 0

    spans = _intervals(log)
    assert len(spans) == 2, log.read_text()
    (a_start, a_end), (b_start, b_end) = spans
    assert b_start >= a_end, (
        f"the two processes overlapped inside the lock: {spans}"
    )
    assert b_end >= b_start


@pytest.mark.slow
def test_timeout_fails_open_rather_than_blocking_a_run(tmp_path, monkeypatch):
    """A lock we cannot take should slow things down, never stop them.

    Slow for the same reason as the test above: one interpreter startup.
    """
    lock, log = tmp_path / "lock", tmp_path / "log"
    release = tmp_path / "release"
    ctx = mp.get_context("spawn")
    holder = ctx.Process(
        target=_hold, args=(str(lock), str(log), "holder", str(release)),
    )
    holder.start()
    _wait_for(log, "enter holder")

    # Without this the retry loop sleeps POLL_SECONDS (1.0) before rechecking
    # the deadline, so *every* timeout below a second still costs a second.
    monkeypatch.setattr(
        "railroad.environment.procthor._scene_lock.POLL_SECONDS", 0.01,
    )
    began = time.monotonic()
    with scene_generation_lock(timeout=0.2, path=lock) as held:
        assert held is False
    elapsed = time.monotonic() - began
    release.touch()

    # Lower bound: it gave up *because of the timeout* -- a lock that failed
    # open instantly (unwritable path, unsupported filesystem) also returns
    # False, and without this the test could not tell the two apart. Upper
    # bound: it gave up rather than waiting the holder out. The holder is
    # still holding at this point -- it waits for `release`, which is only
    # touched above -- so the bound is not racing a fixed hold, and the slack
    # between them is all scheduling headroom.
    assert 0.2 <= elapsed < 5.0, f"expected a ~0.2s give-up, got {elapsed:.2f}s"
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


class TestFilesystemsThatCannotLock:
    """``flock`` fails two very different ways, and only one is worth waiting on.

    ``BlockingIOError`` means another process holds it -- wait. Everything else
    (``ENOLCK``, ``EOPNOTSUPP``, ``EINVAL``) means this filesystem will never
    grant it: NFS without lockd, CIFS, some container overlays. Retrying those
    polls for the entire timeout -- half an hour by default -- while printing
    that it is waiting on another process. ``PROCTHOR_RESOURCES_DIR`` exists to
    put the cache somewhere shared, so this is a reachable path, not a theory.
    """

    @staticmethod
    def _flock_raising(error: OSError):
        def flock(fileno, flags):
            raise error
        return flock

    def test_an_unsupported_filesystem_fails_open_immediately(
        self, tmp_path, monkeypatch,
    ):
        import errno
        import fcntl

        monkeypatch.setattr(
            fcntl, "flock",
            self._flock_raising(OSError(errno.ENOLCK, "No locks available")),
        )
        started = time.monotonic()
        with scene_generation_lock(
            timeout=30.0, path=tmp_path / "lock",
        ) as held:
            assert held is False
        # Not "did it eventually give up" -- did it give up without polling.
        assert time.monotonic() - started < 1.0

    def test_contention_is_still_waited_out(self, tmp_path, monkeypatch):
        """The retry path must survive the narrowing above."""
        import fcntl

        calls = []
        real_flock = fcntl.flock

        def flock(fileno, flags):
            calls.append(flags)
            if len(calls) < 3:
                raise BlockingIOError("busy")
            return real_flock(fileno, flags)

        monkeypatch.setattr(fcntl, "flock", flock)
        monkeypatch.setattr(
            "railroad.environment.procthor._scene_lock.POLL_SECONDS", 0.01,
        )
        with scene_generation_lock(timeout=30.0, path=tmp_path / "lock") as held:
            assert held is True
        assert len(calls) >= 3


def test_an_unwritable_resources_directory_fails_open(monkeypatch):
    """Documented behaviour, but lock_path() creates that directory itself."""
    from railroad.environment.procthor import _scene_lock

    def unwritable():
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(_scene_lock, "lock_path", unwritable)
    with scene_generation_lock(timeout=1.0) as held:
        assert held is False
