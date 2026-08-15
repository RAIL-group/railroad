"""One AI2-THOR simulator at a time, across processes.

Building a scene that is not yet cached starts a Unity ``Controller``. That is
heavy enough that several at once can take a machine down, and the benchmark
runner happily fans out a dozen worker *processes* -- so a threading lock is no
use here. This is an ``flock`` on a file beside the scene cache, which
serialises scene generation across every process on the host, whatever started
them.

The lock is only taken on a cache miss, so a warm cache costs nothing. Callers
must re-check the cache after acquiring it: whoever held it before may have
generated exactly the scene we were about to.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

LOCK_FILENAME = ".scene-generation.lock"
POLL_SECONDS = 1.0

DEFAULT_TIMEOUT = float(os.environ.get("PROCTHOR_SCENE_LOCK_TIMEOUT", 1800.0))
"""Long by default: generating a scene means starting Unity, and a queue of
workers behind one slow generation is the normal case, not a stuck lock."""


def lock_path() -> Path:
    """Where the lock file lives -- beside the ProcTHOR-10k data."""
    from .resources import get_procthor_10k_dir

    return get_procthor_10k_dir() / LOCK_FILENAME


@contextmanager
def scene_generation_lock(
    timeout: Optional[float] = None, path: Optional[Path] = None
) -> Iterator[bool]:
    """Hold the exclusive scene-generation lock for the duration of the block.

    Yields True when the lock was held, False when it could not be taken (no
    ``fcntl`` on this platform, an unwritable directory, a filesystem that
    cannot lock, or the timeout elapsed). Failing open is deliberate: a locking
    problem should slow a machine down, not stop a run that would otherwise
    have worked.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - not reachable on Linux/macOS
        yield False
        return

    deadline = time.monotonic() + (DEFAULT_TIMEOUT if timeout is None else timeout)

    try:
        # lock_path() creates the resources directory, so it fails open too.
        target = path or lock_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = open(target, "a+")
    except OSError:
        yield False
        return

    announced = False
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                # Someone else holds it. This is the case worth waiting out.
                if time.monotonic() >= deadline:
                    yield False
                    return
                if not announced:
                    print("[procthor] waiting for another process to finish "
                          "generating a scene...")
                    announced = True
                time.sleep(POLL_SECONDS)
            except OSError as error:
                # Anything else means flock will never succeed here, not that
                # it is busy: ENOLCK, EOPNOTSUPP, or EINVAL on NFS without
                # lockd, on CIFS, and on some container overlay filesystems.
                # Retrying would poll for the whole timeout -- half an hour by
                # default -- while claiming to wait on another process. That is
                # a live path precisely because PROCTHOR_RESOURCES_DIR exists
                # to put the cache somewhere shared.
                print(f"[procthor] cannot lock {target} ({error.strerror}); "
                      "generating without serialising across processes")
                yield False
                return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
