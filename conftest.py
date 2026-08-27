# conftest.py
import os
import pathlib

DATA_DIR = pathlib.Path(__file__).parent / "resources"


def pytest_configure(config):
    # This hook runs in both controller and workers, but only
    # the controller process has no 'workerinput' attribute.
    if not hasattr(config, "workerinput"):
        # controller process
        if not (DATA_DIR / ".download_complete").exists():
            from railroad.environment.procthor import ensure_all_resources
            ensure_all_resources()
            (DATA_DIR / ".download_complete").touch()


def _worker_count(config):
    """How many xdist workers this run will use, seen from any process."""
    workerinput = getattr(config, "workerinput", None)
    if workerinput is not None:
        return int(workerinput.get("workercount", 1))
    numprocesses = getattr(config.option, "numprocesses", None)
    if isinstance(numprocesses, int) and numprocesses > 0:
        return numprocesses
    return os.cpu_count() or 1


def pytest_collection_modifyitems(config, items):
    """Start one `slow` test on each xdist worker rather than queueing them together.

    `--dist worksteal` hands every worker one contiguous slice of the collection
    up front, and stealing can only take tests from the *tail* of a worker's
    queue -- so a test at the head of a slice can never be rebalanced away. Long
    tests adjacent in collection order therefore share a slice and run back to
    back with no recourse. The two ProcTHOR integration tests did exactly that:
    26.6s and 18.4s in series on one worker, while the other eleven finished
    inside 7s and idled for the remaining 43s.

    Dealing the slow tests round-robin across the slices, each at the head of
    one, starts the long poles concurrently instead. Ordering within each group
    is preserved, so the fast bulk keeps its module grouping.
    """
    workers = _worker_count(config)
    if workers < 2:
        return

    slow = [item for item in items if item.get_closest_marker("slow")]
    if not slow:
        return
    fast = [item for item in items if not item.get_closest_marker("slow")]

    # Mirror xdist's own split: slice i takes len(remaining) // (workers - i).
    sizes, remaining = [], len(items)
    for i in range(workers):
        size = remaining // (workers - i)
        sizes.append(size)
        remaining -= size

    ordered, cursor = [], 0
    for i, size in enumerate(sizes):
        head = slow[i::workers]
        ordered.extend(head)
        fill = max(0, size - len(head))
        ordered.extend(fast[cursor:cursor + fill])
        cursor += fill
    ordered.extend(fast[cursor:])
    items[:] = ordered
