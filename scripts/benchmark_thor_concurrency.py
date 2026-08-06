"""
Benchmark: how many concurrent AI2-THOR Controller instances can this machine
sustain before per-instance latency starts degrading?

Exercises the exact code path datagen.py's get_randomized_procthor_data uses
to build a randomized scene (construct_procthor_kitchen_environment), with a
fresh, never-before-seen object_seed per instance so every call is a genuine
cache miss and must spin up a live ai2thor.controller.Controller (see
ThorInterface.__init__ / _randomize_object_locations in the railroad package).

Must be run with cwd == railroad-env/, so get_procthor_10k_dir() resolves to
the already-downloaded resources/procthor-10k/data.jsonl.

Usage:
    python benchmark_thor_concurrency.py --levels 1 2 4 8 --timeout 180

While this runs, watch `top -o cpu` (or Activity Monitor) in another
terminal to see whether the bottleneck is CPU saturation.
"""
import argparse
import multiprocessing as mp
import subprocess
import time
from pathlib import Path
from statistics import mean, median

PROCTHOR_SEED = 201
TIMEOUT_SENTINEL = "TIMEOUT"


def _child_pids(pid: int) -> list[int]:
    """Direct child PIDs of `pid`, via pgrep (stdlib-only, no psutil dependency)."""
    result = subprocess.run(
        ["pgrep", "-P", str(pid)], capture_output=True, text=True, check=False
    )
    return [int(p) for p in result.stdout.split()]


def _kill_process_tree(pid: int) -> None:
    """
    Kill `pid` and all its descendants (e.g. a hung worker's spawned Unity
    subprocess, which multiprocessing.Process.terminate() does NOT reach).
    Escalates from SIGTERM to SIGKILL for anything still alive after a beat.
    """
    tree = [pid]
    frontier = [pid]
    while frontier:
        children = _child_pids(frontier.pop())
        tree.extend(children)
        frontier.extend(children)

    for target in tree:
        subprocess.run(["kill", str(target)], capture_output=True, check=False)
    time.sleep(1)
    for target in tree:
        subprocess.run(["kill", "-9", str(target)], capture_output=True, check=False)


def _build_one(object_seed: int, result_queue: mp.Queue) -> None:
    """Worker entrypoint: build one randomized scene, time it, report back."""
    from interruption.environments import construct_procthor_kitchen_environment

    start = time.perf_counter()
    error = None
    try:
        construct_procthor_kitchen_environment(PROCTHOR_SEED, object_seed=object_seed)
    except Exception as exc:  # benchmark needs to record any failure, not crash the batch
        error = repr(exc)
    elapsed = time.perf_counter() - start
    result_queue.put((object_seed, elapsed, error))


def run_batch(
    concurrency: int, seed_offset: int, timeout: float
) -> tuple[list[tuple[int, float, str | None]], float]:
    """Run `concurrency` scene builds in parallel, each with a fresh object_seed."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    object_seeds = list(range(seed_offset, seed_offset + concurrency))

    procs = [
        ctx.Process(target=_build_one, args=(seed, result_queue))
        for seed in object_seeds
    ]
    batch_start = time.perf_counter()
    for p in procs:
        p.start()

    results = []
    deadline = batch_start + timeout
    for _ in procs:
        remaining = max(0.0, deadline - time.perf_counter())
        try:
            results.append(result_queue.get(timeout=remaining))
        except Exception:
            pass  # queue empty at deadline; unfinished workers handled below

    batch_elapsed = time.perf_counter() - batch_start

    finished_seeds = {r[0] for r in results}
    for p, seed in zip(procs, object_seeds):
        if seed not in finished_seeds:
            # terminate() only kills the Python worker, not its child Unity
            # subprocess, which would otherwise be orphaned and keep running.
            if p.pid is not None:
                _kill_process_tree(p.pid)
            results.append((seed, timeout, TIMEOUT_SENTINEL))
        p.join(timeout=5)

    results.sort(key=lambda r: r[0])
    return results, batch_elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[1, 2, 4, 8],
        help="Concurrency levels to test, in order, e.g. --levels 1 2 4 8",
    )
    parser.add_argument(
        "--timeout", type=float, default=180.0,
        help="Per-batch timeout in seconds before an unfinished worker is terminated",
    )
    parser.add_argument(
        "--seed-start", type=int, default=int(time.time()) % 100_000 + 500_000,
        help="Base object_seed to start from, so every run uses fresh (uncached) seeds",
    )
    args = parser.parse_args()

    data_file = Path("resources/procthor-10k/data.jsonl")
    if not data_file.exists():
        raise SystemExit(
            f"{data_file} not found relative to cwd. Run this script from railroad-env/."
        )

    print(f"Base object_seed offset: {args.seed_start} (fresh/uncached for this run)")
    header = (
        f"{'concurrency':>11} | {'batch_wall_s':>12} | {'mean_per_s':>10} | "
        f"{'median_per_s':>12} | {'max_per_s':>9} | errors"
    )
    print(header)
    print("-" * len(header))

    next_seed = args.seed_start
    for level in args.levels:
        results, batch_elapsed = run_batch(level, next_seed, args.timeout)
        next_seed += level  # advance so each level uses disjoint, still-fresh seeds

        times = [r[1] for r in results]
        errors = [r for r in results if r[2] is not None]
        print(
            f"{level:>11} | {batch_elapsed:>12.2f} | {mean(times):>10.2f} | "
            f"{median(times):>12.2f} | {max(times):>9.2f} | {len(errors)}"
        )
        for seed, elapsed, err in errors:
            print(f"    seed={seed} FAILED after {elapsed:.1f}s: {err}")

    print(
        "\nIf per-instance time climbs sharply between levels, that's the "
        "concurrency ceiling for this machine. If any batch timed out, check "
        "for orphaned Unity processes with: ps aux | grep thor-OSXIntel64"
    )


if __name__ == "__main__":
    main()
