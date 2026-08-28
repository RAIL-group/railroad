"""
Diagnostic script for the memory growth observed in multiprocess_datagen.py's
worker processes.

Runs the exact same worker-share logic as
multiprocess_datagen._generate_worker_share (unmodified — imported directly),
single-process, in small batches, logging after each batch:

  - RSS (psutil) and tracemalloc's traced-Python-allocation total, to tell
    apart "genuine growing set of live Python objects" from "native/allocator
    fragmentation that never gets returned to the OS" (traced memory flat
    while RSS climbs points at the latter).
  - live instance counts (via gc) for the classes most likely to pile up:
    KitchenProcTHOREnvironment, ThorInterface, SceneGraph,
    InterruptionTrajectory, ExperimentData.
  - how many astar_search calls the batch made, and how many of those
    returned success=False (i.e. exhausted num_steps without finding a
    goal) — a proxy for the trajectory-copying blowup described in
    InterruptionTrajectory.create_child (planner.py).
  - how many extra initialize_experiment_data() calls
    get_randomized_procthor_data made beyond the one that succeeded (i.e.
    rejected object/location-count candidates), and how many of those were
    ThorInterface cache misses (which spin up a real AI2-THOR Controller).

Writes no training data to disk: write_datum_to_file is stubbed out to a
no-op for the duration of the run.

Usage:
    uv run python scripts/diagnose_datagen_memory.py
    uv run python scripts/diagnose_datagen_memory.py --checkpoints 40 --batch-size 5

Note: get_randomized_procthor_data's disk caches (./resources/procthor-10k/
cache and randomized_scenes) persist across runs. A first run measures
"cold" behavior; rerunning will show fewer ThorInterface cache misses for
any seed already touched, since those no longer need a Controller.
"""
import argparse
import gc
import sys
import time
import tracemalloc
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).parent))
import multiprocess_datagen as datagen  # noqa: E402

import interruption.planner as planner_mod  # noqa: E402
from interruption.environments import KitchenProcTHOREnvironment  # noqa: E402
from interruption.experiments import ExperimentData  # noqa: E402
from railroad.environment.procthor.scenegraph import SceneGraph  # noqa: E402
from railroad.environment.procthor.thor_interface import ThorInterface  # noqa: E402

WORKER_ID = 0
CHECKPOINT_SEED_STRIDE = 2_000  # headroom so retries within a checkpoint never collide with the next one


# --- instrumentation: astar_search call/cap-hit counting ---------------------
_astar_stats = {"calls": 0, "cap_hits": 0}
_orig_astar_search = planner_mod.astar_search


def _instrumented_astar_search(*args, **kwargs):
    _astar_stats["calls"] += 1
    result = _orig_astar_search(*args, **kwargs)
    if not result[2]:  # success flag
        _astar_stats["cap_hits"] += 1
    return result


# astar_search is called both directly (multiprocess_datagen imported the
# name into its own module globals) and from inside compute_interruption_value
# (resolved via planner_mod's own globals) — both need patching.
planner_mod.astar_search = _instrumented_astar_search  # ty: ignore[invalid-assignment]
datagen.astar_search = _instrumented_astar_search  # ty: ignore[invalid-assignment]

# Silence the per-call tqdm bar inside astar_search — with dozens of
# compute_interruption_value calls per batch this would otherwise be
# hundreds of progress bars of console spam.
class _NullTqdm:
    def __init__(self, *args, **kwargs):
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, *args, **kwargs):
        pass


planner_mod.tqdm = _NullTqdm  # ty: ignore[invalid-assignment]


# --- instrumentation: rejected-candidate counting in get_randomized_procthor_data ---
_env_construction_stats = {"attempts": 0}
_orig_initialize_experiment_data = datagen.initialize_experiment_data


def _instrumented_initialize_experiment_data(*args, **kwargs):
    _env_construction_stats["attempts"] += 1
    return _orig_initialize_experiment_data(*args, **kwargs)


datagen.initialize_experiment_data = _instrumented_initialize_experiment_data  # ty: ignore[invalid-assignment]

# --- instrumentation: ThorInterface disk-cache misses (-> real Controller spin-up) ---
_thor_cache_stats = {"misses": 0}
_orig_load_cache = ThorInterface._load_cache


def _instrumented_load_cache(self, *args, **kwargs):
    result = _orig_load_cache(self, *args, **kwargs)
    if result is None:
        _thor_cache_stats["misses"] += 1
    return result


ThorInterface._load_cache = _instrumented_load_cache

# --- write_datum_to_file -> no-op, so this run never touches real training data ---
datagen.write_datum_to_file = lambda *args, **kwargs: None  # ty: ignore[invalid-assignment]


def _count_live(cls) -> int:
    return sum(1 for obj in gc.get_objects() if isinstance(obj, cls))


def _snapshot() -> dict:
    gc.collect()
    rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    return {
        "rss_mb": rss_mb,
        "traced_mb": traced_current / (1024 * 1024),
        "traced_peak_mb": traced_peak / (1024 * 1024),
        "n_env": _count_live(KitchenProcTHOREnvironment),
        "n_thor": _count_live(ThorInterface),
        "n_scenegraph": _count_live(SceneGraph),
        "n_traj": _count_live(planner_mod.InterruptionTrajectory),
        "n_expdata": _count_live(ExperimentData),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=int, default=20, help="number of batches to run")
    parser.add_argument("--batch-size", type=int, default=5, help="data points generated per batch")
    args = parser.parse_args()

    tracemalloc.start()

    print("Loading task distribution / base scene (seed=%d)..." % datagen.PROCTHOR_SEED)
    env = datagen.construct_procthor_kitchen_environment(datagen.PROCTHOR_SEED)
    task_distribution = datagen.get_alfred_task_distribution(
        env.scene.objects, set(env.scene.locations), one_object_per_taskdist=True
    )
    num_objects = len(env.scene.objects)
    num_locations = len(env.scene.locations)
    del env
    print(f"Interrupting task distribution size: {len(task_distribution[0])}")
    print(f"Running {args.checkpoints} checkpoints x {args.batch_size} data points/checkpoint, single process.\n")

    rows: list[dict] = []
    baseline = _snapshot()
    print(f"[baseline] rss={baseline['rss_mb']:.1f}MB traced={baseline['traced_mb']:.1f}MB")

    try:
        for i in range(args.checkpoints):
            astar_calls_before = _astar_stats["calls"]
            astar_caps_before = _astar_stats["cap_hits"]
            env_attempts_before = _env_construction_stats["attempts"]
            thor_misses_before = _thor_cache_stats["misses"]

            t0 = time.perf_counter()
            datagen._generate_worker_share(
                num_objects,
                num_locations,
                task_distribution,
                WORKER_ID,
                args.batch_size,
                WORKER_ID * datagen.SEED_STRIDE + i * CHECKPOINT_SEED_STRIDE,
            )
            elapsed = time.perf_counter() - t0

            snap = _snapshot()
            row = {
                "checkpoint": i + 1,
                "elapsed_s": elapsed,
                "astar_calls": _astar_stats["calls"] - astar_calls_before,
                "astar_cap_hits": _astar_stats["cap_hits"] - astar_caps_before,
                "env_rejects": (_env_construction_stats["attempts"] - env_attempts_before) - args.batch_size,
                "thor_cache_misses": _thor_cache_stats["misses"] - thor_misses_before,
                **snap,
            }
            rows.append(row)
            print(
                f"[{i + 1:>3}/{args.checkpoints}] "
                f"rss={snap['rss_mb']:7.1f}MB "
                f"({snap['rss_mb'] - baseline['rss_mb']:+7.1f}MB) "
                f"traced={snap['traced_mb']:6.1f}MB "
                f"astar={row['astar_calls']:4d} calls / {row['astar_cap_hits']:3d} cap-hits  "
                f"env_rejects={row['env_rejects']:3d} thor_misses={row['thor_cache_misses']:3d}  "
                f"live[env={snap['n_env']} thor={snap['n_thor']} sg={snap['n_scenegraph']} "
                f"traj={snap['n_traj']} expdata={snap['n_expdata']}]  "
                f"({elapsed:.1f}s)"
            )
    except KeyboardInterrupt:
        print("\nInterrupted — printing report for the checkpoints completed so far.")

    if len(rows) < 2:
        print("Not enough checkpoints completed to report a trend.")
        return

    first, last = rows[0], rows[-1]
    print("\n--- Summary (first checkpoint -> last checkpoint) ---")
    print(f"RSS:              {first['rss_mb']:.1f}MB -> {last['rss_mb']:.1f}MB "
          f"(delta {last['rss_mb'] - first['rss_mb']:+.1f}MB)")
    print(f"Traced Python mem:{first['traced_mb']:.1f}MB -> {last['traced_mb']:.1f}MB "
          f"(delta {last['traced_mb'] - first['traced_mb']:+.1f}MB)")
    print(f"Live envs:        {first['n_env']} -> {last['n_env']}")
    print(f"Live ThorInterface:{first['n_thor']} -> {last['n_thor']}")
    print(f"Live SceneGraphs: {first['n_scenegraph']} -> {last['n_scenegraph']}")
    print(f"Live trajectories:{first['n_traj']} -> {last['n_traj']}")
    print(f"Live ExperimentData:{first['n_expdata']} -> {last['n_expdata']}")
    total_astar = sum(r["astar_calls"] for r in rows)
    total_cap_hits = sum(r["astar_cap_hits"] for r in rows)
    total_rejects = sum(r["env_rejects"] for r in rows)
    total_thor_misses = sum(r["thor_cache_misses"] for r in rows)
    print(f"\nTotal astar_search calls: {total_astar}  "
          f"({total_cap_hits} hit num_steps without finding a goal, "
          f"{100 * total_cap_hits / total_astar:.1f}%)")
    print(f"Total rejected env candidates: {total_rejects}  "
          f"({total_thor_misses} required spinning up a fresh AI2-THOR Controller)")

    print("\n--- How to read this ---")
    print("- RSS climbing while 'Traced Python mem' stays flat/bounded: no Python-level")
    print("  reference leak — it's native allocator fragmentation from astar_search's")
    print("  large transient allocations (InterruptionTrajectory.create_child copying")
    print("  state_history/plan/interruption_probs on every frontier node), consistent")
    print("  with a high cap-hit rate above.")
    print("- RSS climbing in lockstep with 'Traced Python mem', or live instance counts")
    print("  (esp. n_thor/n_scenegraph/n_expdata) trending up instead of staying roughly")
    print("  flat: a genuine reference leak — something is holding onto discarded")
    print("  ExperimentData/environments across checkpoints.")
    print("- High env_rejects/thor_cache_misses relative to batch size: the retry loop")
    print("  in get_randomized_procthor_data is churning through AI2-THOR Controller")
    print("  spin-up/teardown cycles more than expected.")


if __name__ == "__main__":
    main()
