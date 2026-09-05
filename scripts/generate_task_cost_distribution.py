"""
Build a cumulative distribution function (CDF) over the costs of completing the
interrupting tasks used in the expected-value targets of the training dataset.

Input data lives in ``<procthor-10k>/pickles/task_costs/`` -- one compressed
pickle per (scene, object-placement, plan-step, task) tuple, written by
``scripts/multiprocess_datagen.py`` when ``WRITE_OUT_INDIVIDUAL_TASK_COSTS`` is
set. Each pickle is ``(SceneGraph, LiteralGoal task, float cost)`` where ``cost``
is the A* plan cost to complete that single task from the datum's state.

Usage:
    uv run python scripts/generate_task_cost_distribution.py
    uv run python scripts/generate_task_cost_distribution.py --group-by object
    uv run python scripts/generate_task_cost_distribution.py \
        --data-dir resources/procthor-10k/pickles/task_costs \
        --out-image task_cost_cdf.jpeg --out-data task_cost_cdf.npz
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from interruption.learning.data import load_compressed_pickle
from railroad.environment.procthor.resources import get_procthor_10k_dir
from railroad.environment.procthor.utils import get_generic_name

DEFAULT_DATA_DIR = get_procthor_10k_dir() / "pickles" / "task_costs"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TaskCostRecord:
    """One (task, cost) observation, plus the provenance parsed from the filename.

    Filename convention (see ``_task_datum_pickle_path`` in multiprocess_datagen):
        dat_{scene_seed}_{object_randomization_seed}_{counter}_{task_idx}.pgz
    """

    scene_seed: int
    object_seed: int
    counter: int
    task_idx: int
    task: str          # e.g. "(at apple fridge)"
    cost: float

    @property
    def task_object(self) -> str:
        """Generic name of the object the task references, best-effort.

        ``(at apple fridge)`` -> ``apple``. Falls back to the raw task string
        for task shapes this simple parser does not recognise.
        """
        tokens = self.task.strip("() ").split()
        if len(tokens) >= 2 and tokens[0] == "at":
            return get_generic_name(tokens[1])
        return self.task

    @property
    def task_predicate(self) -> str:
        """First token of the goal literal, e.g. ``at``."""
        return self.task.strip("() ").split()[0] if self.task.strip("() ") else self.task


def _parse_provenance(path: Path) -> tuple[int, int, int, int]:
    """``dat_201_100001_1_0.pgz`` -> (201, 100001, 1, 0)."""
    scene_seed, object_seed, counter, task_idx = path.stem.split("_")[1:]
    return int(scene_seed), int(object_seed), int(counter), int(task_idx)


def load_task_costs(
    data_dir: Path,
    scene_seed: int | None = None,
    limit: int | None = None,
) -> Iterator[TaskCostRecord]:
    """Yield a ``TaskCostRecord`` per pickle under ``data_dir``.

    ``scene_seed`` filters by the leading seed in the filename (cheap, no unpickle).
    ``limit`` caps how many records are yielded (handy while iterating).
    """
    paths = sorted(data_dir.glob("dat_*.pgz"))
    if scene_seed is not None:
        paths = [p for p in paths if p.stem.split("_")[1] == str(scene_seed)]
    if limit is not None:
        paths = paths[:limit]

    for path in tqdm(paths, desc="loading task-cost pickles", unit="file"):
        s_seed, o_seed, counter, task_idx = _parse_provenance(path)
        _scene_graph, task, cost = load_compressed_pickle(path)
        yield TaskCostRecord(
            scene_seed=s_seed,
            object_seed=o_seed,
            counter=counter,
            task_idx=task_idx,
            task=str(task),
            cost=float(cost),
        )


# --------------------------------------------------------------------------- #
# CDF
# --------------------------------------------------------------------------- #
def empirical_cdf(costs: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, F)`` for the empirical CDF: ``F[i] = P(cost <= x[i])``.

    ``x`` is the sorted sample; ``F`` is ``(1..n)/n``. Suitable for a step plot.
    """
    x = np.sort(np.asarray(costs, dtype=float))
    if x.size == 0:
        return x, x
    f = np.arange(1, x.size + 1) / x.size
    return x, f


def summarize(costs: Sequence[float]) -> dict[str, float]:
    """Headline stats for a set of costs; also what gets printed to stdout."""
    arr = np.asarray(costs, dtype=float)
    if arr.size == 0:
        return {}
    pctiles = [5, 25, 50, 75, 90, 95, 99]
    stats = {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    stats.update({f"p{p}": float(np.percentile(arr, p)) for p in pctiles})
    return stats


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_cdf(
    groups: dict[str, list[float]],
    out_image: Path,
    title: str = "Task-completion cost CDF",
) -> None:
    """Step-plot one CDF per group on shared axes and save to ``out_image``."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for label, costs in sorted(groups.items()):
        x, f = empirical_cdf(costs)
        if x.size == 0:
            continue
        ax.step(x, f, where="post", label=f"{label} (n={len(costs)})", linewidth=1.5)

    # median reference line for the pooled distribution
    pooled = [c for costs in groups.values() for c in costs]
    if pooled:
        ax.axvline(float(np.median(pooled)), color="gray", linestyle="--",
                   linewidth=1, label=f"pooled median = {np.median(pooled):.1f}")

    ax.set_xlabel("task-completion cost (A* plan cost)")
    ax.set_ylabel("F(cost) = P(cost ≤ x)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    if len(groups) > 1 or pooled:
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_image, dpi=150)
    print(f"wrote {out_image}")


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #
def group_records(records: list[TaskCostRecord], group_by: str) -> dict[str, list[float]]:
    """Bucket record costs by the chosen key."""
    if group_by == "none":
        return {"all tasks": [r.cost for r in records]}

    key_fn = {
        "object": lambda r: r.task_object,
        "task": lambda r: r.task,
        "predicate": lambda r: r.task_predicate,
        "scene": lambda r: f"scene {r.scene_seed}",
    }[group_by]

    buckets: dict[str, list[float]] = defaultdict(list)
    for r in records:
        buckets[key_fn(r)].append(r.cost)
    return dict(buckets)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help=f"directory of dat_*.pgz task-cost pickles (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--scene-seed", type=int, default=None,
                   help="only include pickles for this ProcTHOR scene seed")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of pickles loaded (debugging)")
    p.add_argument("--group-by", choices=["none", "object", "task", "predicate", "scene"],
                   default="none", help="draw one CDF curve per group")
    p.add_argument("--drop-zero", action="store_true",
                   help="exclude cost==0 observations (tasks already satisfied in that datum's state)")
    p.add_argument("--out-image", type=Path, default=Path("task_cost_cdf.jpeg"))
    p.add_argument("--out-data", type=Path, default=None,
                   help="optional .npz to dump raw costs + provenance for reuse")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    records = list(load_task_costs(args.data_dir, args.scene_seed, args.limit))
    if not records:
        raise SystemExit(f"no task-cost pickles found under {args.data_dir}")

    if args.drop_zero:
        n_before = len(records)
        records = [r for r in records if r.cost > 0.0]
        print(f"dropped {n_before - len(records)} zero-cost observations "
              f"(task already satisfied); {len(records)} remain")
        if not records:
            raise SystemExit("every observation had cost 0 -- nothing to plot")

    costs = [r.cost for r in records]
    stats = summarize(costs)
    print(f"\n{len(records)} task-cost observations")
    for k, v in stats.items():
        print(f"  {k:>6}: {v:.3f}")

    groups = group_records(records, args.group_by)
    subtitle = f"group-by={args.group_by}" + (", zero-cost excluded" if args.drop_zero else "")
    plot_cdf(groups, args.out_image, title=f"Task-completion cost CDF ({subtitle})")

    if args.out_data is not None:
        np.savez(
            args.out_data,
            cost=np.array([r.cost for r in records]),
            scene_seed=np.array([r.scene_seed for r in records]),
            object_seed=np.array([r.object_seed for r in records]),
            counter=np.array([r.counter for r in records]),
            task_idx=np.array([r.task_idx for r in records]),
            task=np.array([r.task for r in records]),
        )
        print(f"wrote {args.out_data}")

    # TODO(research): decide how this CDF feeds the expected-value targets --
    # e.g. reweight `interrupting_task_dist` probabilities by cost, or normalise
    # the regression targets against a pooled cost quantile.


if __name__ == "__main__":
    main()
