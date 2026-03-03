"""
Evaluate procthor_planning.py results: scatter plots of learned vs optimistic plan cost.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import scipy.stats


def parse_cost_file(path: Path) -> dict:
    data = {}
    with open(path) as f:
        for line in f:
            key, value = line.strip().split(": ", 1)
            data[key] = value
    return data


def load_results(results_dir: Path, seeds: list[int], num_robots: int) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        opt_file = results_dir / f"cost_procthor_optimistic_num_robots_{num_robots}_{seed}.txt"
        if not opt_file.exists():
            raise FileNotFoundError(f"Missing file {opt_file}")
        learned_file = results_dir / f"cost_procthor_learned_num_robots_{num_robots}_{seed}.txt"
        if not learned_file.exists():
            raise FileNotFoundError(f"Missing file {learned_file}")

        opt = parse_cost_file(opt_file)
        learned = parse_cost_file(learned_file)

        if opt["success"] != "True" or learned["success"] != "True":
            continue

        rows.append({
            "seed": seed,
            "baseline_cost": float(opt["plan_cost"]),
            "learned_cost": float(learned["plan_cost"]),
        })

    return pd.DataFrame(rows)


def build_plot(ax, data: pd.DataFrame, title_prefix: str, cmap="Blues"):
    xy = np.vstack([data["baseline_cost"], data["learned_cost"]])
    z = scipy.stats.gaussian_kde(xy)(xy)

    data = data.copy()
    data["zs"] = z
    data = data.sort_values(by=["zs"])
    z = data["zs"]
    colors = matplotlib.colormaps[cmap]((z - z.min()) / (z.max() - z.min()) * 0.75 + 0.25)

    ax.scatter(data["baseline_cost"], data["learned_cost"], c=colors)
    ax.set_aspect("equal", adjustable="box")
    cb = 1.05 * max(data["baseline_cost"].max(), data["learned_cost"].max())
    ax.plot([0, cb], [0, cb], "k", alpha=0.3)
    ax.set_xlim([0, cb])
    ax.set_ylim([0, cb])
    ax.set_xlabel("Optimistic Baseline Cost")
    ax.set_ylabel("Learned Planner Cost")

    cost_mean_base = data["baseline_cost"].mean()
    cost_mean_learn = data["learned_cost"].mean()
    improvement = (cost_mean_base - cost_mean_learn) / cost_mean_base
    title_string = (
        f"{title_prefix}\n"
        f"Optimistic Baseline Cost: {cost_mean_base:.2f}\n"
        f"Learned Cost: {cost_mean_learn:.2f}\n"
        f"Improvement %: {100 * improvement:.2f} | {data['baseline_cost'].size} seeds"
    )
    ax.set_title(title_string)
    print(title_string)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ProcTHOR planning results")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--seed_start", type=int, default=1000)
    parser.add_argument("--seed_end", type=int, default=1050)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    seeds = list(range(args.seed_start, args.seed_end + 1))

    data_1 = load_results(results_dir, seeds, num_robots=1)
    data_2 = load_results(results_dir, seeds, num_robots=2)

    fig = plt.figure(dpi=200, figsize=(14, 6))
    spec = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.3)

    if not data_1.empty:
        ax1 = fig.add_subplot(spec[0])
        build_plot(ax1, data_1, "1 Robot", cmap="Blues")
    else:
        print("No valid 1-robot results found.")

    if not data_2.empty:
        ax2 = fig.add_subplot(spec[1])
        build_plot(ax2, data_2, "2 Robots", cmap="Oranges")
    else:
        print("No valid 2-robot results found.")

    plt.tight_layout()
    plt.savefig(results_dir / "eval_procthor.png", dpi=300)
    print(f"\nSaved plot to {results_dir / 'eval_procthor.png'}")


if __name__ == "__main__":
    main()
