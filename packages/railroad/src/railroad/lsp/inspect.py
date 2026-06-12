"""Inspection utilities for generated LSP training data.

Summarizes a training-data directory (written by
:class:`~railroad.lsp.data.TrainingDataWriter`) and renders a figure of
sampled data: each row shows the frontier-centered panorama with the
frontier (image center) and goal directions marked, alongside a
top-down egocentric view of the robot, frontier, and goal.

CLI: ``railroad lsp inspect-data <data_dir>``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence

import numpy as np

from .data import load_datum, read_index
from .types import TrainingDatum

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def bearing_to_column(bearing: float, width: int) -> float:
    """Pano column of a bearing (rad, +left) in an image-centered frame."""
    return (width / 2.0 - bearing * width / (2.0 * math.pi)) % width


def _ego_bearing(xy: tuple[float, float]) -> float:
    """Bearing of an egocentric (x forward, y left) point, + to the left."""
    return math.atan2(xy[1], xy[0])


def _cost_summary(datum: TrainingDatum) -> str:
    if datum.label:
        return (
            f"success: cost={datum.success_cost:.1f}, "
            f"optimistic={datum.optimistic_cost:.1f}"
            if datum.success_cost is not None
            and datum.optimistic_cost is not None
            else "success"
        )
    if datum.exploration_cost is not None:
        return f"failure: exploration cost={datum.exploration_cost:.1f}"
    return "failure"


def plot_datum(
    datum: TrainingDatum,
    ax_pano: "Axes",
    ax_ego: "Axes",
    title: str | None = None,
) -> None:
    """Draw one datum: annotated panorama + top-down egocentric view."""
    image = datum.image
    height, width = image.shape[:2]
    label_color = "tab:green" if datum.label else "tab:red"

    # Panorama: frontier is image-centered by construction; mark the
    # goal direction at its bearing column.
    ax_pano.imshow(image)
    ax_pano.axvline(width / 2.0, color=label_color, lw=2.0)
    goal_col = bearing_to_column(_ego_bearing(datum.goal_xy_ego), width)
    ax_pano.axvline(goal_col, color="tab:orange", lw=2.0, ls="--")
    ax_pano.text(
        width / 2.0 + 4, 4, "frontier", color=label_color,
        fontsize=8, va="top",
    )
    ax_pano.text(
        goal_col + 4, height - 4, "goal", color="tab:orange",
        fontsize=8, va="bottom",
    )
    ax_pano.set_xticks([])
    ax_pano.set_yticks([])
    if title is not None:
        ax_pano.set_title(title, fontsize=9, loc="left")

    # Top-down egocentric view: x forward is up, y left is plot-left.
    fx, fy = datum.frontier_xy_ego
    gx, gy = datum.goal_xy_ego
    ax_ego.plot(0, 0, "k^", ms=9, label="robot")
    ax_ego.plot(-fy, fx, "o", color=label_color, ms=8, label="frontier")
    ax_ego.plot(-gy, gx, "*", color="tab:orange", ms=12, label="goal")
    ax_ego.plot([0, -fy], [0, fx], color=label_color, lw=1.0, alpha=0.6)
    ax_ego.plot([0, -gy], [0, gx], color="tab:orange", lw=1.0,
                ls="--", alpha=0.6)
    span = max(abs(v) for v in (fx, fy, gx, gy, 1.0)) * 1.2
    ax_ego.set_xlim(-span, span)
    ax_ego.set_ylim(-span, span)
    ax_ego.set_aspect("equal")
    ax_ego.tick_params(labelsize=7)
    ax_ego.set_xlabel("right (cells)", fontsize=7)
    ax_ego.set_ylabel("forward (cells)", fontsize=7)
    ax_ego.grid(alpha=0.3)


def make_inspection_figure(
    data_dir: str | Path,
    indices: Sequence[int],
) -> "Figure":
    """Figure with one annotated row per datum index in *data_dir*."""
    import matplotlib.pyplot as plt

    data_dir = Path(data_dir)
    index = read_index(data_dir)
    nrows = len(indices)
    fig, axes = plt.subplots(
        nrows, 2,
        figsize=(13, 2.6 * nrows),
        squeeze=False,
        gridspec_kw={"width_ratios": [4, 1]},
    )
    for row, i in enumerate(indices):
        entry = index[i]
        datum = load_datum(data_dir / entry["file"])
        title = (
            f"[{i}] {entry['file']}  |  {entry.get('frontier_id', '?')}  |  "
            f"label={'FEASIBLE' if datum.label else 'infeasible'}  |  "
            f"{_cost_summary(datum)}  |  t={entry.get('time', float('nan')):.1f}s"
        )
        plot_datum(datum, axes[row][0], axes[row][1], title=title)
    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def _select_indices(
    num_data: int, num: int, indices: Sequence[int] | None
) -> List[int]:
    if indices is not None:
        bad = [i for i in indices if not 0 <= i < num_data]
        if bad:
            raise IndexError(
                f"datum indices {bad} out of range (have {num_data} data)"
            )
        return list(indices)
    if num >= num_data:
        return list(range(num_data))
    # Evenly spaced through the run so early and late frontiers show up.
    return sorted({round(i * (num_data - 1) / (num - 1)) for i in range(num)})


def summarize(data_dir: str | Path) -> str:
    """Human-readable summary of a training-data directory."""
    data_dir = Path(data_dir)
    index = read_index(data_dir)
    lines = [f"LSP training data: {data_dir}"]

    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta:
            lines.append(
                "  run: " + ", ".join(f"{k}={v}" for k, v in meta.items())
            )

    if not index:
        lines.append("  no data found (empty or missing index.jsonl)")
        return "\n".join(lines)

    num_positive = sum(1 for e in index if e.get("label"))
    frontier_ids = {e.get("frontier_id") for e in index}
    times = [e["time"] for e in index if isinstance(e.get("time"), (int, float))]
    lines.append(
        f"  {len(index)} data from {len(frontier_ids)} frontiers: "
        f"{num_positive} feasible / {len(index) - num_positive} infeasible"
    )
    if times:
        lines.append(f"  capture times: t={min(times):.1f}s .. t={max(times):.1f}s")

    # Cost statistics come from the npz files (NaN encodes None); npz
    # loading is lazy per-array, so the images are never decompressed.
    costs: Dict[str, List[float]] = {
        "success_cost": [], "optimistic_cost": [], "exploration_cost": []
    }
    for entry in index:
        with np.load(data_dir / entry["file"]) as data:
            for key, values in costs.items():
                value = float(data[key])
                if not math.isnan(value):
                    values.append(value)
    for key, values in costs.items():
        if values:
            lines.append(
                f"  {key}: min={min(values):.1f} "
                f"median={float(np.median(values)):.1f} max={max(values):.1f}"
            )
    return "\n".join(lines)


def inspect_data(
    data_dir: str | Path,
    *,
    num: int = 6,
    indices: Sequence[int] | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """Print a summary and render sampled data; returns the figure path."""
    data_dir = Path(data_dir)
    print(summarize(data_dir))

    index = read_index(data_dir)
    if not index:
        return None
    selected = _select_indices(len(index), num, indices)

    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = make_inspection_figure(data_dir, selected)
    out_path: Path | None = None
    if save_path is not None or not show:
        out_path = Path(save_path) if save_path is not None else (
            data_dir / "inspect.png"
        )
        fig.savefig(out_path, dpi=150)
        print(f"Saved inspection figure ({len(selected)} data) to {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path
