"""Tests for LSP training-data inspection (GL-free, synthetic data)."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from railroad.lsp import TrainingDataWriter, TrainingDatum, inspect_data
from railroad.lsp.inspect import (
    _select_indices,
    bearing_to_column,
    make_inspection_figure,
    summarize,
)


def _write_sample_data(out_dir: Path, num: int = 4) -> Path:
    writer = TrainingDataWriter(out_dir, {"env": "maze", "seed": 7})
    for i in range(num):
        feasible = i % 2 == 0
        writer.write(TrainingDatum(
            image=np.full((8, 32, 3), 40 * i, dtype=np.uint8),
            frontier_xy_ego=(5.0 + i, 0.0),
            goal_xy_ego=(10.0, -3.0 - i),
            label=feasible,
            success_cost=20.0 + i if feasible else None,
            optimistic_cost=15.0 + i if feasible else None,
            exploration_cost=None if feasible else 8.0 + i,
            metadata={"frontier_id": f"frontier_{i}", "time": float(i)},
        ))
    writer.close()
    return out_dir


def test_bearing_to_column_conventions() -> None:
    width = 32
    # Bearing 0 (straight ahead) is the image center.
    assert bearing_to_column(0.0, width) == width / 2
    # Positive bearing (left) maps left of center; negative maps right.
    assert bearing_to_column(math.pi / 2, width) == width / 4
    assert bearing_to_column(-math.pi / 2, width) == 3 * width / 4
    # Behind the robot wraps to the image edge.
    assert bearing_to_column(math.pi, width) % width == 0.0


def test_select_indices() -> None:
    # Explicit indices pass through; out-of-range raises.
    assert _select_indices(10, 3, [0, 4]) == [0, 4]
    with pytest.raises(IndexError):
        _select_indices(3, 2, [5])
    # Fewer data than requested: everything.
    assert _select_indices(2, 6, None) == [0, 1]
    # Sampling spans the full run, first and last included.
    sampled = _select_indices(100, 4, None)
    assert sampled[0] == 0 and sampled[-1] == 99
    assert sampled == sorted(set(sampled))


def test_summarize_reports_counts_and_costs(tmp_path: Path) -> None:
    data_dir = _write_sample_data(tmp_path / "data")
    text = summarize(data_dir)
    assert "4 data" in text
    assert "2 feasible / 2 infeasible" in text
    assert "env=maze" in text
    assert "success_cost" in text and "exploration_cost" in text


def test_summarize_empty_dir(tmp_path: Path) -> None:
    text = summarize(tmp_path)
    assert "no data found" in text


def test_make_inspection_figure(tmp_path: Path) -> None:
    data_dir = _write_sample_data(tmp_path / "data")
    fig = make_inspection_figure(data_dir, [0, 3])
    # One (pano, ego) row per requested datum.
    assert len(fig.axes) == 4
    plt.close(fig)


def test_inspect_data_writes_figure(tmp_path: Path) -> None:
    data_dir = _write_sample_data(tmp_path / "data")
    out = inspect_data(data_dir, num=3)
    assert out is not None
    assert out == data_dir / "inspect.png"
    assert out.exists() and out.stat().st_size > 0

    custom = tmp_path / "figure.png"
    assert inspect_data(data_dir, indices=[1], save_path=custom) == custom
    assert custom.exists()


def test_inspect_data_empty_dir_returns_none(tmp_path: Path) -> None:
    assert inspect_data(tmp_path) is None
