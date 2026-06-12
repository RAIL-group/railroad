"""Tests for the LSP network training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from railroad.lsp import TrainingDataWriter, TrainingDatum

torch = pytest.importorskip("torch")

from railroad.lsp.train import (  # noqa: E402
    TrainConfig,
    split_seed_dirs,
    train_network,
)


def _datum(rng: np.random.Generator, label: bool) -> TrainingDatum:
    return TrainingDatum(
        image=rng.integers(0, 255, size=(64, 128, 3), dtype=np.uint8),
        frontier_xy_ego=(float(rng.normal()), float(rng.normal())),
        goal_xy_ego=(float(rng.normal()), float(rng.normal())),
        label=label,
        success_cost=50.0 if label else None,
        optimistic_cost=40.0 if label else None,
        exploration_cost=None if label else 20.0,
    )


@pytest.fixture()
def experiment_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    exp_dir = tmp_path / "exp"
    for seed in range(4):
        with TrainingDataWriter(
            exp_dir / f"seed_{seed:05d}", {"seed": seed}
        ) as writer:
            for i in range(3):
                writer.write(_datum(rng, label=(i + seed) % 2 == 0))
    return exp_dir


def test_split_seed_dirs(experiment_dir: Path) -> None:
    train_dirs, val_dirs = split_seed_dirs(experiment_dir, 0.25, seed=0)
    assert len(train_dirs) == 3
    assert len(val_dirs) == 1
    assert not set(train_dirs) & set(val_dirs)
    # The same seed reproduces the same split.
    assert split_seed_dirs(experiment_dir, 0.25, seed=0) == (
        train_dirs, val_dirs
    )


def test_split_single_dir_skips_validation(experiment_dir: Path) -> None:
    train_dirs, val_dirs = split_seed_dirs(
        experiment_dir / "seed_00000", 0.25, seed=0
    )
    assert len(train_dirs) == 1
    assert val_dirs == []


def test_train_network(experiment_dir: Path, tmp_path: Path) -> None:
    save_dir = tmp_path / "training"
    result = train_network(TrainConfig(
        data_dir=experiment_dir,
        save_dir=save_dir,
        num_epochs=2,
        batch_size=4,
        val_fraction=0.25,
        num_workers=0,
        device="cpu",
    ))

    assert result.network_file == save_dir / "LSPFrontierNet.pt"
    assert result.network_file.exists()
    assert len(result.history) == 2
    assert all(
        np.isfinite(entry["train"]["total"]) and np.isfinite(entry["val"]["total"])
        for entry in result.history
    )
    assert (save_dir / "loss_curves.png").exists()

    log = json.loads((save_dir / "training_log.json").read_text())
    assert log["config"]["num_epochs"] == 2
    assert len(log["train_dirs"]) == 3
    assert len(log["val_dirs"]) == 1

    # The saved weights round-trip into a usable planning-time model.
    from railroad.lsp.model import load_frontier_statistics_model

    model = load_frontier_statistics_model(result.network_file, device="cpu")
    from railroad.lsp import FrontierObservation

    (stats,) = model([FrontierObservation(
        image=np.zeros((64, 128, 3), dtype=np.uint8),
        frontier_xy_ego=(1.0, 2.0),
        goal_xy_ego=(3.0, 4.0),
    )])
    assert 0.0 <= stats.prob_feasible <= 1.0
