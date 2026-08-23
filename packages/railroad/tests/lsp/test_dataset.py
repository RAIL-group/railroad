"""Tests for the torch dataset over LSP training data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from railroad.lsp import LSPFrontierDataset, TrainingDataWriter, TrainingDatum

torch = pytest.importorskip("torch")


def _datum(label: bool = True) -> TrainingDatum:
    return TrainingDatum(
        image=np.full((4, 16, 3), 128, dtype=np.uint8),
        frontier_xy_ego=(3.0, -1.0),
        goal_xy_ego=(10.0, 4.0),
        label=label,
        success_cost=12.5 if label else None,
        optimistic_cost=10.0 if label else None,
        exploration_cost=None if label else 7.5,
        metadata={"frontier_id": "f1"},
    )


@pytest.fixture()
def experiment_dir(tmp_path: Path) -> Path:
    exp_dir = tmp_path / "exp"
    for seed, labels in [(0, [True, False]), (1, [True])]:
        with TrainingDataWriter(
            exp_dir / f"seed_{seed:05d}", {"seed": seed}
        ) as writer:
            for label in labels:
                writer.write(_datum(label=label))
    # In-flight data must not be picked up.
    with TrainingDataWriter(exp_dir / ".tmp" / "seed_00002", {}) as writer:
        writer.write(_datum())
    return exp_dir


def test_dataset_over_run_that_emitted_nothing(tmp_path: Path) -> None:
    # A seed that never saw a labeled frontier is a length-0 dataset, not a
    # "contains neither an index.jsonl nor seed_* directories" error.
    out_dir = tmp_path / "data"
    with TrainingDataWriter(out_dir, {"seed": 0}):
        pass

    assert len(LSPFrontierDataset(out_dir)) == 0


def test_dataset_spans_seed_dirs(experiment_dir: Path) -> None:
    dataset = LSPFrontierDataset(experiment_dir)
    assert len(dataset) == 3

    item = dataset[0]
    assert item["image"].dtype == torch.float32
    assert item["image"].shape == (3, 4, 16)
    assert item["image"].min().item() >= 0.0
    assert item["image"].max().item() <= 1.0
    assert torch.allclose(
        item["frontier_xy_ego"], torch.tensor([3.0, -1.0])
    )
    assert item["label"].item() == 1.0
    assert item["success_cost"].item() == pytest.approx(12.5)
    assert torch.isnan(item["exploration_cost"])

    infeasible = dataset[1]
    assert infeasible["label"].item() == 0.0
    assert torch.isnan(infeasible["success_cost"])
    assert infeasible["exploration_cost"].item() == pytest.approx(7.5)


def test_dataset_on_single_writer_dir(experiment_dir: Path) -> None:
    dataset = LSPFrontierDataset(experiment_dir / "seed_00000")
    assert len(dataset) == 2


def test_dataset_explicit_dirs(experiment_dir: Path) -> None:
    dataset = LSPFrontierDataset(dirs=[experiment_dir / "seed_00001"])
    assert len(dataset) == 1


def test_dataset_argument_validation(experiment_dir: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        LSPFrontierDataset()
    with pytest.raises(ValueError, match="exactly one"):
        LSPFrontierDataset(experiment_dir, dirs=[experiment_dir])
    with pytest.raises(ValueError, match="not a directory"):
        LSPFrontierDataset(experiment_dir / "nothing-here")
    empty = experiment_dir / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="neither"):
        LSPFrontierDataset(empty)


def test_dataset_works_with_dataloader(experiment_dir: Path) -> None:
    from torch.utils.data import DataLoader

    dataset = LSPFrontierDataset(experiment_dir)
    # The dataset is deliberately duck-typed, not a Dataset subclass.
    loader = DataLoader(dataset, batch_size=3)  # ty: ignore[invalid-argument-type]
    (batch,) = list(loader)
    assert batch["image"].shape == (3, 3, 4, 16)
    assert batch["label"].shape == (3,)
