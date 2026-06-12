"""Training pipeline for :class:`~railroad.lsp.model.LSPFrontierNet`.

Trains on the data ``railroad lsp generate-data`` produces (exposed via
``railroad lsp train-network``). Validation is split at the *seed*
level — whole rollout directories are held out — because data from one
rollout are highly correlated (the same frontiers re-emitted as their
labels or vantages change).

Like :mod:`railroad.lsp.model`, this module requires torch and must be
imported explicitly.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from .dataset import LSPFrontierDataset, _discover_dirs
from .model import NETWORK_FILENAME, LSPFrontierNet, default_device

LOSS_KEYS = ("total", "feasibility", "delta_success_cost", "exploration_cost")


@dataclass
class TrainConfig:
    """Knobs for ``train_network``."""

    data_dir: str | Path
    save_dir: str | Path
    # Filename of the saved weights inside save_dir.
    network_filename: str = NETWORK_FILENAME
    num_epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 2e-3
    learning_rate_decay: float = 0.6
    relative_positive_weight: float = 2.0
    cost_scale: float = 100.0
    val_fraction: float = 0.1
    num_workers: int = 4
    device: str | None = None
    seed: int = 0
    # Progress is printed every this many training batches (0 disables).
    log_interval: int = 25


@dataclass
class TrainResult:
    """What a training run produced."""

    network_file: Path
    # One entry per epoch: {"train": {loss key: mean}, "val": {... or {}}}
    history: List[Dict[str, Dict[str, float]]] = field(default_factory=list)


def split_seed_dirs(
    data_dir: str | Path, val_fraction: float, seed: int
) -> Tuple[List[Path], List[Path]]:
    """Split the data directories under *data_dir* into train and val.

    With fewer than two directories (a single run's data) there is
    nothing to hold out and validation is skipped.
    """
    dirs = _discover_dirs(Path(data_dir))
    if len(dirs) < 2 or val_fraction <= 0.0:
        return dirs, []
    shuffled = list(dirs)
    random.Random(seed).shuffle(shuffled)
    num_val = min(max(1, round(val_fraction * len(shuffled))), len(shuffled) - 1)
    return sorted(shuffled[num_val:]), sorted(shuffled[:num_val])


def _run_epoch(
    network: LSPFrontierNet,
    loader: "torch.utils.data.DataLoader",
    config: TrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    progress_prefix: str = "",
) -> Dict[str, float]:
    """One pass over *loader*; trains when *optimizer* is given.

    While training, a progress line (losses averaged over the interval,
    plus throughput) is printed every ``config.log_interval`` batches.
    """
    network.train(optimizer is not None)
    sums = dict.fromkeys(LOSS_KEYS, 0.0)
    interval_sums = dict.fromkeys(LOSS_KEYS, 0.0)
    num_batches = 0
    interval_data = 0
    interval_start = time.perf_counter()
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            output = network(
                batch["image"].to(device),
                batch["frontier_xy_ego"].to(device),
                batch["goal_xy_ego"].to(device),
            )
            losses = network.loss(
                output,
                batch,
                relative_positive_weight=config.relative_positive_weight,
                cost_scale=config.cost_scale,
            )
            if optimizer is not None:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()
            for key in LOSS_KEYS:
                sums[key] += losses[key].item()
                interval_sums[key] += losses[key].item()
            num_batches += 1
            interval_data += len(batch["label"])

            if (optimizer is not None and config.log_interval
                    and num_batches % config.log_interval == 0):
                elapsed = time.perf_counter() - interval_start
                means = {
                    key: value / config.log_interval
                    for key, value in interval_sums.items()
                }
                print(
                    f"  {progress_prefix}batch {num_batches}/{len(loader)}  "
                    f"{_format_losses(means)}  "
                    f"[{interval_data / max(elapsed, 1e-9):.1f} data/s]",
                    flush=True,
                )
                interval_sums = dict.fromkeys(LOSS_KEYS, 0.0)
                interval_data = 0
                interval_start = time.perf_counter()
    return {key: value / max(num_batches, 1) for key, value in sums.items()}


def _format_losses(losses: Dict[str, float]) -> str:
    return (
        f"total={losses['total']:.4f} "
        f"(feas={losses['feasibility']:.4f} "
        f"delta={losses['delta_success_cost']:.4f} "
        f"expl={losses['exploration_cost']:.4f})"
    )


def _save_loss_curves(
    history: List[Dict[str, Dict[str, float]]], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history) + 1))
    fig, axes = plt.subplots(1, len(LOSS_KEYS), figsize=(4 * len(LOSS_KEYS), 3))
    for ax, key in zip(axes, LOSS_KEYS):
        ax.plot(epochs, [entry["train"][key] for entry in history],
                label="train")
        if history[0]["val"]:
            ax.plot(epochs, [entry["val"][key] for entry in history],
                    label="val")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _make_loader(
    dataset: LSPFrontierDataset, config: TrainConfig, *, shuffle: bool
) -> "torch.utils.data.DataLoader":
    # The dataset is deliberately duck-typed, not a Dataset subclass.
    return torch.utils.data.DataLoader(
        dataset,  # ty: ignore[invalid-argument-type]
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
    )


def train_network(config: TrainConfig) -> TrainResult:
    """Train an :class:`LSPFrontierNet` and save it under ``save_dir``.

    Writes the weights (``config.network_filename``, by default
    ``LSPFrontierNet.pt``), a ``training_log.json`` with the config and
    per-epoch losses, and a ``loss_curves.png``.
    """
    device = (
        default_device() if config.device is None
        else torch.device(config.device)
    )
    torch.manual_seed(config.seed)

    train_dirs, val_dirs = split_seed_dirs(
        config.data_dir, config.val_fraction, config.seed
    )
    train_dataset = LSPFrontierDataset(dirs=train_dirs)
    val_dataset = LSPFrontierDataset(dirs=val_dirs) if val_dirs else None
    train_loader = _make_loader(train_dataset, config, shuffle=True)
    val_loader = (
        _make_loader(val_dataset, config, shuffle=False)
        if val_dataset is not None else None
    )
    print(
        f"Training on {len(train_dataset)} data from "
        f"{len(train_dirs)} dirs; validating on "
        f"{len(val_dataset) if val_dataset else 0} data from "
        f"{len(val_dirs)} dirs (device: {device})",
        flush=True,
    )

    network = LSPFrontierNet().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=config.learning_rate_decay
    )

    history: List[Dict[str, Dict[str, float]]] = []
    for epoch in range(1, config.num_epochs + 1):
        train_losses = _run_epoch(
            network, train_loader, config, device, optimizer,
            progress_prefix=f"epoch {epoch}/{config.num_epochs}  ",
        )
        val_losses = (
            _run_epoch(network, val_loader, config, device, optimizer=None)
            if val_loader is not None else {}
        )
        history.append({"train": train_losses, "val": val_losses})
        line = (f"epoch {epoch}/{config.num_epochs}  "
                f"train: {_format_losses(train_losses)}")
        if val_losses:
            line += f"  val: {_format_losses(val_losses)}"
        print(line, flush=True)
        scheduler.step()

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    network_file = save_dir / config.network_filename
    torch.save(network.cpu().state_dict(), network_file)
    with open(save_dir / "training_log.json", "w") as f:
        json.dump(
            {
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in asdict(config).items()
                },
                "train_dirs": [str(d) for d in train_dirs],
                "val_dirs": [str(d) for d in val_dirs],
                "history": history,
            },
            f,
            indent=2,
        )
    _save_loss_curves(history, save_dir / "loss_curves.png")
    print(f"Saved trained network to {network_file}")
    return TrainResult(network_file=network_file, history=history)
