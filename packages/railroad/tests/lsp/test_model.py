"""Tests for the learned frontier-statistics network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from railroad.lsp import FrontierObservation, FrontierStatistics

torch = pytest.importorskip("torch")

from railroad.lsp.model import (  # noqa: E402
    LSPFrontierNet,
    load_frontier_statistics_model,
)

# Small images keep the tests fast; the adaptive pool makes the network
# size-agnostic (production panoramas are 256x512).
IMAGE_SHAPE = (64, 128)


def _batch(labels: list[bool]) -> dict[str, "torch.Tensor"]:
    n = len(labels)
    nan = float("nan")
    return {
        "image": torch.rand(n, 3, *IMAGE_SHAPE),
        "frontier_xy_ego": torch.randn(n, 2) * 30,
        "goal_xy_ego": torch.randn(n, 2) * 60,
        "label": torch.tensor([float(label) for label in labels]),
        "success_cost": torch.tensor(
            [100.0 if label else nan for label in labels]
        ),
        "optimistic_cost": torch.tensor(
            [80.0 if label else nan for label in labels]
        ),
        "exploration_cost": torch.tensor(
            [nan if label else 40.0 for label in labels]
        ),
    }


def _forward(net: LSPFrontierNet, batch: dict) -> "torch.Tensor":
    return net(batch["image"], batch["frontier_xy_ego"], batch["goal_xy_ego"])


def test_forward_shape() -> None:
    net = LSPFrontierNet()
    out = _forward(net, _batch([True, False, True]))
    assert out.shape == (3, 3)


def test_loss_components_finite_and_backward() -> None:
    net = LSPFrontierNet()
    batch = _batch([True, False, True, False])
    losses = net.loss(_forward(net, batch), batch)
    assert set(losses) == {
        "total", "feasibility", "delta_success_cost", "exploration_cost"
    }
    for value in losses.values():
        assert torch.isfinite(value)
    losses["total"].backward()  # NaN labels must not poison gradients
    for param in net.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_loss_masks_inapplicable_costs() -> None:
    net = LSPFrontierNet()
    # All infeasible: no example contributes to the delta-success term.
    batch = _batch([False, False])
    losses = net.loss(_forward(net, batch), batch)
    assert losses["delta_success_cost"].item() == 0.0
    # All feasible: no example contributes to the exploration term.
    batch = _batch([True, True])
    losses = net.loss(_forward(net, batch), batch)
    assert losses["exploration_cost"].item() == 0.0


def test_relative_positive_weight_scales_feasibility() -> None:
    net = LSPFrontierNet()
    batch = _batch([True, True])
    out = _forward(net, batch)
    base = net.loss(out, batch, relative_positive_weight=1.0)["feasibility"]
    doubled = net.loss(out, batch, relative_positive_weight=2.0)["feasibility"]
    assert doubled.item() == pytest.approx(2 * base.item(), rel=1e-5)


def _observation() -> FrontierObservation:
    return FrontierObservation(
        image=np.random.default_rng(0).integers(
            0, 255, size=(*IMAGE_SHAPE, 3), dtype=np.uint8
        ),
        frontier_xy_ego=(12.0, -3.0),
        goal_xy_ego=(40.0, 8.0),
    )


def test_load_frontier_statistics_model(tmp_path: Path) -> None:
    network_file = tmp_path / "net.pt"
    torch.save(LSPFrontierNet().state_dict(), network_file)

    model = load_frontier_statistics_model(network_file, device="cpu")
    assert model([]) == []

    statistics = model([_observation(), _observation()])
    assert len(statistics) == 2
    for stats in statistics:
        assert isinstance(stats, FrontierStatistics)
        assert 0.0 <= stats.prob_feasible <= 1.0
        assert stats.delta_success_cost >= 0.0
        assert stats.exploration_cost >= 0.0


def test_load_model_from_training_dir(tmp_path: Path) -> None:
    torch.save(LSPFrontierNet().state_dict(), tmp_path / "LSPFrontierNet.pt")
    model = load_frontier_statistics_model(tmp_path, device="cpu")
    assert len(model([_observation()])) == 1
