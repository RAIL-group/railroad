"""Learned frontier-statistics network (PyTorch).

:class:`LSPFrontierNet` maps a :class:`~railroad.lsp.FrontierObservation`
— the frontier-centered panorama plus egocentric frontier/goal locations
— to the three planner-facing statistics: a feasibility logit, the delta
success cost, and the exploration cost. The architecture follows the
reference VisLSPOriented model: stacked conv/pool encoder blocks with
the (batch-normalized) location scalars injected as extra channels
partway through, then a small fully-connected head.

Train with ``railroad lsp train-network`` (:mod:`railroad.lsp.train`);
at planning time, wrap the saved weights for
:class:`~railroad.lsp.LearnedFrontierStatistics` via
:func:`load_frontier_statistics_model`.

Requires torch, so — like :mod:`railroad.lsp.environment` — this module
is not imported by ``railroad.lsp`` itself and must be imported
explicitly::

    from railroad.lsp.model import LSPFrontierNet
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .types import FrontierObservation, FrontierStatistics

NETWORK_FILENAME = "LSPFrontierNet.pt"


def default_device() -> torch.device:
    """The best available torch device: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _EncoderBlock(nn.Module):
    """Two (conv3x3 -> batch norm -> leaky ReLU) layers, then a 2x2 pool."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        channels = in_channels
        for _ in range(2):
            layers += [
                nn.Conv2d(channels, out_channels, kernel_size=3, padding=1,
                          bias=False),
                nn.BatchNorm2d(out_channels, momentum=0.01),
                nn.LeakyReLU(0.1, inplace=True),
            ]
            channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LSPFrontierNet(nn.Module):
    """CNN from a frontier observation to frontier statistics.

    Inputs are batches of the tensors :class:`~railroad.lsp.LSPFrontierDataset`
    yields: a [0, 1] float CHW panorama plus the egocentric frontier and
    goal locations (grid cells, x forward / y left). The output is
    ``(B, 3)``: ``[feasibility logit, delta_success_cost,
    exploration_cost]`` — apply a sigmoid to the first to get
    ``prob_feasible``.
    """

    name = "LSPFrontierNet"

    def __init__(self) -> None:
        super().__init__()
        self.enc_1 = _EncoderBlock(3, 16)
        self.enc_2 = _EncoderBlock(16, 32)
        # The four location scalars enter here as constant channels.
        self.coord_bn = nn.BatchNorm2d(4)
        self.enc_3 = _EncoderBlock(32 + 4, 64)
        self.enc_4 = _EncoderBlock(64, 64)
        self.enc_5 = _EncoderBlock(64, 128)
        self.enc_6 = _EncoderBlock(128, 128)
        # Identity for the native 256x512 panoramas (whose encoding is
        # already 4x8); makes other input sizes work too.
        self.pool = nn.AdaptiveAvgPool2d((4, 8))
        self.conv_1x1 = nn.Conv2d(128, 8, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 4 * 8, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 3),
        )

    def forward(
        self,
        image: torch.Tensor,
        frontier_xy_ego: torch.Tensor,
        goal_xy_ego: torch.Tensor,
    ) -> torch.Tensor:
        x = self.enc_2(self.enc_1(image))
        coords = torch.cat((frontier_xy_ego, goal_xy_ego), dim=1)
        coords = self.coord_bn(
            coords[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        )
        x = torch.cat((x, coords), dim=1)
        x = self.enc_6(self.enc_5(self.enc_4(self.enc_3(x))))
        return self.fc(self.conv_1x1(self.pool(x)))

    def loss(
        self,
        output: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        *,
        relative_positive_weight: float = 1.0,
        cost_scale: float = 100.0,
    ) -> Dict[str, torch.Tensor]:
        """Loss against a :class:`~railroad.lsp.LSPFrontierDataset` batch.

        Feasibility is a weighted binary cross-entropy; the two costs
        are squared errors (scaled by *cost_scale*) masked to the
        examples where each applies — delta success cost on feasible
        frontiers, exploration cost on infeasible ones — with NaN
        labels (cost not defined) masked out as well. Returns the
        per-term losses plus their sum under ``"total"``.
        """
        device = output.device
        label = batch["label"].to(device)

        logits = output[:, 0]
        feasibility = -(
            relative_positive_weight * label * functional.logsigmoid(logits)
            + (1.0 - label) * functional.logsigmoid(-logits)
        ).mean()

        def masked_cost_loss(
            pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
        ) -> torch.Tensor:
            mask = mask * torch.isfinite(target).float()
            error = torch.square((pred - torch.nan_to_num(target)) / cost_scale)
            return (error * mask).sum() / mask.sum().clamp(min=1.0)

        delta_target = (
            batch["success_cost"].to(device)
            - batch["optimistic_cost"].to(device)
        )
        delta_success_cost = masked_cost_loss(output[:, 1], delta_target, label)
        exploration_cost = masked_cost_loss(
            output[:, 2], batch["exploration_cost"].to(device), 1.0 - label
        )

        return {
            "total": feasibility + delta_success_cost + exploration_cost,
            "feasibility": feasibility,
            "delta_success_cost": delta_success_cost,
            "exploration_cost": exploration_cost,
        }


class _NetworkFrontierStatisticsModel:
    """:class:`~railroad.lsp.FrontierStatisticsModel` over a trained net."""

    def __init__(self, network: LSPFrontierNet, device: torch.device) -> None:
        self._network = network
        self._device = device

    def __call__(
        self, observations: Sequence[FrontierObservation]
    ) -> List[FrontierStatistics]:
        if not observations:
            return []
        image = torch.stack([
            torch.from_numpy(obs.image).permute(2, 0, 1).float() / 255.0
            for obs in observations
        ])
        frontier_xy = torch.tensor(
            [obs.frontier_xy_ego for obs in observations], dtype=torch.float32
        )
        goal_xy = torch.tensor(
            [obs.goal_xy_ego for obs in observations], dtype=torch.float32
        )
        with torch.no_grad():
            output = self._network(
                image.to(self._device),
                frontier_xy.to(self._device),
                goal_xy.to(self._device),
            ).cpu()
        return [
            FrontierStatistics(
                prob_feasible=float(torch.sigmoid(row[0])),
                delta_success_cost=max(0.0, float(row[1])),
                exploration_cost=max(0.0, float(row[2])),
            )
            for row in output
        ]


def load_frontier_statistics_model(
    network_file: str | Path,
    device: str | torch.device | None = None,
) -> _NetworkFrontierStatisticsModel:
    """Load trained weights as a model for ``LearnedFrontierStatistics``.

    *network_file* is the state dict ``railroad lsp train-network``
    saves (or a training directory containing one). The result maps a
    batch of observations to statistics::

        from railroad.lsp import LearnedFrontierStatistics
        from railroad.lsp.model import load_frontier_statistics_model

        estimator = LearnedFrontierStatistics(
            load_frontier_statistics_model("training/LSPFrontierNet.pt")
        )
    """
    network_file = Path(network_file)
    if network_file.is_dir():
        network_file = network_file / NETWORK_FILENAME
    device = default_device() if device is None else torch.device(device)
    network = LSPFrontierNet()
    network.load_state_dict(
        torch.load(network_file, map_location=device, weights_only=True)
    )
    network.to(device)
    network.eval()
    return _NetworkFrontierStatisticsModel(network, device)
