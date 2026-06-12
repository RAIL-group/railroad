"""PyTorch-friendly access to LSP training data.

:class:`LSPFrontierDataset` spans the per-seed directories produced by
``railroad lsp generate-data`` (or a single
:class:`~railroad.lsp.TrainingDataWriter` directory) and yields tensor
dicts ready for a training loop. Torch is imported lazily, so
``railroad.lsp`` itself stays torch-free.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

from .data import load_datum, read_index

if TYPE_CHECKING:
    import torch

_SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


def _require_torch():  # noqa: ANN202
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "LSPFrontierDataset requires PyTorch. Install it with "
            "'pip install torch'."
        ) from e
    return torch


def _discover_dirs(root: Path) -> List[Path]:
    if not root.is_dir():
        raise ValueError(f"{root} is not a directory")
    if (root / "index.jsonl").exists():
        return [root]  # a single TrainingDataWriter directory
    seed_dirs = sorted(
        child for child in root.iterdir()
        if child.is_dir() and _SEED_DIR_RE.match(child.name)
    )
    if not seed_dirs:
        raise ValueError(
            f"{root} contains neither an index.jsonl nor seed_* directories"
        )
    return seed_dirs


class LSPFrontierDataset:
    """Map-style dataset over LSP training-data directories.

    Construct from an experiment directory (``seed_*`` children, as laid
    out by ``railroad lsp generate-data``), a single data directory, or
    an explicit list of directories via *dirs*. Duck-typed for
    ``torch.utils.data.DataLoader`` (defines ``__len__``/``__getitem__``);
    deliberately not a ``torch.utils.data.Dataset`` subclass so this
    module imports without torch.

    Each item is a dict of tensors:

    - ``image``: float32, CHW, scaled to [0, 1]
    - ``frontier_xy_ego``, ``goal_xy_ego``: float32, shape (2,)
    - ``label``: float32 scalar (1.0 = frontier leads to the goal)
    - ``success_cost``, ``optimistic_cost``, ``exploration_cost``:
      float32 scalars, NaN where the cost does not apply — mask with
      ``label`` in the loss (the standard LSP recipe).
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        dirs: Sequence[str | Path] | None = None,
    ) -> None:
        if dirs is not None:
            if root is not None:
                raise ValueError("Pass exactly one of root or dirs")
            data_dirs = [Path(d) for d in dirs]
        elif root is not None:
            data_dirs = _discover_dirs(Path(root))
        else:
            raise ValueError("Pass exactly one of root or dirs")
        self._entries: List[Tuple[Path, str]] = [
            (data_dir, entry["file"])
            for data_dir in data_dirs
            for entry in read_index(data_dir)
        ]

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        torch = _require_torch()
        data_dir, filename = self._entries[idx]
        datum = load_datum(data_dir / filename)

        def cost(value: float | None) -> "torch.Tensor":
            return torch.tensor(
                float("nan") if value is None else value, dtype=torch.float32
            )

        image = torch.from_numpy(datum.image).permute(2, 0, 1).float() / 255.0
        return {
            "image": image,
            "frontier_xy_ego": torch.tensor(
                datum.frontier_xy_ego, dtype=torch.float32
            ),
            "goal_xy_ego": torch.tensor(datum.goal_xy_ego, dtype=torch.float32),
            "label": torch.tensor(float(datum.label), dtype=torch.float32),
            "success_cost": cost(datum.success_cost),
            "optimistic_cost": cost(datum.optimistic_cost),
            "exploration_cost": cost(datum.exploration_cost),
        }
