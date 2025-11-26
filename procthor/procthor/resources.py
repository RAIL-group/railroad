# procthor/resources.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import prior


# Default base dir: ./resources, overridable by env var.
# These defaults mirror your original intent, but are now configurable.
DEFAULT_RESOURCES_BASE = Path(
    os.environ.get("PROCTHOR_RESOURCES_DIR", Path.cwd() / "resources")
)

# Subdir for the procthor-10k dataset; also overridable.
DEFAULT_PROCTHOR_10K_SUBDIR = os.environ.get("PROCTHOR_DATA_SUBDIR", "procthor-10k")


def get_procthor_10k_dir(base_dir: Optional[Path] = None) -> Path:
    """
    Return the directory where procthor-10k data should live.
    Creates the directory if it does not exist.
    """
    if base_dir is None:
        base_dir = DEFAULT_RESOURCES_BASE
    data_dir = Path(base_dir) / DEFAULT_PROCTHOR_10K_SUBDIR
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def ensure_procthor_10k(
    base_dir: Optional[Path] = None,
    *,
    force: bool = False,
) -> Path:
    """
    Ensure that procthor-10k data is downloaded and consistent.

    Semantics:
    - Uses a 'download_complete.tmp' marker file to indicate success.
    - Writes to a temporary file first, then atomically renames to 'data.jsonl'.
    - If `force=False` and both marker + data file exist, this is a no-op.

    Returns
    -------
    Path
        Directory containing the dataset.
    """
    data_dir = get_procthor_10k_dir(base_dir)
    data_path = data_dir / "data.jsonl"
    marker_path = data_dir / "download_complete.marker"
    tmp_path = data_dir / "data.jsonl.tmp"

    # Fast path: prior successful download.
    if not force and marker_path.exists() and data_path.exists():
        return data_dir

    # Download dataset via prior.
    dataset = prior.load_dataset("procthor-10k")
    train_data = dataset["train"]

    # Write to a temporary file first.
    with tmp_path.open("w") as f:
        for entry in train_data:
            json.dump(entry, f)
            f.write("\n")

    # Atomically replace the target file on the same filesystem.
    tmp_path.replace(data_path)

    # Only create the marker after a successful write.
    marker_path.touch()

    return data_dir
