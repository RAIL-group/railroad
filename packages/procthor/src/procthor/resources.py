from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import prior  # Already needed for procthor-10k

# Base dir for all resources (overridable by env):
#   PROCTHOR_RESOURCES_DIR=/path/to/resources
DEFAULT_RESOURCES_BASE = Path(
    os.environ.get("PROCTHOR_RESOURCES_DIR", Path.cwd() / "resources")
)

# Subdirs, overridable via env if needed
DEFAULT_PROCTHOR_10K_SUBDIR = os.environ.get("PROCTHOR_DATA_SUBDIR", "procthor-10k")
DEFAULT_SBERT_SUBDIR = os.environ.get("PROCTHOR_SBERT_SUBDIR", "sentence_transformers")
DEFAULT_AI2THOR_SUBDIR = os.environ.get("PROCTHOR_AI2THOR_SUBDIR", "ai2thor")

# Default SBERT model name
DEFAULT_SBERT_MODEL_NAME = os.environ.get(
    "PROCTHOR_SBERT_MODEL_NAME", "bert-base-nli-stsb-mean-tokens"
)


def get_procthor_10k_dir(base_dir: Optional[Path] = None) -> Path:
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

    Uses a 'download_complete.marker' marker and writes to a temporary file first.
    """
    data_dir = get_procthor_10k_dir(base_dir)
    data_path = data_dir / "data.jsonl"
    marker_path = data_dir / "download_complete.marker"
    tmp_path = data_dir / "data.jsonl.tmp"

    # Fast path: prior successful download.
    if not force and marker_path.exists() and data_path.exists():
        return data_dir

    print("Ensuring ProcTHOR-10k Dataset Downloaded.")
    dataset = prior.load_dataset("procthor-10k")
    train_data = dataset["train"]

    with tmp_path.open("w") as f:
        for entry in train_data:
            json.dump(entry, f)
            f.write("\n")

    tmp_path.replace(data_path)
    marker_path.touch()

    return data_dir


def get_sbert_dir(base_dir: Optional[Path] = None) -> Path:
    if base_dir is None:
        base_dir = DEFAULT_RESOURCES_BASE
    model_dir = Path(base_dir) / DEFAULT_SBERT_SUBDIR
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def ensure_sbert_model(
    base_dir: Optional[Path] = None,
    *,
    model_name: str = DEFAULT_SBERT_MODEL_NAME,
    force: bool = False,
) -> Path:
    """
    Ensure that the SBERT model is downloaded and saved.

    Semantics:
    - Uses a 'download_complete.marker' marker in the model directory.
    - Assumes SentenceTransformer.save() will create a 'model.safetensors'
      (or equivalent) in that directory.
    - If `force=False` and both marker + model file exist, this is a no-op.
    """
    model_dir = get_sbert_dir(base_dir)
    marker_path = model_dir / "download_complete.marker"
    safetensor_path = model_dir / "model.safetensors"

    # Fast path: model already present and previously marked complete.
    if not force and marker_path.exists() and safetensor_path.exists():
        return model_dir

    # Download / load the model and save to disk.
    print("Ensuring SentenceTransformer Model Downloaded.")
    from sentence_transformers import SentenceTransformer  # lazy import
    model = SentenceTransformer(model_name)
    model.save(str(model_dir))

    # Only mark complete **after** successful save.
    marker_path.touch()

    return model_dir


def get_ai2thor_marker_dir(base_dir: Optional[Path] = None) -> Path:
    """
    Returns a directory in which we place a 'download_complete.marker' marker
    indicating that AI2-THOR has been successfully initialized at least once.

    Note: AI2-THOR itself manages where the actual simulator binaries live
    (usually under a user cache). We just maintain a marker on our side.
    """
    if base_dir is None:
        base_dir = DEFAULT_RESOURCES_BASE
    marker_dir = Path(base_dir) / DEFAULT_AI2THOR_SUBDIR
    marker_dir.mkdir(parents=True, exist_ok=True)
    return marker_dir


def ensure_ai2thor_simulator(
    base_dir: Optional[Path] = None,
    *,
    force: bool = False,
) -> Path:
    """
    Ensure that the AI2-THOR simulator is available.

    Semantics:
    - Uses a 'download_complete.marker' marker under resources/ai2thor (or
      configured subdir) to record that we've successfully instantiated a
      Controller() once.
    - AI2-THOR itself manages actual binary downloads to its cache dir.
    """
    marker_dir = get_ai2thor_marker_dir(base_dir)
    marker_path = marker_dir / "download_complete.marker"

    if not force and marker_path.exists():
        return marker_dir

    # Instantiating Controller triggers AI2-THOR's own download logic.
    print("Ensuring AI2THOR Simulator Downloaded.")
    from ai2thor.controller import Controller  # lazy import
    controller = Controller()
    # Optionally, you may want to immediately stop it:
    try:
        controller.stop()
    except Exception:
        # Not critical; the important part is that the download completed.
        pass

    marker_path.touch()

    return marker_dir


# ---------------------------------------------------------------------------
# Convenience: ensure everything
# ---------------------------------------------------------------------------

def ensure_all_resources(
    base_dir: Optional[Path] = None,
    *,
    force: bool = False,
) -> None:
    """
    Convenience function to ensure all core resources are available.
    """
    ensure_procthor_10k(base_dir=base_dir, force=force)
    ensure_sbert_model(base_dir=base_dir, force=force)
    ensure_ai2thor_simulator(base_dir=base_dir, force=force)
