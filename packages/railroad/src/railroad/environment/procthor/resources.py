"""Resource management for ProcTHOR environments.

Handles downloading and caching of:
- ProcTHOR-10k dataset
- SentenceTransformer model
- AI2-THOR simulator binaries
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# Base dir for all resources (overridable by env):
DEFAULT_RESOURCES_BASE = Path(
    os.environ.get("PROCTHOR_RESOURCES_DIR", Path.cwd() / "resources")
)

DEFAULT_PROCTHOR_10K_SUBDIR = os.environ.get("PROCTHOR_DATA_SUBDIR", "procthor-10k")
DEFAULT_SBERT_SUBDIR = os.environ.get("PROCTHOR_SBERT_SUBDIR", "sentence_transformers")
DEFAULT_AI2THOR_SUBDIR = os.environ.get("PROCTHOR_AI2THOR_SUBDIR", "ai2thor")
DEFAULT_SBERT_MODEL_NAME = os.environ.get(
    "PROCTHOR_SBERT_MODEL_NAME", "bert-base-nli-stsb-mean-tokens"
)


def get_procthor_10k_dir(base_dir: Optional[Path] = None) -> Path:
    """Get the ProcTHOR-10k data directory."""
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
    """Ensure ProcTHOR-10k data is downloaded."""
    import prior

    data_dir = get_procthor_10k_dir(base_dir)
    data_path = data_dir / "data.jsonl"
    marker_path = data_dir / "download_complete.marker"
    tmp_path = data_dir / "data.jsonl.tmp"

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
    """Get the SBERT model directory."""
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
    """Ensure SBERT model is downloaded."""
    model_dir = get_sbert_dir(base_dir)
    marker_path = model_dir / "download_complete.marker"
    safetensor_path = model_dir / "model.safetensors"

    if not force and marker_path.exists() and safetensor_path.exists():
        return model_dir

    print("Ensuring SentenceTransformer Model Downloaded.")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    model.save(str(model_dir))
    marker_path.touch()
    return model_dir


def get_ai2thor_marker_dir(base_dir: Optional[Path] = None) -> Path:
    """Get the AI2-THOR marker directory."""
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
    """Ensure AI2-THOR simulator is available."""
    marker_dir = get_ai2thor_marker_dir(base_dir)
    marker_path = marker_dir / "download_complete.marker"

    if not force and marker_path.exists():
        return marker_dir

    print("Ensuring AI2THOR Simulator Downloaded.")
    from ai2thor.controller import Controller
    controller = Controller()
    try:
        controller.stop()
    except Exception:
        pass

    marker_path.touch()
    return marker_dir


def ensure_all_resources(
    base_dir: Optional[Path] = None,
    *,
    force: bool = False,
) -> None:
    """Ensure all ProcTHOR resources are available."""
    ensure_procthor_10k(base_dir=base_dir, force=force)
    ensure_sbert_model(base_dir=base_dir, force=force)
    ensure_ai2thor_simulator(base_dir=base_dir, force=force)


def _cache_scene_worker(seed: int) -> None:
    from .scene import ProcTHORScene
    ProcTHORScene(seed=seed)


def cache_procthor_scene(start_seed: int, end_seed: Optional[int] = None) -> None:
    """
    Helper function to cache a ProcTHOR scene for a given seed.
    This runs the caching logic in a separate, blocking process to ensure
    that the AI2-THOR controller is properly isolated and cleaned up,
    which is crucial when caching multiple scenes in a loop.
    """
    import multiprocessing
    from rich.progress import Progress
    ctx = multiprocessing.get_context("spawn")
    end_seed = start_seed if end_seed is None else end_seed
    with Progress() as progress:
        task = progress.add_task("[cyan]Caching scene", total=(end_seed - start_seed + 1))
        for seed in range(start_seed, end_seed + 1):
            scene_cache_path = get_procthor_10k_dir() / f"cache/scene_{seed}.pkl"
            if scene_cache_path.exists():
                progress.update(task, advance=1, description=f"[cyan]Cache exists [{seed}/{end_seed}]")
                continue
            process = ctx.Process(target=_cache_scene_worker, args=(seed,))
            process.start()
            process.join()  # Block until the caching process is complete
            process.close()
            progress.update(task, advance=1, description=f"[cyan]Caching scene [{seed}/{end_seed}]")
        progress.update(task, advance=1, description="[cyan]Caching complete!")
