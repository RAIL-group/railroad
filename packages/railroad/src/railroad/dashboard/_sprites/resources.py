"""Where the emoji font and the matching model live on disk.

Deliberately a copy of the resources-base constant rather than an import from
``environment.procthor.resources``: importing any submodule of that package runs
its ``__init__``, which downloads the ProcTHOR-10k dataset and the AI2-THOR
binaries. Drawing a plot must not do that. The value is kept identical so both
land in the same tree.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

DEFAULT_RESOURCES_BASE = Path(
    os.environ.get("PROCTHOR_RESOURCES_DIR", Path.cwd() / "resources")
)

EMOJI_SUBDIR = os.environ.get("RAILROAD_EMOJI_SUBDIR", "emoji")
EMOJI_SBERT_SUBDIR = os.environ.get("RAILROAD_EMOJI_SBERT_SUBDIR", "emoji_sbert")
EMOJI_SBERT_MODEL_NAME = os.environ.get(
    "RAILROAD_EMOJI_SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

NOTO_COMMIT = "f3ae03f5e9b3b8516fa151f7168159ca1a3e7515"
"""Pinned rather than tracking ``main``, so the glyphs cannot change underfoot.

A commit rather than a release tag: the newest tag predates Unicode 14, which
would cost the vocabulary a good share of the household objects it needs.
"""

NOTO_URL = (
    f"https://raw.githubusercontent.com/googlefonts/noto-emoji/{NOTO_COMMIT}"
    "/fonts/NotoColorEmoji.ttf"
)
NOTO_FILENAME = "NotoColorEmoji.ttf"


def get_emoji_dir(base_dir: Path | None = None) -> Path:
    """Get the emoji font directory."""
    directory = Path(base_dir or DEFAULT_RESOURCES_BASE) / EMOJI_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_emoji_sbert_dir(base_dir: Path | None = None) -> Path:
    """Get the glyph-matching model directory."""
    directory = Path(base_dir or DEFAULT_RESOURCES_BASE) / EMOJI_SBERT_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_emoji_font(base_dir: Path | None = None, *, force: bool = False) -> Path:
    """Ensure a colour emoji font is available, downloading Noto if needed.

    Explicit rather than lazy: plotting resolves fonts by lookup only, so that
    drawing a figure offline can never block on a ten-megabyte fetch.

    Returns whatever font the machine will actually use, which on macOS and most
    desktop Linux is one already installed -- downloading a second copy of
    something the system ships would be ten megabytes for nothing.
    """
    directory = get_emoji_dir(base_dir)
    font_path = directory / NOTO_FILENAME
    marker_path = directory / "download_complete.marker"

    if not force and marker_path.exists() and font_path.exists():
        return font_path

    if not force:
        from .fonts import find_font

        installed = find_font()
        if installed is not None:
            return installed

    print("Ensuring Noto Color Emoji Downloaded.")
    with urllib.request.urlopen(NOTO_URL) as response:
        content = response.read()
    # Publish whole or not at all: a partially written font would be cached
    # forever and fail to parse on every later run.
    tmp_path = font_path.with_suffix(font_path.suffix + f".{os.getpid()}.tmp")
    tmp_path.write_bytes(content)
    tmp_path.replace(font_path)
    marker_path.touch()
    return font_path


def ensure_emoji_sbert_model(
    base_dir: Path | None = None,
    *,
    model_name: str = EMOJI_SBERT_MODEL_NAME,
    force: bool = False,
) -> Path:
    """Ensure the glyph-matching sentence model is downloaded.

    A separate, much smaller model than the one ``procthor.learning`` uses: this
    one is chosen for its size and its tokenizer, which doubles as the word list
    that splits ``coffeemachine`` into ``coffee machine``.
    """
    model_dir = get_emoji_sbert_dir(base_dir)
    marker_path = model_dir / "download_complete.marker"

    if not force and marker_path.exists() and (model_dir / "modules.json").exists():
        return model_dir

    print("Ensuring Glyph Matching Model Downloaded.")
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(model_name).save(str(model_dir))
    marker_path.touch()
    return model_dir
