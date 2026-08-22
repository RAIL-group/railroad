"""Emoji matching, rasterization, and resources."""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Protocol

DEFAULT_RESOURCES_BASE = Path(
    os.environ.get("PROCTHOR_RESOURCES_DIR", Path.cwd() / "resources")
)
EMOJI_SUBDIR = os.environ.get("RAILROAD_EMOJI_SUBDIR", "emoji")
MODEL_SUBDIR = os.environ.get("RAILROAD_EMOJI_MODEL_SUBDIR", "emoji_sbert")
MODEL_NAME = os.environ.get(
    "RAILROAD_EMOJI_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
NOTO_FILENAME = "NotoColorEmoji.ttf"
NOTO_URL = (
    "https://raw.githubusercontent.com/googlefonts/noto-emoji/"
    "f3ae03f5e9b3b8516fa151f7168159ca1a3e7515/fonts/NotoColorEmoji.ttf"
)
SYSTEM_FONT_PATHS = (
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    "/Library/Fonts/Apple Color Emoji.ttc",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/google-noto-emoji/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/TTF/NotoColorEmoji.ttf",
    "C:/Windows/Fonts/seguiemj.ttf",
)
CANDIDATE_STRIKES = (16, 20, 24, 26, 32, 40, 48, 52, 64, 96, 109, 128, 160)
FALLBACK_CODEPOINT = 0x1F4E6
MIN_SIMILARITY = 0.45

OVERRIDES = {
    "baseballbat": 0x26BE,
    "bowl": 0x1F963,
    "coffeemachine": 0x2615,
    "desklamp": 0x1F4A1,
    "dumbbell": 0x1F3CB,
    "faucet": 0x1F6B0,
    "fridge": 0x1F9CA,
    "houseplant": 0x1FAB4,
    "ladle": 0x1F944,
    "laptop": 0x1F4BB,
    "refrigerator": 0x1F9CA,
    "remotecontrol": 0x1F4FA,
    "safe": 0x1F510,
    "soapbottle": 0x1F9F4,
    "spraybottle": 0x1F9F4,
    "statue": 0x1F5FF,
    "tissuebox": 0x1F927,
    "winebottle": 0x1F377,
}

_SUFFIX = re.compile(r"_\d+$")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ENTRIES: dict[str, tuple[tuple[str, int], ...]] = {}
_STRIKES: dict[str, tuple[int, ...]] = {}
_RASTERS: dict[tuple[str, int, int], Any] = {}
_MODELS: dict[str, Any] = {}
_MATRICES: dict[str, Any] = {}


def object_sprites_enabled() -> bool:
    return os.environ.get("RAILROAD_OBJECT_SPRITES", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def emoji_dir(base_dir: Path | None = None) -> Path:
    return Path(base_dir or DEFAULT_RESOURCES_BASE) / EMOJI_SUBDIR


def model_dir(base_dir: Path | None = None) -> Path:
    return Path(base_dir or DEFAULT_RESOURCES_BASE) / MODEL_SUBDIR


def find_font() -> Path | None:
    override = os.environ.get("RAILROAD_EMOJI_FONT")
    candidates = ([Path(override)] if override else []) + [
        *(Path(path) for path in SYSTEM_FONT_PATHS),
        emoji_dir() / NOTO_FILENAME,
    ]
    return next((path for path in candidates if path.is_file()), None)


def ensure_emoji_font(base_dir: Path | None = None, *, force: bool = False) -> Path:
    directory = emoji_dir(base_dir)
    font_path = directory / NOTO_FILENAME
    if not force and font_path.is_file():
        return font_path
    if not force:
        installed = find_font()
        if installed is not None:
            return installed

    directory.mkdir(parents=True, exist_ok=True)
    print("Ensuring Noto Color Emoji Downloaded.")
    with urllib.request.urlopen(NOTO_URL) as response:
        content = response.read()
    temporary = font_path.with_suffix(f"{font_path.suffix}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(font_path)
    return font_path


def ensure_emoji_model(base_dir: Path | None = None, *, force: bool = False) -> Path:
    directory = model_dir(base_dir)
    if not force and (directory / "modules.json").is_file():
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    print("Ensuring Emoji Matching Model Downloaded.")
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(MODEL_NAME).save(str(directory))
    _MODELS.pop(str(directory), None)
    _MATRICES.clear()
    return directory


def ensure_emoji_resources(
    base_dir: Path | None = None, *, force: bool = False
) -> None:
    ensure_emoji_font(base_dir, force=force)
    ensure_emoji_model(base_dir, force=force)


def entries(font_path: Path) -> tuple[tuple[str, int], ...]:
    key = str(font_path)
    if key in _ENTRIES:
        return _ENTRIES[key]
    from fontTools.ttLib import TTCollection, TTFont

    if font_path.suffix.lower() == ".ttc":
        collection = TTCollection(key, lazy=True)
        try:
            cmap = collection.fonts[0].getBestCmap() or {}
        finally:
            collection.close()
    else:
        font = TTFont(key, lazy=True)
        try:
            cmap = font.getBestCmap() or {}
        finally:
            font.close()
    excluded = range(0x1F1E6, 0x1F200), range(0x1F3FB, 0x1F400)
    result = []
    for codepoint in sorted(cmap):
        if not (0x203C <= codepoint <= 0x3299 or 0x1F000 <= codepoint <= 0x1FAFF):
            continue
        if any(codepoint in block for block in excluded):
            continue
        label = unicodedata.name(chr(codepoint), "")
        if label:
            result.append((label.lower().replace("-", " "), codepoint))
    _ENTRIES[key] = tuple(result)
    return _ENTRIES[key]


def _phrase(name: str) -> str:
    name = _CAMEL.sub(" ", _SUFFIX.sub("", name))
    return " ".join(name.replace("_", " ").replace("-", " ").lower().split())


def _key(name: str) -> str:
    return _NON_ALNUM.sub("", _phrase(name))


def _load_model() -> Any:
    directory = model_dir()
    if not (directory / "modules.json").is_file():
        return None
    key = str(directory)
    if key not in _MODELS:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None
        _MODELS[key] = SentenceTransformer(key)
    return _MODELS[key]


def _matrix(font_path: Path, candidates: tuple[tuple[str, int], ...]) -> Any:
    import numpy as np

    model = _load_model()
    if model is None:
        return None
    digest = hashlib.sha256(
        f"{MODEL_NAME}|{font_path}|{font_path.stat().st_mtime_ns}|{len(candidates)}".encode()
    ).hexdigest()[:16]
    if digest in _MATRICES:
        return _MATRICES[digest]
    cache = model_dir() / f"vocab_{digest}.npz"
    if cache.is_file():
        _MATRICES[digest] = np.load(cache)["labels"]
    else:
        _MATRICES[digest] = np.asarray(
            model.encode(
                [label for label, _ in candidates],
                normalize_embeddings=True,
                batch_size=256,
                show_progress_bar=False,
            )
        )
        temporary = cache.with_name(f".{cache.stem}.{os.getpid()}.npz")
        np.savez(temporary, labels=_MATRICES[digest])
        temporary.replace(cache)
    return _MATRICES[digest]


def match(name: str, font_path: Path) -> int:
    key = _key(name)
    if key in OVERRIDES:
        return OVERRIDES[key]
    candidates = entries(font_path)
    exact = {_key(label): codepoint for label, codepoint in candidates}
    if key in exact:
        return exact[key]

    matrix, model = _matrix(font_path, candidates), _load_model()
    if matrix is None or model is None:
        return FALLBACK_CODEPOINT
    import numpy as np

    phrase = _phrase(name)
    queries = [phrase, phrase.rsplit(" ", 1)[-1]] if " " in phrase else [phrase]
    embedded = np.asarray(
        model.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )
    similarity = (embedded @ matrix.T).max(axis=0)
    best = int(similarity.argmax())
    return (
        candidates[best][1]
        if similarity[best] >= MIN_SIMILARITY
        else FALLBACK_CODEPOINT
    )


def probe_strikes(font_path: Path) -> tuple[int, ...]:
    key = str(font_path)
    if key not in _STRIKES:
        from PIL import ImageFont

        usable = []
        for size in CANDIDATE_STRIKES:
            try:
                ImageFont.truetype(key, size)
            except OSError:
                continue
            usable.append(size)
        _STRIKES[key] = tuple(usable)
    return _STRIKES[key]


def rasterize(codepoint: int, target_px: int, font_path: Path | None = None) -> Any:
    import numpy as np

    path = font_path or find_font()
    if path is None:
        return None
    key = (str(path), codepoint, target_px)
    if key in _RASTERS:
        return _RASTERS[key]
    strikes = probe_strikes(path)
    if not strikes:
        return None

    from PIL import Image, ImageDraw, ImageFont

    strike = next((size for size in strikes if size >= target_px), strikes[-1])
    image = Image.new("RGBA", (strike, strike))
    ImageDraw.Draw(image).text(
        (strike / 2, strike / 2),
        chr(codepoint),
        font=ImageFont.truetype(str(path), strike),
        embedded_color=True,
        anchor="mm",
    )
    if image.getbbox() is None:
        _RASTERS[key] = None
        return None
    if strike != target_px:
        image = image.resize((target_px, target_px), Image.Resampling.LANCZOS)
    _RASTERS[key] = np.asarray(image, dtype=np.uint8)
    return _RASTERS[key]


class GlyphProvider(Protocol):
    def glyph_for(self, name: str) -> Any: ...


class EmojiGlyphProvider:
    def __init__(self, font_path: Path, size_px: int = 64) -> None:
        self.font_path = font_path
        self.size_px = size_px
        self.cache: dict[str, Any] = {}

    def glyph_for(self, name: str) -> Any:
        if name not in self.cache:
            glyph = rasterize(match(name, self.font_path), self.size_px, self.font_path)
            if glyph is None:
                glyph = rasterize(FALLBACK_CODEPOINT, self.size_px, self.font_path)
            self.cache[name] = glyph
        return self.cache[name]


def get_glyph_provider(size_px: int = 64) -> GlyphProvider | None:
    path = find_font()
    return EmojiGlyphProvider(path, size_px) if path else None


def _reset_caches() -> None:
    _ENTRIES.clear()
    _STRIKES.clear()
    _RASTERS.clear()
    _MODELS.clear()
    _MATRICES.clear()
