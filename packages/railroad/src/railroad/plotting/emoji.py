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
DOWNLOAD_TIMEOUT_S = 30.0
"""Socket timeout for the font fetch.

`ensure_all_resources` runs at `import railroad.environment.procthor`, and an
un-timed `urlopen` behind a captive portal hangs that import forever.
"""
FALLBACK_CODEPOINT = 0x1F4E6
MIN_SIMILARITY = 0.45

OVERRIDES = {
    "baseballbat": 0x26BE,
    "bowl": 0x1F963,
    "coffeemachine": 0x2615,
    "coffeemug": 0x2615,
    "desklamp": 0x1F4A1,
    "dumbbell": 0x1F3CB,
    "faucet": 0x1F6B0,
    "fridge": 0x1F9CA,
    "houseplant": 0x1FAB4,
    "ladle": 0x1F944,
    "laptop": 0x1F4BB,
    "mug": 0x2615,
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
_EXACT: dict[str, dict[str, int]] = {}
_MATCHES: dict[tuple[str, str], int] = {}
_PROVIDERS: dict[tuple[str, int, str], Any] = {}


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


def find_font(base_dir: Path | None = None) -> Path | None:
    override = os.environ.get("RAILROAD_EMOJI_FONT")
    candidates = ([Path(override)] if override else []) + [
        *(Path(path) for path in SYSTEM_FONT_PATHS),
        emoji_dir(base_dir) / NOTO_FILENAME,
    ]
    return next((path for path in candidates if path.is_file()), None)


def ensure_emoji_font(base_dir: Path | None = None, *, force: bool = False) -> Path:
    directory = emoji_dir(base_dir)
    font_path = directory / NOTO_FILENAME
    if not force and font_path.is_file():
        return font_path
    if not force:
        installed = find_font(base_dir)
        if installed is not None:
            return installed

    directory.mkdir(parents=True, exist_ok=True)
    print("Ensuring Noto Color Emoji Downloaded.")
    with urllib.request.urlopen(NOTO_URL, timeout=DOWNLOAD_TIMEOUT_S) as response:
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


def _load_model(base_dir: Path | None = None) -> Any:
    directory = model_dir(base_dir)
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


def _matrix(
    font_path: Path,
    candidates: tuple[tuple[str, int], ...],
    base_dir: Path | None = None,
) -> Any:
    import numpy as np

    model = _load_model(base_dir)
    if model is None:
        return None
    digest = hashlib.sha256(
        f"{MODEL_NAME}|{font_path}|{font_path.stat().st_mtime_ns}|{len(candidates)}".encode()
    ).hexdigest()[:16]
    if digest in _MATRICES:
        return _MATRICES[digest]
    cache = model_dir(base_dir) / f"vocab_{digest}.npz"
    if cache.is_file():
        # An .npz is a lazily-read zip: indexing the NpzFile hands back the
        # array but leaves the archive's descriptor open.
        with np.load(cache) as archive:
            _MATRICES[digest] = archive["labels"]
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


def _exact(font_path: Path, candidates: tuple[tuple[str, int], ...]) -> dict[str, int]:
    key = str(font_path)
    if key not in _EXACT:
        _EXACT[key] = {_key(label): codepoint for label, codepoint in candidates}
    return _EXACT[key]


def match(name: str, font_path: Path, base_dir: Path | None = None) -> int:
    key = _key(name)
    if key in OVERRIDES:
        return OVERRIDES[key]
    memo = (str(font_path), key)
    if memo in _MATCHES:
        return _MATCHES[memo]
    candidates = entries(font_path)
    exact = _exact(font_path, candidates)
    if key in exact:
        _MATCHES[memo] = exact[key]
        return exact[key]

    matrix, model = _matrix(font_path, candidates, base_dir), _load_model(base_dir)
    if matrix is None or model is None:
        # Deliberately not memoized: the model is downloadable, and a run that
        # fetches it should stop answering with the box.
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
    _MATCHES[memo] = (
        candidates[best][1]
        if similarity[best] >= MIN_SIMILARITY
        else FALLBACK_CODEPOINT
    )
    return _MATCHES[memo]


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


def _ink_square(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """The smallest square around *box*, centred on it.

    Square because sprites are drawn from a square raster, and centred on the
    ink rather than on the em box so that tall and wide glyphs are both whole.
    """
    left, top, right, bottom = box
    x, y = (left + right) / 2, (top + bottom) / 2
    half = max(right - left, bottom - top) / 2
    return round(x - half), round(y - half), round(x + half), round(y + half)


def rasterize(
    codepoint: int,
    target_px: int,
    font_path: Path | None = None,
    base_dir: Path | None = None,
) -> Any:
    import numpy as np

    path = font_path or find_font(base_dir)
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
    # `anchor="mm"` centres on the font's vertical metrics, but a colour
    # strike's ink sits well above that midpoint -- a teddy bear at strike 64
    # starts nine pixels above a 64px canvas -- so drawing straight into a
    # strike-sized square shaved the top off most glyphs. Draw with room to
    # spare and let the ink itself decide the frame.
    canvas = strike * 3
    image = Image.new("RGBA", (canvas, canvas))
    ImageDraw.Draw(image).text(
        (canvas / 2, canvas / 2),
        chr(codepoint),
        font=ImageFont.truetype(str(path), strike),
        embedded_color=True,
        anchor="mm",
    )
    box = image.getbbox()
    if box is None:
        _RASTERS[key] = None
        return None
    image = image.crop(_ink_square(box))
    if image.size != (target_px, target_px):
        image = image.resize((target_px, target_px), Image.Resampling.LANCZOS)
    _RASTERS[key] = np.asarray(image, dtype=np.uint8)
    return _RASTERS[key]


class GlyphProvider(Protocol):
    def glyph_for(self, name: str) -> Any: ...


class EmojiGlyphProvider:
    def __init__(
        self, font_path: Path, size_px: int = 64, base_dir: Path | None = None
    ) -> None:
        self.font_path = font_path
        self.size_px = size_px
        self.base_dir = base_dir
        self.cache: dict[str, Any] = {}

    def glyph_for(self, name: str) -> Any:
        if name not in self.cache:
            codepoint = match(name, self.font_path, self.base_dir)
            glyph = rasterize(codepoint, self.size_px, self.font_path)
            if glyph is None:
                glyph = rasterize(FALLBACK_CODEPOINT, self.size_px, self.font_path)
            self.cache[name] = glyph
        return self.cache[name]


def get_glyph_provider(
    size_px: int = 64, base_dir: Path | None = None
) -> GlyphProvider | None:
    """A provider for *base_dir*'s font, reused across renders.

    Caching the provider, not just building one, is what keeps a second render
    of the same figure -- `--save-plot --save-video` renders twice -- from
    re-resolving every object name through the sentence model.
    """
    path = find_font(base_dir)
    if path is None:
        return None
    key = (str(path), size_px, str(base_dir or ""))
    if key not in _PROVIDERS:
        _PROVIDERS[key] = EmojiGlyphProvider(path, size_px, base_dir)
    return _PROVIDERS[key]


def _reset_caches() -> None:
    """Drop every process-wide cache keyed on the font or resource root.

    Tests repoint `SYSTEM_FONT_PATHS` and `DEFAULT_RESOURCES_BASE`; without
    this a lookup cached under one of those outlives the patch that made it.
    """
    for cache in (
        _ENTRIES, _STRIKES, _RASTERS, _MODELS, _MATRICES,
        _EXACT, _MATCHES, _PROVIDERS,
    ):
        cache.clear()
    _MODELS.clear()
    _MATRICES.clear()
