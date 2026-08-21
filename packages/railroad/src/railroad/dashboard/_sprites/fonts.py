"""Finding a colour emoji font and turning codepoints into pixels.

matplotlib cannot draw colour emoji as text -- its text stack has no COLR/CBDT
support -- so glyphs are rasterised through Pillow and placed as images instead.
Colour emoji fonts are bitmap fonts, which means they render at a fixed set of
*strike* sizes and raise on anything else, and the set differs by font: Apple
Color Emoji offers nine sizes, Noto exactly one.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

from .resources import NOTO_FILENAME, get_emoji_dir

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
"""Sizes worth probing. Which of them work is a property of the font."""

_FONT_PATH: Path | None = None
_LOOKED_UP = False
"""Kept apart from _FONT_PATH so that "looked, and there is none" is not
retried on every plot -- and does not warn again either."""

_STRIKES: dict[str, tuple[int, ...]] = {}
_RASTER_CACHE: dict[tuple[str, int, int], Any] = {}
_WARNED = False


def find_font(*, download: bool = False) -> Path | None:
    """Locate a colour emoji font, or ``None`` if the machine has none.

    Lookup only by default. Callers that want the font fetched should call
    :func:`~._sprites.resources.ensure_emoji_font` explicitly, so that no plot
    silently turns into a network request.
    """
    global _FONT_PATH, _LOOKED_UP, _WARNED
    if _LOOKED_UP and not download:
        return _FONT_PATH

    override = os.environ.get("RAILROAD_EMOJI_FONT")
    candidates = [Path(override)] if override else []
    candidates += [Path(path) for path in SYSTEM_FONT_PATHS]
    candidates.append(get_emoji_dir() / NOTO_FILENAME)

    found = next((path for path in candidates if path.is_file()), None)
    if found is None and download:
        from .resources import ensure_emoji_font

        found = ensure_emoji_font()
    if found is None and not _WARNED:
        _WARNED = True
        warnings.warn(
            "No colour emoji font found, so object glyphs will not be drawn. "
            "Set RAILROAD_EMOJI_FONT, install one, or call "
            "railroad.dashboard._sprites.resources.ensure_emoji_font().",
            RuntimeWarning,
            stacklevel=2,
        )
    _FONT_PATH = found
    _LOOKED_UP = True
    return found


def probe_strikes(font_path: Path) -> tuple[int, ...]:
    """The pixel sizes *font_path* will actually render at.

    Probed rather than assumed: the usable set is font-specific, and asking for
    a size a bitmap font does not carry raises rather than scaling.
    """
    key = str(font_path)
    if key in _STRIKES:
        return _STRIKES[key]

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


def _choose_strike(strikes: tuple[int, ...], target_px: int) -> int:
    """Smallest strike that covers *target_px*, else the largest available."""
    return next((size for size in strikes if size >= target_px), strikes[-1])


def rasterize(codepoint: int, target_px: int, font_path: Path | None = None) -> Any:
    """Render one codepoint to an RGBA array of exactly ``target_px`` square.

    Returns ``None`` when no font is available or the font cannot draw the
    codepoint -- callers treat that as "no sprite", never as an error.
    """
    import numpy as np

    path = font_path or find_font()
    if path is None:
        return None
    key = (str(path), codepoint, target_px)
    if key in _RASTER_CACHE:
        return _RASTER_CACHE[key]

    strikes = probe_strikes(path)
    if not strikes:
        return None

    from PIL import Image, ImageDraw, ImageFont

    strike = _choose_strike(strikes, target_px)
    font = ImageFont.truetype(str(path), strike)
    image = Image.new("RGBA", (strike, strike), (0, 0, 0, 0))
    ImageDraw.Draw(image).text(
        (strike / 2, strike / 2),
        chr(codepoint),
        font=font,
        embedded_color=True,
        anchor="mm",
    )
    if image.getbbox() is None:
        # Nothing was drawn -- the font has no glyph here, whatever its cmap said.
        _RASTER_CACHE[key] = None
        return None
    if strike != target_px:
        image = image.resize((target_px, target_px), Image.Resampling.LANCZOS)

    array = np.asarray(image, dtype=np.uint8)
    _RASTER_CACHE[key] = array
    return array


def _reset_caches() -> None:
    """Forget every memoised lookup. For tests that move the font around."""
    global _FONT_PATH, _LOOKED_UP, _WARNED
    _FONT_PATH = None
    _LOOKED_UP = False
    _WARNED = False
    _STRIKES.clear()
    _RASTER_CACHE.clear()
