"""The set of glyphs a font can actually draw, with a label for each.

Both halves come from the font itself, so there is no table to maintain and no
risk of naming a glyph the installed font cannot render. Labels come from
``unicodedata``, which ships with Python.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

EMOJI_RANGES = (
    (0x203C, 0x2300),
    (0x2300, 0x23FF),   # includes U+23F0 ALARM CLOCK
    (0x25A0, 0x27BF),
    (0x2B00, 0x2BFF),
    (0x2E80, 0x3299),
    (0x1F000, 0x1FAFF),
)
"""Deliberately generous.

A tighter filter around the pictograph blocks drops glyphs that read as ordinary
household objects -- ALARM CLOCK and WATCH both live down in the 0x23xx range --
and the cost of a loose filter is only a few dozen extra candidates.
"""

EXCLUDED_RANGES = (
    (0x1F1E6, 0x1F1FF),  # regional indicators -- flag halves
    (0x1F3FB, 0x1F3FF),  # skin tone modifiers
    (0x1F9B0, 0x1F9B3),  # hair components
    (0x20E3, 0x20E3),    # combining enclosing keycap
    (0xFE00, 0xFE0F),    # variation selectors
)
"""Codepoints that only render as part of a sequence.

A cmap lists them, but drawing one alone produces an empty bitmap, so a name
that matched one would silently get no sprite at all. Their labels -- "regional
indicator symbol letter r" -- could never be the right answer anyway.
"""

_VOCABULARY: dict[str, tuple[tuple[str, int], ...]] = {}


def _in_range(codepoint: int) -> bool:
    if any(low <= codepoint <= high for low, high in EXCLUDED_RANGES):
        return False
    return any(low <= codepoint <= high for low, high in EMOJI_RANGES)


def build(font_path: Path) -> tuple[tuple[str, int], ...]:
    """``(label, codepoint)`` for every emoji *font_path* can draw."""
    key = str(font_path)
    if key in _VOCABULARY:
        return _VOCABULARY[key]

    from fontTools.ttLib import TTCollection, TTFont

    if font_path.suffix.lower() == ".ttc":
        collection = TTCollection(key, lazy=True)
        cmap = collection.fonts[0].getBestCmap()
    else:
        cmap = TTFont(key, lazy=True).getBestCmap()
    if not cmap:
        # A font with no usable unicode cmap can tell us nothing about what it
        # can draw, which is the same situation as having no font at all.
        _VOCABULARY[key] = ()
        return _VOCABULARY[key]

    entries = []
    for codepoint in sorted(cmap):
        if not _in_range(codepoint):
            continue
        name = unicodedata.name(chr(codepoint), "")
        if not name:
            continue
        entries.append((name.lower().replace("-", " "), codepoint))
    _VOCABULARY[key] = tuple(entries)
    return _VOCABULARY[key]


def _reset_caches() -> None:
    _VOCABULARY.clear()
