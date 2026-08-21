"""Emoji sprites for the objects a plan is about.

Split so that everything the trajectory plots need for *placement* --
:mod:`timeline` and :mod:`layout` -- depends on nothing but numpy, while turning
a name into pixels sits behind a single seam. A machine with no colour emoji
font renders exactly what it did before this package existed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Anchor",
    "GlyphProvider",
    "ObjectTimeline",
    "assign_slots",
    "build_timelines",
    "fan_offset",
    "get_glyph_provider",
    "is_available",
    "ring_radius",
    "sample",
    "select_objects",
]

from .layout import assign_slots, fan_offset, ring_radius
from .timeline import (
    Anchor,
    ObjectTimeline,
    build_timelines,
    sample,
    select_objects,
)

SPRITE_PX = 64
"""Raster size. Sprites are scaled to points when placed, so this only sets how
much detail survives; 64 is the largest Apple Color Emoji strike below the point
where the extra pixels stop showing."""


@runtime_checkable
class GlyphProvider(Protocol):
    """Turns an object name into an RGBA array, or ``None`` for no sprite.

    The whole optional-dependency surface of this package narrows to here, which
    is what lets the plotting tests exercise the compositor with a flat coloured
    square and no font, model or network.
    """

    def glyph_for(self, name: str) -> Any: ...


class _EmojiGlyphProvider:
    def __init__(self, font_path: Path, size_px: int = SPRITE_PX) -> None:
        self._font_path = font_path
        self._size_px = size_px
        self._cache: dict[str, Any] = {}

    def glyph_for(self, name: str) -> Any:
        if name not in self._cache:
            from . import fonts, matching

            codepoint = matching.match(name, self._font_path)
            glyph = fonts.rasterize(codepoint, self._size_px, self._font_path)
            if glyph is None:
                # A cmap entry is not a promise of pixels: a few codepoints draw
                # only as part of a sequence and come back blank on their own.
                # Better a generic box than an object that quietly disappears.
                glyph = fonts.rasterize(
                    matching.FALLBACK_CODEPOINT, self._size_px, self._font_path
                )
            self._cache[name] = glyph
        return self._cache[name]


def is_available() -> bool:
    """Whether object glyphs can be drawn at all on this machine."""
    from . import fonts

    return fonts.find_font() is not None


def get_glyph_provider(
    *, download: bool = False, size_px: int = SPRITE_PX
) -> GlyphProvider | None:
    """A provider, or ``None`` when no colour emoji font can be found.

    Lookup only unless *download* is set: auto-enabling sprites must never turn
    a call to ``plot_trajectories`` into a ten-megabyte fetch.
    """
    from . import fonts

    font_path = fonts.find_font(download=download)
    if font_path is None:
        return None
    return _EmojiGlyphProvider(font_path, size_px)
