"""Rasterising emoji through Pillow.

Colour emoji fonts are bitmap fonts: they render at a fixed set of strike sizes
and raise on anything else, and which sizes those are differs by font. Nothing
here may reach the network.
"""

import numpy as np
import pytest

from railroad.dashboard._sprites import fonts, resources

pytestmark = pytest.mark.skipif(
    fonts.find_font() is None, reason="no colour emoji font on this machine"
)

TEDDY_BEAR = 0x1F9F8
LATIN_A = 0x0041


@pytest.fixture
def font_path():
    return fonts.find_font()


def test_probing_reports_at_least_one_usable_strike(font_path):
    strikes = fonts.probe_strikes(font_path)
    assert strikes
    assert list(strikes) == sorted(strikes)
    assert all(isinstance(size, int) for size in strikes)


def test_rasterizing_gives_an_rgba_array_of_the_requested_size(font_path):
    glyph = fonts.rasterize(TEDDY_BEAR, 24, font_path)
    assert glyph.shape == (24, 24, 4)
    assert glyph.dtype == np.uint8
    assert glyph[..., 3].max() > 0, "nothing was drawn"


def test_a_size_above_every_strike_still_renders(font_path):
    """Noto ships a single 109px strike, so this path is not hypothetical."""
    largest = fonts.probe_strikes(font_path)[-1]
    glyph = fonts.rasterize(TEDDY_BEAR, largest + 64, font_path)
    assert glyph.shape == (largest + 64, largest + 64, 4)


def test_a_codepoint_the_font_cannot_draw_returns_none(font_path):
    """No sprite is a fine outcome; an exception in the middle of a plot is not."""
    assert fonts.rasterize(LATIN_A, 24, font_path) is None


def test_rasters_are_cached(font_path):
    first = fonts.rasterize(TEDDY_BEAR, 32, font_path)
    assert fonts.rasterize(TEDDY_BEAR, 32, font_path) is first


def test_an_explicitly_missing_font_disables_sprites(monkeypatch, tmp_path):
    """The degradation path: a warning, then no glyphs, never an exception."""
    monkeypatch.setenv("RAILROAD_EMOJI_FONT", str(tmp_path / "absent.ttf"))
    monkeypatch.setattr(fonts, "SYSTEM_FONT_PATHS", ())
    # The resources base is captured at import, so move the value, not the env var.
    monkeypatch.setattr(resources, "DEFAULT_RESOURCES_BASE", tmp_path)
    fonts._reset_caches()
    try:
        with pytest.warns(RuntimeWarning, match="No colour emoji font"):
            assert fonts.find_font() is None
        assert fonts.rasterize(TEDDY_BEAR, 24) is None
    finally:
        fonts._reset_caches()
