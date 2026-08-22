import io

import numpy as np
import pytest

from railroad.plotting import emoji

# This module repoints SYSTEM_FONT_PATHS / DEFAULT_RESOURCES_BASE, so it needs
# the cache reset around every test. The fixture is not autouse -- see conftest.
pytestmark = pytest.mark.usefixtures("reset_emoji_caches")

TEDDY_BEAR = 0x1F9F8


@pytest.fixture
def font_path():
    path = emoji.find_font()
    if path is None:
        pytest.skip("no color emoji font on this machine")
    return path


def test_rasterize_returns_a_cached_rgba_square(font_path):
    glyph = emoji.rasterize(TEDDY_BEAR, 24, font_path)
    assert glyph.shape == (24, 24, 4)
    assert glyph.dtype == np.uint8
    assert glyph[..., 3].max() > 0
    assert emoji.rasterize(TEDDY_BEAR, 24, font_path) is glyph


def test_rasterize_scales_above_the_largest_strike(font_path):
    size = emoji.probe_strikes(font_path)[-1] + 32
    assert emoji.rasterize(TEDDY_BEAR, size, font_path).shape == (size, size, 4)


def test_rasterized_glyph_is_whole_and_centred(font_path):
    alpha = emoji.rasterize(TEDDY_BEAR, 64, font_path)[..., 3]
    rows = np.nonzero(alpha.any(axis=1))[0]
    cols = np.nonzero(alpha.any(axis=0))[0]
    # A colour strike's ink sits above the vertical midpoint its anchor
    # centres on, so framing on the em box lost the top rows off the raster.
    # Cropping to the ink leaves it whole, centred, and touching one pair of
    # edges exactly.
    assert rows[0] == pytest.approx(63 - rows[-1], abs=1)
    assert cols[0] == pytest.approx(63 - cols[-1], abs=1)
    assert max(rows[-1] - rows[0], cols[-1] - cols[0]) == 63


def test_missing_codepoint_returns_none(font_path):
    assert emoji.rasterize(0x0041, 24, font_path) is None


def test_font_lookup_does_not_create_the_resource_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("RAILROAD_EMOJI_FONT", str(tmp_path / "absent.ttf"))
    monkeypatch.setattr(emoji, "SYSTEM_FONT_PATHS", ())
    monkeypatch.setattr(emoji, "DEFAULT_RESOURCES_BASE", tmp_path / "missing")
    assert emoji.find_font() is None
    assert not (tmp_path / "missing").exists()


def test_download_is_visible_after_an_initial_miss(monkeypatch, tmp_path):
    monkeypatch.delenv("RAILROAD_EMOJI_FONT", raising=False)
    monkeypatch.setattr(emoji, "SYSTEM_FONT_PATHS", ())
    monkeypatch.setattr(emoji, "DEFAULT_RESOURCES_BASE", tmp_path)
    monkeypatch.setattr(
        emoji.urllib.request,
        "urlopen",
        lambda _url, timeout=None: io.BytesIO(b"font"),
    )
    assert emoji.find_font() is None
    downloaded = emoji.ensure_emoji_font()
    assert emoji.find_font() == downloaded


@pytest.mark.parametrize(
    "value, enabled", [("1", True), ("0", False), ("false", False)]
)
def test_sprite_resource_switch(monkeypatch, value, enabled):
    monkeypatch.setenv("RAILROAD_OBJECT_SPRITES", value)
    assert emoji.object_sprites_enabled() is enabled


@pytest.mark.parametrize("enabled", [True, False])
def test_procthor_provisions_enabled_sprite_resources(monkeypatch, tmp_path, enabled):
    monkeypatch.setenv("PROCTHOR_AUTO_DOWNLOAD", "0")
    from railroad.environment.procthor import resources as procthor_resources

    for name in (
        "ensure_procthor_10k",
        "ensure_sbert_model",
        "ensure_ai2thor_simulator",
    ):
        monkeypatch.setattr(procthor_resources, name, lambda **_kw: None)
    calls = []
    monkeypatch.setattr(emoji, "object_sprites_enabled", lambda: enabled)
    monkeypatch.setattr(emoji, "ensure_emoji_resources", lambda **kw: calls.append(kw))

    procthor_resources.ensure_all_resources(base_dir=tmp_path, force=True)
    assert bool(calls) is enabled
