import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox  # noqa: E402

from railroad.plotting import sprites as _sprites  # noqa: E402


@pytest.fixture
def flat_glyph(monkeypatch):
    block = np.zeros((16, 16, 4), dtype=np.uint8)
    block[..., (0, 3)] = 255

    class Provider:
        def glyph_for(self, _name):
            return block

    monkeypatch.setattr(_sprites, "get_glyph_provider", lambda **_kw: Provider())


def sprites_on(ax):
    return [artist for artist in ax.artists if isinstance(artist, AnnotationBbox)]


def test_static_sprite_is_relevant_placed_and_framed(
    fetch_dashboard, location_coords, flat_glyph
):
    _figure, ax = plt.subplots()
    fetch_dashboard.plot_trajectories(ax=ax, location_coords=location_coords)
    (sprite,) = sprites_on(ax)
    x, y = sprite.xybox
    assert sprite.get_zorder() > 6
    assert not sprite.get_label()
    assert abs(x - location_coords["counter"][0]) < 2
    assert abs(y - location_coords["counter"][1]) < 2
    assert min(ax.get_xlim()) <= x <= max(ax.get_xlim())
    assert min(ax.get_ylim()) <= y <= max(ax.get_ylim())
    plt.close("all")


def test_object_override_draws_incidental_objects(
    fetch_dashboard, location_coords, flat_glyph
):
    _figure, ax = plt.subplots()
    fetch_dashboard.plot_trajectories(
        ax=ax, location_coords=location_coords, glyph_objects=["mug", "sponge"]
    )
    assert len(sprites_on(ax)) == 2
    plt.close("all")


def test_sprites_can_be_disabled(fetch_dashboard, location_coords, flat_glyph):
    _figure, ax = plt.subplots()
    fetch_dashboard.plot_trajectories(
        ax=ax, location_coords=location_coords, object_sprites=False
    )
    assert not sprites_on(ax)
    plt.close("all")


def test_missing_provider_is_a_noop(fetch_dashboard, location_coords, monkeypatch):
    monkeypatch.setattr(_sprites, "get_glyph_provider", lambda **_kw: None)
    _figure, ax = plt.subplots()
    fetch_dashboard.plot_trajectories(ax=ax, location_coords=location_coords)
    assert not sprites_on(ax)
    plt.close("all")


def test_object_found_at_the_endpoint_is_visible(
    fetch_dashboard, location_coords, flat_glyph
):
    fetch_dashboard._goal_time = 10.0
    _figure, ax = plt.subplots()
    fetch_dashboard.plot_trajectories(ax=ax, location_coords=location_coords)
    assert len(sprites_on(ax)) == 1
    plt.close("all")
