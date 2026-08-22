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


def test_integer_location_coordinates_are_accepted(fetch_dashboard, flat_glyph):
    _figure, ax = plt.subplots()
    fetch_dashboard.plot_trajectories(
        ax=ax, location_coords={"shelf": (2, 2), "counter": (8, 6)}
    )
    assert len(sprites_on(ax)) == 1
    plt.close("all")


def test_the_environment_switch_disables_sprites(
    fetch_dashboard, location_coords, flat_glyph, monkeypatch
):
    monkeypatch.setenv("RAILROAD_OBJECT_SPRITES", "0")
    _figure, ax = plt.subplots()
    fetch_dashboard.plot_trajectories(ax=ax, location_coords=location_coords)
    assert not sprites_on(ax)
    plt.close("all")


def test_a_failing_glyph_source_costs_only_the_glyphs(
    fetch_dashboard, location_coords, monkeypatch
):
    def explode(**_kw):
        raise OSError("read-only resources directory")

    monkeypatch.setattr(_sprites, "get_glyph_provider", explode)
    _figure, axes = plt.subplots()
    ax = fetch_dashboard.plot_trajectories(ax=axes, location_coords=location_coords)
    assert not sprites_on(ax)
    assert ax.collections, "the trail is what the caller actually asked for"
    plt.close("all")


def test_sprite_extent_is_read_off_the_axes(fetch_dashboard):
    _figure, ax = plt.subplots()
    ax.set_xlim(0, 10)
    narrow = _sprites.sprite_extent(ax)
    ax.set_xlim(0, 100)
    assert _sprites.sprite_extent(ax) == pytest.approx(10 * narrow)
    plt.close("all")


def test_plot_image_forwards_the_sprite_controls(
    fetch_dashboard, location_coords, flat_glyph, monkeypatch
):
    seen: dict = {}
    original = fetch_dashboard._build_sprites

    def record(ax, env_coords, **kwargs):
        seen.update(kwargs)
        return original(ax, env_coords, **kwargs)

    monkeypatch.setattr(fetch_dashboard, "_build_sprites", record)
    assert fetch_dashboard.get_plot_image(
        location_coords=location_coords, glyph_objects=["sponge"],
    )
    assert seen == {"glyph_objects": ["sponge"], "object_sprites": True}


@pytest.mark.parametrize("found_time", [0.0, 50.0, 99.0, 99.5, 100.0])
def test_the_fade_always_completes_by_the_end(fetch_dashboard, found_time):
    """Frame 0 posters the final instant, so the real one must match it."""
    t_end = 100.0
    line = _sprites.ObjectTimeline(
        "mug", found_time, (_sprites.Anchor(0.0, "rest", xy=(0.0, 0.0), loc="shelf"),)
    )
    _positions, alphas = _sprites.sample(
        line, np.array([t_end]), {},
        fade=fetch_dashboard._sprite_fade(found_time, t_end),
    )
    assert alphas[-1] == 1.0
