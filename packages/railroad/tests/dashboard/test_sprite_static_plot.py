"""Object glyphs on the static trajectory plot."""

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox  # noqa: E402

from railroad.dashboard import _sprites  # noqa: E402


@pytest.fixture
def flat_glyph(monkeypatch):
    """A solid block instead of a real emoji, so no font or model is needed."""
    import numpy as np

    block = np.zeros((16, 16, 4), dtype=np.uint8)
    block[..., 0] = 255
    block[..., 3] = 255

    class _Provider:
        def glyph_for(self, name):
            return block

    monkeypatch.setattr(_sprites, "get_glyph_provider", lambda **_kw: _Provider())
    return block


def _sprites_on(ax):
    return [a for a in ax.artists if isinstance(a, AnnotationBbox)]


def test_only_plan_relevant_objects_are_drawn(fetch_dashboard, location_coords, flat_glyph):
    """The search also reveals a sponge; it is not what the plan is about."""
    _fig, ax = plt.subplots()
    fetch_dashboard.plot_trajectories(ax=ax, location_coords=location_coords)
    assert len(_sprites_on(ax)) == 1
    plt.close("all")


def test_the_sprite_sits_above_the_trail(fetch_dashboard, location_coords, flat_glyph):
    _fig, ax = plt.subplots()
    fetch_dashboard.plot_trajectories(ax=ax, location_coords=location_coords)
    (sprite,) = _sprites_on(ax)
    assert sprite.get_zorder() > 6
    plt.close("all")


def test_sprites_stay_out_of_the_legend(fetch_dashboard, location_coords, flat_glyph):
    """Axes.legend collects labelled artists; a glyph there would be nonsense."""
    _fig, ax = plt.subplots()
    fetch_dashboard.plot_trajectories(ax=ax, location_coords=location_coords)
    (sprite,) = _sprites_on(ax)
    assert not sprite.get_label()
    plt.close("all")


def test_the_final_position_is_the_place_destination(
    fetch_dashboard, location_coords, flat_glyph
):
    """The static plot shows the end of the plan, so the mug is on the counter."""
    _fig, ax = plt.subplots()
    fetch_dashboard.plot_trajectories(ax=ax, location_coords=location_coords)
    (sprite,) = _sprites_on(ax)
    x, y = sprite.xybox
    assert abs(x - location_coords["counter"][0]) < 2.0
    assert abs(y - location_coords["counter"][1]) < 2.0
    plt.close("all")


def test_sprites_are_inside_the_framing(fetch_dashboard, location_coords, flat_glyph):
    """Limits are pinned at the end of plotting, and a clipped sprite vanishes."""
    _fig, ax = plt.subplots()
    fetch_dashboard.plot_trajectories(ax=ax, location_coords=location_coords)
    (sprite,) = _sprites_on(ax)
    x, y = sprite.xybox
    low_x, high_x = sorted(ax.get_xlim())
    low_y, high_y = sorted(ax.get_ylim())
    assert low_x <= x <= high_x
    assert low_y <= y <= high_y
    plt.close("all")


def test_glyph_objects_overrides_the_selection(
    fetch_dashboard, location_coords, flat_glyph
):
    _fig, ax = plt.subplots()
    fetch_dashboard.plot_trajectories(
        ax=ax, location_coords=location_coords, glyph_objects=["mug", "sponge"],
    )
    assert len(_sprites_on(ax)) == 2
    plt.close("all")


def test_sprites_can_be_turned_off(fetch_dashboard, location_coords, flat_glyph):
    _fig, ax = plt.subplots()
    fetch_dashboard.plot_trajectories(
        ax=ax, location_coords=location_coords, object_sprites=False,
    )
    assert _sprites_on(ax) == []
    plt.close("all")


def test_no_glyph_source_leaves_the_plot_unchanged(
    fetch_dashboard, location_coords, monkeypatch
):
    """The degradation path, which is what makes on-by-default safe."""
    monkeypatch.setattr(_sprites, "get_glyph_provider", lambda **_kw: None)
    _fig, ax = plt.subplots()
    fetch_dashboard.plot_trajectories(ax=ax, location_coords=location_coords)
    assert _sprites_on(ax) == []
    plt.close("all")
