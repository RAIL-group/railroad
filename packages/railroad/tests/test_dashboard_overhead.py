"""Drawing the scene's overhead image underneath the trajectory.

A synthetic scene throughout, so these say nothing about whether a *particular*
scene's extent is right -- that belongs to the providers' own tests -- only
that the dashboard wires it up in the right order, with the right limits, and
degrades to a plain grid when a scene cannot place its image.
"""

import matplotlib

from env_helpers import move_dashboard, move_env
matplotlib.use("Agg")

import numpy as np
import pytest

from railroad.environment.types import TopDownView
from railroad.navigation.constants import UNOBSERVED_VAL

GRID_SHAPE = (30, 20)
# Deliberately wider than the grid, as ProcTHOR's is: its grid spans only the
# agent-reachable bbox, while the camera frames the whole house.
PHOTO_EXTENT = dict(min_x=-6.0, max_x=35.0, min_y=-4.0, max_y=23.0)


class _Scene:
    """A scene that reports an overhead image, or declines to."""

    def __init__(self, view):
        self._view = view

    def get_top_down_view(self):
        return self._view


def _photo_view():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :] = (10, 120, 200)
    return TopDownView(image=image, **PHOTO_EXTENT)


def _dashboard(scene, *, unobserved: bool = False):
    grid = np.zeros(GRID_SHAPE)
    grid[0, :] = 1.0
    grid[-1, :] = 1.0
    if unobserved:
        # Mid-exploration: a band the robot has not looked at yet.
        grid[:, GRID_SHAPE[1] // 2:] = UNOBSERVED_VAL

    db = move_dashboard(move_env(occupancy_grid=grid, scene=scene))
    return db, {"A": (4.0, 4.0), "B": (25.0, 15.0)}


def _main_axes(db, coords):
    figure = db._render_static_plot((10, 6), location_coords=coords)
    assert figure is not None
    return figure, figure.axes[0]


def _layer(ax, zorder):
    """The image drawn at *zorder*: -1 photo, 0 grid, 0.5 known-region outline."""
    matches = [im for im in ax.get_images() if im.get_zorder() == zorder]
    assert len(matches) <= 1, f"{len(matches)} images at zorder {zorder}"
    return matches[0] if matches else None


class TestPhotoUnderlay:
    @pytest.mark.parametrize("scene", [None, "photo"])
    def test_the_figure_is_just_the_main_axes_and_a_sidebar(self, scene):
        """The image moved under the trajectory, so its own column is gone."""
        db, coords = _dashboard(_Scene(_photo_view()) if scene else None)
        figure, _ax = _main_axes(db, coords)
        assert len(figure.axes) == 2

    def test_a_known_map_gets_the_image_alone_beneath_the_trajectory(self):
        """Nothing for the grid or the outline to add once it is all observed.

        The image still has to sit *under* the grid rather than over it, or it
        would hide the very thing it is drawn beneath.
        """
        db, coords = _dashboard(_Scene(_photo_view()))
        _figure, ax = _main_axes(db, coords)

        photo, grid = _layer(ax, -1), _layer(ax, 0)
        assert photo is not None and grid is not None
        assert photo.get_zorder() < grid.get_zorder()
        assert photo.get_extent() == pytest.approx((
            PHOTO_EXTENT["min_x"], PHOTO_EXTENT["max_x"],
            PHOTO_EXTENT["max_y"], PHOTO_EXTENT["min_y"],
        ))

        assert np.asarray(grid.get_alpha()).max() == pytest.approx(0.0)
        outline = _layer(ax, 0.5)
        assert outline is None or np.asarray(outline.get_array())[:, :, 3].max() == 0

        # Without an image the grid is opaque again, since it is all there is.
        db, coords = _dashboard(None)
        _figure, ax = _main_axes(db, coords)
        assert _layer(ax, 0).get_alpha() is None

    def test_how_far_the_robot_has_seen_is_outlined_not_shaded(self):
        """The one thing an image cannot show is where the robot has not looked.

        It renders ground truth everywhere, so the extent of what has been
        observed is drawn -- but as an outline, since shading the unobserved
        region buries the map the image is there to show.
        """
        db, coords = _dashboard(_Scene(_photo_view()), unobserved=True)
        _figure, ax = _main_axes(db, coords)
        outline = _layer(ax, 0.5)
        assert outline is not None, "no known-region outline drawn"
        opacity = np.asarray(outline.get_array())[:, :, 3]
        assert opacity.max() > 0.5, "outline is invisible"
        assert (opacity > 0).mean() < 0.2, "outline has become a wash"

    def test_the_frame_covers_the_grid_and_the_image_s_content(self):
        """Not the image's full extent: ProcTHOR's square camera pads a
        non-square house with blank skybox, and framing on that padding shrinks
        the map for nothing.

        The image still *draws* at its full extent -- only the view is trimmed,
        so a misfire here crops slightly and can never misplace anything.
        """
        # Content in the middle half of the image, blank around it.
        image = np.full((40, 40, 3), 255, dtype=np.uint8)
        image[10:30, 10:30] = (10, 120, 200)
        view = TopDownView(image=image, min_x=0.0, max_x=40.0, min_y=0.0, max_y=40.0)

        db, coords = _dashboard(_Scene(view))
        _figure, ax = _main_axes(db, coords)
        n_x, n_y = GRID_SHAPE
        assert ax.get_xlim() == pytest.approx((-0.5, 30.0))
        assert ax.get_ylim() == pytest.approx((30.0, -0.5))
        assert _layer(ax, -1).get_extent() == pytest.approx((0.0, 40.0, 40.0, 0.0))

        # An image with no blank border -- railsim's, one pixel per cell --
        # must not be trimmed at all.
        db, coords = _dashboard(_Scene(_photo_view()))
        _figure, ax = _main_axes(db, coords)
        assert ax.get_xlim() == pytest.approx(
            (PHOTO_EXTENT["min_x"], PHOTO_EXTENT["max_x"])
        )
        # Inverted, so the y axis keeps increasing downward.
        assert ax.get_ylim() == pytest.approx(
            (PHOTO_EXTENT["max_y"], PHOTO_EXTENT["min_y"])
        )

        # And with no image at all, the grid alone sets the frame.
        db, coords = _dashboard(None)
        _figure, ax = _main_axes(db, coords)
        assert ax.get_xlim() == pytest.approx((-0.5, n_x - 0.5))
        assert ax.get_ylim() == pytest.approx((n_y - 0.5, -0.5))

    @pytest.mark.parametrize("scene", [None, "photo"])
    def test_what_a_caller_plots_outside_the_grid_is_not_cropped(self, scene):
        """Pinning the limits turns autoscale off, which used to be free.

        ``plot_trajectories`` explicitly supports caller-supplied
        ``location_coords``, which need not all fall inside the occupancy grid.
        Before the limits were set explicitly those simply expanded the view;
        they must still be visible rather than silently clipped -- the marker
        and its label as much as the path reaching it.
        """
        db, _coords = _dashboard(_Scene(_photo_view()) if scene else None)
        n_x, n_y = GRID_SHAPE
        outside = {"A": (4.0, 4.0), "B": (n_x + 25.0, n_y + 18.0)}
        _figure, ax = _main_axes(db, outside)
        assert ax.get_xlim()[1] >= outside["B"][0]
        assert ax.get_ylim()[0] >= outside["B"][1]

        # Every black location marker inside the frame, not just the trail.
        left, right = ax.get_xlim()
        bottom, top = ax.get_ylim()
        for collection in ax.collections:
            for x, y in collection.get_offsets():
                assert left <= x <= right and top <= y <= bottom

    @pytest.mark.parametrize("scene", ["declines", "legacy"])
    def test_a_scene_that_cannot_place_an_image_gets_a_plain_grid(self, scene):
        """A stale ProcTHOR cache declines; an older provider offers only
        ``get_top_down_image``, which says nothing about where the image sits.
        Either way an unplaceable image is worse than none."""

        class LegacyScene:
            def get_top_down_image(self, orthographic=True):
                return np.zeros((8, 8, 3), dtype=np.uint8)

        provider = _Scene(None) if scene == "declines" else LegacyScene()
        db, coords = _dashboard(provider)
        _figure, ax = _main_axes(db, coords)
        assert len(ax.get_images()) == 1
