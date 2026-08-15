"""Drawing the scene's overhead image underneath the trajectory.

The overhead image used to occupy its own column of the figure, alongside an
occupancy-grid trajectory panel it could not be registered against. Now that
scenes report where their image sits, it goes underneath the trajectory in the
main axes and the extra column is gone.

These tests use a synthetic scene rather than ProcTHOR or railsim, so they say
nothing about whether a *particular* scene's extent is right -- that belongs to
the providers' own tests -- only that the dashboard wires it up in the right
order, with the right limits, and degrades to a plain grid when a scene cannot
place its image.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from railroad import operators
from railroad.core import Fluent as F, State
from railroad.dashboard import PlannerDashboard
from railroad.environment import ObjectSearchEnvironment
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

    move_op = operators.construct_move_operator_blocking(lambda r, a, b: 10.0)
    no_op = operators.construct_no_op_operator(no_op_time=1.0, extra_cost=10.0)
    env = ObjectSearchEnvironment(
        state=State(0.0, {F("at r1 A"), F("free r1")}, []),
        objects_by_type={"robot": {"r1"}, "location": {"A", "B"}},
        operators=[move_op, no_op],
    )
    env.occupancy_grid = grid  # ty: ignore[unresolved-attribute]
    if scene is not None:
        env.scene = scene  # ty: ignore[unresolved-attribute]

    db = PlannerDashboard(F("at r1 B"), env, force_interactive=False, print_on_exit=False)
    db.known_robots = {"r1"}
    db._entity_positions = {"r1": [(0.0, "A", None), (10.0, "B", None)]}
    db._goal_time = 10.0
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
    def test_the_image_is_drawn_below_the_grid(self):
        db, coords = _dashboard(_Scene(_photo_view()))
        _figure, ax = _main_axes(db, coords)
        photo, grid = _layer(ax, -1), _layer(ax, 0)
        assert photo is not None and grid is not None
        assert photo.get_zorder() < grid.get_zorder()
        left, right, bottom, top = photo.get_extent()
        assert (left, right, bottom, top) == pytest.approx((
            PHOTO_EXTENT["min_x"], PHOTO_EXTENT["max_x"],
            PHOTO_EXTENT["max_y"], PHOTO_EXTENT["min_y"],
        ))

    def test_the_grid_vanishes_over_an_image_when_the_map_is_known(self):
        """With the whole map known, the image says everything the grid does."""
        db, coords = _dashboard(_Scene(_photo_view()))
        _figure, ax = _main_axes(db, coords)
        assert np.asarray(_layer(ax, 0).get_alpha()).max() == pytest.approx(0.0)

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
        # An outline, not a wash: only a thin border is painted at all.
        assert (opacity > 0).mean() < 0.2

    def test_a_fully_known_map_gets_no_outline(self):
        """Nothing to delimit: the robot has seen all of it."""
        db, coords = _dashboard(_Scene(_photo_view()))
        _figure, ax = _main_axes(db, coords)
        outline = _layer(ax, 0.5)
        assert outline is None or np.asarray(outline.get_array())[:, :, 3].max() == 0

    def test_the_grid_turns_translucent_only_over_an_image(self):
        """Opaque obstacle fill would hide the very thing being drawn under it."""
        db, coords = _dashboard(_Scene(_photo_view()))
        _figure, ax = _main_axes(db, coords)
        assert isinstance(_layer(ax, 0).get_alpha(), np.ndarray)

        db, coords = _dashboard(None)
        _figure, ax = _main_axes(db, coords)
        assert ax.get_images()[0].get_alpha() is None

    def test_the_axes_widen_to_include_the_image(self):
        """Clamping to the grid would crop off the walls it stops short of."""
        db, coords = _dashboard(_Scene(_photo_view()))
        _figure, ax = _main_axes(db, coords)
        assert ax.get_xlim() == pytest.approx(
            (PHOTO_EXTENT["min_x"], PHOTO_EXTENT["max_x"])
        )
        # Inverted, so the y axis keeps increasing downward.
        assert ax.get_ylim() == pytest.approx(
            (PHOTO_EXTENT["max_y"], PHOTO_EXTENT["min_y"])
        )

    def test_without_an_image_the_axes_frame_the_grid(self):
        db, coords = _dashboard(None)
        _figure, ax = _main_axes(db, coords)
        n_x, n_y = GRID_SHAPE
        assert ax.get_xlim() == pytest.approx((-0.5, n_x - 0.5))
        assert ax.get_ylim() == pytest.approx((n_y - 0.5, -0.5))

    @pytest.mark.parametrize("scene", [None, "photo"])
    def test_a_path_leaving_the_grid_is_not_cropped(self, scene):
        """Pinning the limits turns autoscale off, which used to be free.

        Callers may plot coordinates that do not all fall inside the occupancy
        grid; before the limits were set explicitly those simply expanded the
        view, and they must still be visible rather than silently clipped.
        """
        db, _coords = _dashboard(_Scene(_photo_view()) if scene else None)
        n_x, n_y = GRID_SHAPE
        outside = {"A": (4.0, 4.0), "B": (n_x + 25.0, n_y + 18.0)}
        _figure, ax = _main_axes(db, outside)
        assert ax.get_xlim()[1] >= outside["B"][0]
        assert ax.get_ylim()[0] >= outside["B"][1]

    def test_a_scene_that_cannot_place_its_image_gets_a_plain_grid(self):
        """The stale-ProcTHOR-cache path: no image beats a misaligned one."""
        db, coords = _dashboard(_Scene(None))
        _figure, ax = _main_axes(db, coords)
        assert len(ax.get_images()) == 1

    def test_an_unpositioned_legacy_scene_is_ignored(self):
        """``get_top_down_image`` alone says nothing about where the image sits."""

        class LegacyScene:
            def get_top_down_image(self, orthographic=True):
                return np.zeros((8, 8, 3), dtype=np.uint8)

        db, coords = _dashboard(LegacyScene())
        _figure, ax = _main_axes(db, coords)
        assert len(ax.get_images()) == 1


class TestContentFraming:
    """ProcTHOR's square camera pads a non-square house with blank skybox.

    Framing on that padding shrinks the map for nothing, so the limits follow
    the image's content instead. This only moves the *view*: the image is still
    drawn at its true extent, so a misfire crops slightly and never misplaces.
    """

    def _blank_edged_view(self):
        # Content occupies the middle half of the image, blank around it.
        image = np.full((40, 40, 3), 255, dtype=np.uint8)
        image[10:30, 10:30] = (10, 120, 200)
        return TopDownView(image=image, min_x=0.0, max_x=40.0,
                           min_y=0.0, max_y=40.0)

    def test_blank_margin_is_left_out_of_the_frame(self):
        db, coords = _dashboard(_Scene(self._blank_edged_view()))
        _figure, ax = _main_axes(db, coords)
        # Grid is 30x20, image content spans 10..30 -- union of the two.
        assert ax.get_xlim() == pytest.approx((-0.5, 30.0))
        assert ax.get_ylim() == pytest.approx((30.0, -0.5))

    def test_the_image_still_draws_at_its_full_extent(self):
        """Framing is a view choice; it must not move the pixels."""
        db, coords = _dashboard(_Scene(self._blank_edged_view()))
        _figure, ax = _main_axes(db, coords)
        photo = min(ax.get_images(), key=lambda im: im.get_zorder())
        assert photo.get_extent() == pytest.approx((0.0, 40.0, 40.0, 0.0))

    def test_an_image_with_no_blank_border_frames_unchanged(self):
        """railsim's is one pixel per cell with no skybox; it must not shrink."""
        db, coords = _dashboard(_Scene(_photo_view()))
        _figure, ax = _main_axes(db, coords)
        assert ax.get_xlim() == pytest.approx(
            (PHOTO_EXTENT["min_x"], PHOTO_EXTENT["max_x"])
        )


class TestFigureLayout:
    @pytest.mark.parametrize("scene", [None, "photo"])
    def test_there_is_no_separate_overhead_panel(self, scene):
        """Regression guard for the GridSpec collapse to main + sidebar."""
        db, coords = _dashboard(_Scene(_photo_view()) if scene else None)
        figure, _ax = _main_axes(db, coords)
        assert len(figure.axes) == 2
