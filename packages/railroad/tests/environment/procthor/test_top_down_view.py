"""Placing the ProcTHOR overhead image on the occupancy grid.

The scene's top-down image used to live in its own subplot, because nothing
downstream knew where it sat in world space: the camera pose and
``orthographicSize`` were computed at render time and discarded, and the pickle
cache kept only the pixels. Every normal run is a cache hit with no controller,
so there was nothing left to ask.

The camera parameters are recorded now, in meters, and converted to grid cells
on the way out. These tests pin the conversion -- especially the half-cell
convention and the ``origin="upper"`` tuple ordering, which are where this kind
of thing silently goes half a cell wrong.
"""

import pickle
import warnings

import numpy as np
import pytest

from railroad.environment.procthor import resources
from railroad.environment.procthor.thor_interface import (
    SCENE_CACHE_VERSION,
    TOP_DOWN_MARGIN_M,
    ThorInterface,
    _EXTENT_WARNED,
)


def _room(corners):
    """A scene-JSON room with the given (x, z) floor polygon."""
    return {"floorPolygon": [{"x": x, "y": 0.0, "z": z} for x, z in corners]}

RESOLUTION = 0.05
GRID_SHAPE = (69, 47)


@pytest.fixture(autouse=True)
def _forget_warned_seeds():
    """The once-per-seed warning guard is module state; keep tests independent."""
    _EXTENT_WARNED.clear()
    yield
    _EXTENT_WARNED.clear()


def _interface(
    *, extent_m=(0.0, 1.0, 0.0, 1.0), offset=(0.0, 0.0), resolution=RESOLUTION,
    image=None, include_extent=True, seed=7,
) -> ThorInterface:
    """A cache-backed ThorInterface without ``__init__``, which loads a scene."""
    thor = object.__new__(ThorInterface)
    thor.seed = seed
    thor.grid_resolution = resolution
    thor.grid_offset = np.array(offset)
    thor.controller = None
    # Rooms whose footprint is exactly extent_m, so the cache reads as current
    # unless a test deliberately makes it disagree.
    min_x, max_x, min_z, max_z = extent_m
    inset = TOP_DOWN_MARGIN_M
    thor.rooms = [_room([
        (min_x + inset, min_z + inset), (max_x - inset, min_z + inset),
        (max_x - inset, max_z - inset), (min_x + inset, max_z - inset),
    ])]
    cached = {
        'cache_version': SCENE_CACHE_VERSION,
        'reachable_positions': [],
        'image_ortho': np.zeros((4, 4, 3), dtype=np.uint8) if image is None else image,
        'image_persp': None,
    }
    if include_extent:
        cached['image_ortho_extent_m'] = extent_m
    thor.cached_data = cached
    return thor


def _grid_covering_extent(offset, resolution, shape):
    """The world footprint an image covering exactly *shape* cells would have.

    Cell k spans ``k - 0.5`` to ``k + 0.5``, so the outer edges of an n-cell
    span sit half a cell beyond the first and last cell centers.
    """
    n_x, n_y = shape
    return (
        offset[0] - 0.5 * resolution,
        offset[0] + (n_x - 0.5) * resolution,
        offset[1] - 0.5 * resolution,
        offset[1] + (n_y - 0.5) * resolution,
    )


class TestExtentMath:
    """Meters -> grid cells, independent of where the footprint came from."""

    @staticmethod
    def _place(extent_m, *, offset=(0.0, 0.0), resolution=RESOLUTION):
        thor = object.__new__(ThorInterface)
        thor.grid_resolution = resolution
        thor.grid_offset = np.array(offset)
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        return thor._view_from_extent(image, extent_m)

    def test_an_image_covering_the_grid_gets_the_grid_s_own_extent(self):
        """The anchor test: half-cell convention, ordering and y flip at once.

        An image whose world footprint is exactly the grid's footprint must
        land on the extent matplotlib infers for the grid itself,
        ``(-0.5, n_x - 0.5, n_y - 0.5, -0.5)``. Any half-cell slip, a swapped
        axis, or a flipped y shows up here.
        """
        offset = (0.0, 0.0)
        view = self._place(
            _grid_covering_extent(offset, RESOLUTION, GRID_SHAPE), offset=offset,
        )
        n_x, n_y = GRID_SHAPE
        assert view.extent == pytest.approx((-0.5, n_x - 0.5, n_y - 0.5, -0.5))

    def test_the_same_anchor_holds_with_a_negative_grid_offset(self):
        """Scenes with ``min_x < 0`` take a different branch when the grid is built."""
        offset = (-1.35, -0.4)
        view = self._place(
            _grid_covering_extent(offset, RESOLUTION, GRID_SHAPE), offset=offset,
        )
        n_x, n_y = GRID_SHAPE
        assert view.extent == pytest.approx((-0.5, n_x - 0.5, n_y - 0.5, -0.5))

    def test_bottom_is_below_top_so_y_keeps_increasing_downward(self):
        """``origin="upper"`` draws row 0 at *top*, and row 0 is min z."""
        left, right, bottom, top = self._place((0.0, 2.0, 0.0, 3.0)).extent
        assert left < right
        assert bottom > top

    def test_the_cached_extent_is_meters_so_resolution_rescales_it(self):
        """The cache filename encodes no resolution; cells would corrupt this."""
        extent_m = (0.0, 4.0, 0.0, 4.0)
        coarse = self._place(extent_m, resolution=0.10)
        fine = self._place(extent_m, resolution=0.05)
        # Half the cell size covers the same meters in twice as many cells.
        assert fine.max_x == pytest.approx(2 * coarse.max_x)
        assert fine.max_y == pytest.approx(2 * coarse.max_y)


class TestGridCoordinateHelpers:
    def test_grid_to_world_inverts_scale_to_grid(self):
        thor = _interface(offset=(-1.35, 0.4))
        for cell in [(0, 0), (12, 45), (68, 3)]:
            assert thor.scale_to_grid(thor.grid_to_world(cell)) == cell

    def test_a_world_point_round_trips_within_half_a_cell(self):
        thor = _interface(offset=(-1.35, 0.4))
        point = (0.717, -0.233)
        back = thor.grid_to_world(thor.scale_to_grid(point))
        assert back == pytest.approx(point, abs=RESOLUTION / 2)

    def test_scale_to_grid_still_returns_plain_ints(self):
        """It keys ``g2p_map``; numpy scalars there would break dict lookups."""
        cell = _interface(offset=(-1.35, 0.4)).scale_to_grid((0.7, -0.2))
        assert all(isinstance(c, int) and not isinstance(c, np.integer) for c in cell)

    def test_the_continuous_form_is_what_scale_to_grid_rounds(self):
        thor = _interface(offset=(-1.35, 0.4))
        point = (0.717, -0.233)
        exact = thor.scale_to_grid_continuous(point)
        assert thor.scale_to_grid(point) == (round(exact[0]), round(exact[1]))


class TestStaleCache:
    def test_a_current_cache_is_used(self):
        thor = _interface()
        view = thor.get_top_down_view()
        assert view is not None

    def test_a_cache_without_an_extent_yields_no_view(self):
        """Better no image than one that can only be drawn misaligned."""
        thor = _interface(include_extent=False)
        with pytest.warns(RuntimeWarning, match="cached before the top-down camera"):
            assert thor.get_top_down_view() is None

    def test_a_cache_framed_differently_is_rejected(self):
        """The footprint is recomputable, so a disagreeing cache is stale.

        This is what catches a changed ``TOP_DOWN_MARGIN_M``: the pixels on
        disk cover a different patch of world than the current build would
        frame, and drawing them anyway would misplace everything.
        """
        thor = _interface()
        cache = thor.cached_data
        assert cache is not None
        cache['image_ortho_extent_m'] = tuple(
            v + 1.0 for v in cache['image_ortho_extent_m']
        )
        with pytest.warns(RuntimeWarning, match="was rendered with the footprint"):
            assert thor.get_top_down_view() is None

    def test_the_warning_names_the_seed_and_the_cache_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(resources, "DEFAULT_RESOURCES_BASE", tmp_path / "elsewhere")
        thor = _interface(include_extent=False, seed=4001)
        with pytest.warns(RuntimeWarning) as caught:
            thor.get_top_down_view()
        message = str(caught[0].message)
        assert "4001" in message
        assert str(tmp_path / "elsewhere") in message

    def test_it_warns_once_per_seed_however_often_it_is_asked(self):
        """save_video can reach this per frame; one line is the whole point."""
        thor = _interface(include_extent=False)
        with pytest.warns(RuntimeWarning):
            thor.get_top_down_view()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            thor.get_top_down_view()
        assert caught == []


class _FakeEvent:
    def __init__(self, metadata, frames=()):
        self.metadata = metadata
        self.third_party_camera_frames = list(frames)


class _FakeController:
    """Stands in for Unity, recording the pose the camera was actually given."""

    def __init__(self, *, rotation, scene_size=(4.0, 3.0), frame_shape=(480, 480)):
        self._rotation = rotation
        self._scene_size = scene_size
        self._frame_shape = frame_shape
        self.camera_pose: dict = {}

    def step(self, action, **kwargs):
        if action == "GetMapViewCameraProperties":
            return _FakeEvent({
                "actionReturn": {
                    "position": {"x": 1.5, "y": 2.0, "z": -0.5},
                    "rotation": dict(self._rotation),
                    "orthographicSize": 1.0,
                },
                "sceneBounds": {
                    "size": {"x": self._scene_size[0], "y": 3.0, "z": self._scene_size[1]},
                },
            })
        assert action == "AddThirdPartyCamera"
        kwargs.pop("skyboxColor", None)
        kwargs.pop("raise_for_failure", None)
        self.camera_pose = kwargs
        height, width = self._frame_shape
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        return _FakeEvent({}, frames=[frame])


def _controller_backed(controller, seed=5) -> ThorInterface:
    thor = object.__new__(ThorInterface)
    thor.seed = seed
    thor.controller = controller
    thor.cached_data = None
    thor.grid_resolution = RESOLUTION
    thor.grid_offset = np.array((0.0, 0.0))
    thor.rooms = [_room([(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)])]
    return thor


class TestFootprint:
    """The framing is ours, computed from the scene JSON, not THOR's.

    AI2-THOR frames its map view on ``sceneBounds`` -- the union of every
    enabled renderer -- which includes geometry invisible from above and is
    inconsistent enough across ProcTHOR-10k to be an open upstream bug
    (allenai/ai2thor#1181). Choosing the footprint here is what makes it
    knowable without a live controller.
    """

    def test_the_footprint_squares_up_the_room_polygons_plus_a_margin(self):
        thor = object.__new__(ThorInterface)
        thor.rooms = [_room([(1.0, 2.0), (9.0, 2.0), (9.0, 6.0), (1.0, 6.0)])]
        min_x, max_x, min_z, max_z = thor.top_down_footprint()
        half = 0.5 * 8.0 + TOP_DOWN_MARGIN_M  # the longer span wins
        assert (min_x, max_x) == pytest.approx((5.0 - half, 5.0 + half))
        assert (min_z, max_z) == pytest.approx((4.0 - half, 4.0 + half))

    def test_it_covers_rooms_the_agent_cannot_reach(self):
        """Framing on reachable space would crop a sealed room out entirely."""
        thor = object.__new__(ThorInterface)
        thor.rooms = [
            _room([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]),
            _room([(0.0, 4.0), (4.0, 4.0), (4.0, 9.0), (0.0, 9.0)]),
        ]
        min_x, max_x, min_z, max_z = thor.top_down_footprint()
        assert min_z <= 0.0 and max_z >= 9.0
        assert min_x <= 0.0 and max_x >= 4.0


class TestCameraSetup:
    def test_the_camera_is_pinned_straight_down(self):
        controller = _FakeController(rotation={"x": 90.0, "y": 0.0, "z": 0.0})
        _controller_backed(controller)._render_top_down_from_controller()
        assert controller.camera_pose["rotation"] == {"x": 90.0, "y": 0.0, "z": 0.0}

    def test_a_yawed_map_view_is_overridden_and_reported(self):
        """A yaw would skew the footprint into something no extent can express."""
        controller = _FakeController(rotation={"x": 90.0, "y": 35.0, "z": 0.0})
        thor = _controller_backed(controller, seed=4001)
        with pytest.warns(RuntimeWarning, match="rotation"):
            thor._render_top_down_from_controller()
        assert controller.camera_pose["rotation"] == {"x": 90.0, "y": 0.0, "z": 0.0}

    def test_the_camera_is_placed_on_our_footprint_not_thor_s(self):
        """The reported extent must be exactly what the camera was told to frame."""
        controller = _FakeController(rotation={"x": 90.0, "y": 0.0, "z": 0.0})
        thor = _controller_backed(controller)
        _image, extent = thor._render_top_down_from_controller()

        assert extent is not None
        assert extent == pytest.approx(thor.top_down_footprint())
        min_x, max_x, min_z, max_z = extent
        pose = controller.camera_pose
        assert pose["position"]["x"] == pytest.approx(0.5 * (min_x + max_x))
        assert pose["position"]["z"] == pytest.approx(0.5 * (min_z + max_z))
        assert pose["orthographicSize"] == pytest.approx(0.5 * (max_z - min_z))
        # THOR's own framing must not leak into the horizontal placement.
        assert pose["position"]["x"] != pytest.approx(1.5)

    def test_the_scene_bounds_still_set_the_camera_height(self):
        """Height cannot skew an orthographic footprint, so THOR's value is fine."""
        controller = _FakeController(rotation={"x": 90.0, "y": 0.0, "z": 0.0},
                                     scene_size=(4.0, 3.0))
        _controller_backed(controller)._render_top_down_from_controller()
        assert controller.camera_pose["position"]["y"] == pytest.approx(2.0 + 1.1 * 4.0)

    def test_a_non_square_frame_is_refused_rather_than_skewed(self):
        """A square footprint is only right for a square frame."""
        controller = _FakeController(rotation={"x": 90.0, "y": 0.0, "z": 0.0},
                                     frame_shape=(480, 640))
        with pytest.raises(RuntimeError, match="square"):
            _controller_backed(controller)._render_top_down_from_controller()

    def test_the_perspective_render_reports_no_footprint(self):
        """A projective view has no rectangular footprint to report."""
        controller = _FakeController(rotation={"x": 90.0, "y": 0.0, "z": 0.0})
        _image, extent = _controller_backed(controller)._render_top_down_from_controller(
            orthographic=False
        )
        assert extent is None


class TestCacheFormat:
    def test_a_written_cache_stores_the_extent_as_plain_floats(self, tmp_path):
        """Not a dataclass: pickling one pins its module path forever."""
        thor = object.__new__(ThorInterface)
        thor.seed = 11
        # Never used: both controller-touching calls are stubbed out below.
        thor.controller = None
        extent = (-0.4, 3.9, -0.4, 3.9)
        thor._render_top_down_from_controller = (  # ty: ignore[invalid-assignment]
            lambda orthographic=True: (
                np.zeros((4, 4, 3), dtype=np.uint8), extent if orthographic else None,
            )
        )
        thor._get_reachable_positions_from_controller = lambda: []  # ty: ignore[invalid-assignment]

        cache = thor._save_and_get_cache(str(tmp_path))
        assert cache['cache_version'] == SCENE_CACHE_VERSION
        assert cache['image_ortho_extent_m'] == extent

        with open(tmp_path / "scene_11.pkl", 'rb') as handle:
            reloaded = pickle.load(handle)
        stored = reloaded['image_ortho_extent_m']
        assert isinstance(stored, tuple)
        assert all(type(value) is float for value in stored)
