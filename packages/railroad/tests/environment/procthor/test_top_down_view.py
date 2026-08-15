"""Placing the ProcTHOR overhead image on the occupancy grid.

The camera is told what to frame and the footprint is recorded, so a cached
image can be positioned with no controller running. These pin the conversion
into grid cells, and the guard that stops an image being drawn where it does
not belong.
"""

import pickle
import warnings

import numpy as np
import pytest

from railroad.environment.procthor import resources
from railroad.environment.procthor.thor_interface import (
    SCENE_CACHE_VERSION,
    _decode_image,
    TOP_DOWN_MARGIN_M,
    ThorInterface,
    _EXTENT_WARNED,
)

RESOLUTION = 0.05
GRID_SHAPE = (69, 47)


def _room(corners):
    """A scene-JSON room with the given (x, z) floor polygon."""
    return {"floorPolygon": [{"x": x, "y": 0.0, "z": z} for x, z in corners]}


@pytest.fixture(autouse=True)
def _forget_warned_seeds():
    """The once-per-seed warning guard is module state; keep tests independent."""
    _EXTENT_WARNED.clear()
    yield
    _EXTENT_WARNED.clear()


def _interface(
    *, extent_m=(0.0, 1.0, 0.0, 1.0), offset=(0.0, 0.0), resolution=RESOLUTION,
    include_extent=True, seed=7,
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
        'image_ortho': np.zeros((4, 4, 3), dtype=np.uint8),
        'image_persp': None,
    }
    if include_extent:
        cached['image_ortho_extent_m'] = extent_m
    thor.cached_data = cached
    return thor


@pytest.mark.parametrize("offset", [(0.0, 0.0), (-1.35, -0.4)])
def test_an_image_covering_the_grid_gets_the_grid_s_own_extent(offset):
    """The anchor: half-cell convention, axis order and y flip, at once.

    An image whose world footprint is exactly the grid's footprint must land on
    the extent matplotlib infers for the grid itself, so any half-cell slip,
    swapped axis or flipped y shows up here. Cell k spans k-0.5 to k+0.5. The
    negative offset covers scenes with ``min_x < 0``, which take their own
    branch when the grid is built.
    """
    n_x, n_y = GRID_SHAPE
    # Straight at the conversion, not through get_top_down_view: a real
    # footprint is square, and a square one cannot cover a 69x47 grid exactly.
    thor = _interface(offset=offset)
    covering_the_grid = (
        offset[0] - 0.5 * RESOLUTION, offset[0] + (n_x - 0.5) * RESOLUTION,
        offset[1] - 0.5 * RESOLUTION, offset[1] + (n_y - 0.5) * RESOLUTION,
    )
    view = thor._view_from_extent(np.zeros((4, 4, 3), dtype=np.uint8), covering_the_grid)
    assert view.extent == pytest.approx((-0.5, n_x - 0.5, n_y - 0.5, -0.5))
    # bottom > top, which is what keeps the y axis increasing downward.
    assert view.extent[2] > view.extent[3]

    # The extent is stored in meters, so halving the cell size doubles the
    # cell coordinates it maps to. Storing cells instead would silently
    # corrupt any run that does not use the default resolution, since the
    # cache filename does not record one.
    finer = _interface(offset=offset, resolution=RESOLUTION / 2)
    assert finer._view_from_extent(
        np.zeros((4, 4, 3), dtype=np.uint8), covering_the_grid,
    ).max_x == pytest.approx(2 * view.max_x)

    # scale_to_grid keys g2p_map, so it must keep returning plain ints --
    # numpy scalars there break every lookup -- and grid_to_world must invert
    # it, being the inverse this module previously had no name for.
    assert thor.scale_to_grid(thor.grid_to_world((12, 45))) == (12, 45)
    assert all(type(c) is int for c in thor.scale_to_grid((0.7, -0.2)))


def test_the_cache_round_trips_or_is_refused(tmp_path, monkeypatch):
    """The extent must survive a pickle round trip, and anything untrustworthy
    must be turned away rather than drawn in the wrong place: written before
    the extent existed, or framed differently, which is what a changed
    TOP_DOWN_MARGIN_M looks like on disk. Warned once per seed, since
    ``save_video`` reaches this every frame.
    """
    # Plain floats, not a dataclass: pickling one pins its module path, and
    # renaming it later would turn every cache on disk into an AttributeError.
    writer = object.__new__(ThorInterface)
    writer.seed = 11
    writer.controller = None
    extent = (-0.4, 3.9, -0.4, 3.9)
    writer._render_top_down_from_controller = (  # ty: ignore[invalid-assignment]
        lambda orthographic=True: (
            np.zeros((4, 4, 3), dtype=np.uint8), extent if orthographic else None,
        )
    )
    writer._get_reachable_positions_from_controller = lambda: []  # ty: ignore[invalid-assignment]
    writer._save_and_get_cache(str(tmp_path))
    with open(tmp_path / "scene_11.pkl", 'rb') as handle:
        written = pickle.load(handle)
    stored = written['image_ortho_extent_m']
    assert isinstance(stored, tuple)
    assert all(type(value) is float for value in stored)
    # JPEG, since a 2048 render is ~50x this size raw.
    assert isinstance(written['image_ortho'], bytes)
    assert _decode_image(written['image_ortho']).shape == (4, 4, 3)

    monkeypatch.setattr(resources, "DEFAULT_RESOURCES_BASE", tmp_path / "elsewhere")
    for description, thor in [
        ("no extent recorded", _interface(include_extent=False, seed=4001)),
        ("framed differently", _interface(seed=4001)),
    ]:
        if description == "framed differently":
            cache = thor.cached_data
            assert cache is not None
            cache['image_ortho_extent_m'] = tuple(
                v + 1.0 for v in cache['image_ortho_extent_m']
            )
        _EXTENT_WARNED.clear()
        with pytest.warns(RuntimeWarning) as caught:
            assert thor.get_top_down_view() is None, description
        message = str(caught[0].message)
        assert "4001" in message and str(tmp_path / "elsewhere") in message

        with warnings.catch_warnings(record=True) as second:
            warnings.simplefilter("always")
            thor.get_top_down_view()
        assert second == [], f"warned twice for {description}"


class _FakeEvent:
    def __init__(self, metadata, frames=()):
        self.metadata = metadata
        self.third_party_camera_frames = list(frames)


class _FakeController:
    """Stands in for Unity, recording the pose the camera was actually given."""

    def __init__(self, *, rotation, frame_shape=(480, 480)):
        self._rotation = rotation
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
                "sceneBounds": {"size": {"x": 4.0, "y": 3.0, "z": 3.0}},
            })
        assert action == "AddThirdPartyCamera"
        kwargs.pop("skyboxColor", None)
        kwargs.pop("raise_for_failure", None)
        self.camera_pose = kwargs
        height, width = self._frame_shape
        return _FakeEvent({}, frames=[np.zeros((height, width, 3), dtype=np.uint8)])


def _controller_backed(controller) -> ThorInterface:
    thor = object.__new__(ThorInterface)
    thor.seed = 5
    thor.controller = controller
    thor.cached_data = None
    thor.grid_resolution = RESOLUTION
    thor.grid_offset = np.array((0.0, 0.0))
    thor.rooms = [_room([(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)])]
    return thor


def test_the_camera_frames_our_footprint_not_thor_s():
    """THOR frames on ``sceneBounds``, which is not reconstructible offline.

    So the camera is told where to look and how much to cover, and the extent
    reported back is exactly that. sceneBounds still sets the camera *height*,
    which cannot skew an orthographic footprint.
    """
    controller = _FakeController(rotation={"x": 90.0, "y": 0.0, "z": 0.0})
    thor = _controller_backed(controller)
    _image, extent = thor._render_top_down_from_controller()

    assert extent is not None
    assert extent == pytest.approx(thor.top_down_footprint())

    # That footprint squares up the room polygons plus a margin -- framed on
    # the rooms, not on reachable space, since seeds 7005 and 8619 contain
    # rooms the agent cannot enter and framing on where it can walk would crop
    # them out of the picture entirely.
    thor.rooms = [
        _room([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]),
        _room([(0.0, 4.0), (4.0, 4.0), (4.0, 9.0), (0.0, 9.0)]),  # unreachable
    ]
    half = 0.5 * 9.0 + TOP_DOWN_MARGIN_M  # the longer span wins
    assert thor.top_down_footprint() == pytest.approx(
        (2.0 - half, 2.0 + half, 4.5 - half, 4.5 + half)
    )
    min_x, max_x, min_z, max_z = extent
    pose = controller.camera_pose
    assert pose["position"]["x"] == pytest.approx(0.5 * (min_x + max_x))
    assert pose["position"]["z"] == pytest.approx(0.5 * (min_z + max_z))
    assert pose["orthographicSize"] == pytest.approx(0.5 * (max_z - min_z))
    assert pose["position"]["y"] == pytest.approx(2.0 + 1.1 * 4.0)
    # A projective view has no rectangular footprint to report.
    assert thor._render_top_down_from_controller(orthographic=False)[1] is None


def test_a_camera_that_cannot_be_mapped_is_refused_rather_than_skewed():
    """Both of these would misplace every overlay while looking plausible.

    A yaw rotates the footprint out of axis-alignment, which no rectangular
    extent can express; a non-square frame stretches one axis against a square
    footprint.
    """
    yawed = _FakeController(rotation={"x": 90.0, "y": 35.0, "z": 0.0})
    with pytest.warns(RuntimeWarning, match="rotation"):
        _controller_backed(yawed)._render_top_down_from_controller()
    assert yawed.camera_pose["rotation"] == {"x": 90.0, "y": 0.0, "z": 0.0}

    oblong = _FakeController(rotation={"x": 90.0, "y": 0.0, "z": 0.0},
                             frame_shape=(480, 640))
    with pytest.raises(RuntimeError, match="square"):
        _controller_backed(oblong)._render_top_down_from_controller()
