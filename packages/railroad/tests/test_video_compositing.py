"""Guards for the matplotlib behaviours ``save_video`` composites against.

``PlannerDashboard.save_video`` does not redraw the whole figure per frame.
It caches rasters and replays individual artists over them, which relies on
three matplotlib behaviours that are real but undocumented. If any of them
changes, the video silently renders differently rather than failing, so they
are pinned here instead of being discovered from a corrupted animation.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402


@pytest.fixture
def fig():
    figure = plt.figure(figsize=(2, 2), dpi=50)
    FigureCanvasAgg(figure)
    yield figure
    plt.close(figure)


def _buffer(figure) -> np.ndarray:
    return np.asarray(figure.canvas.buffer_rgba()).copy()


class TestIncrementalScatter:
    """The trail accumulates into the cached raster a few points at a time."""

    @staticmethod
    def _scatter(ax, n):
        pts = np.column_stack([np.linspace(0.1, 0.9, n), np.linspace(0.1, 0.9, n)])
        sizes = np.linspace(25.0, 2.0, n)
        colors = plt.get_cmap("Reds")(np.linspace(0.25, 1.0, n))
        scatter = ax.scatter([], [], s=[], zorder=5, alpha=1.0)
        # Mirrors save_video: keeps Collection.draw off its single-element
        # "stamp one marker" shortcut, which antialiases differently.
        scatter.set_antialiased([True, True])
        return scatter, pts, sizes, colors

    def _render(self, fig, chunks):
        """Draw a fixed 12-point trail, split into *chunks* consecutive draws."""
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        scatter, pts, sizes, colors = self._scatter(ax, 12)
        scatter.set_animated(True)
        fig.canvas.draw()
        for lo, hi in chunks:
            scatter.set_offsets(pts[lo:hi])
            scatter.set_sizes(sizes[lo:hi])
            scatter.set_facecolors(colors[lo:hi])
            fig.draw_artist(scatter)
        return _buffer(fig)

    def test_one_point_at_a_time_matches_all_at_once(self, fig):
        """Point-by-point accumulation must equal a single full draw.

        This is the invariant the incremental trail rests on, and the reason
        for the two-element antialias list: a one-element collection would
        otherwise take a different rendering path.
        """
        whole = self._render(fig, [(0, 12)])
        fig.clear()
        one_by_one = self._render(fig, [(i, i + 1) for i in range(12)])
        assert np.array_equal(whole, one_by_one)

    def test_uneven_chunks_match_all_at_once(self, fig):
        whole = self._render(fig, [(0, 12)])
        fig.clear()
        chunked = self._render(fig, [(0, 5), (5, 6), (6, 12)])
        assert np.array_equal(whole, chunked)

    def test_reports_whether_the_antialias_workaround_still_matters(self, fig):
        """Records whether the shortcut being guarded against is still live.

        Skips rather than fails when it is not: whether matplotlib renders a
        one-element collection differently is a property of the installed
        version, not of this repo, so it must not break CI on a version bump.
        The invariant that actually matters is asserted above.
        """
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        scatter, pts, sizes, colors = self._scatter(ax, 12)
        scatter.set_antialiased(True)  # matplotlib's default: scalar
        scatter.set_animated(True)
        fig.canvas.draw()
        for i in range(12):
            scatter.set_offsets(pts[i:i + 1])
            scatter.set_sizes(sizes[i:i + 1])
            scatter.set_facecolors(colors[i:i + 1])
            fig.draw_artist(scatter)
        shortcut = _buffer(fig)

        fig.clear()
        correct = self._render(fig, [(0, 12)])
        if np.array_equal(shortcut, correct):
            pytest.skip(
                f"matplotlib {matplotlib.__version__} renders one-element "
                "collections like larger ones; the set_antialiased call in "
                "save_video is now a no-op and could be dropped"
            )


class TestAnimatedArtistFiltering:
    """save_video hides images but only flags everything else animated."""

    def test_animated_artists_are_skipped(self, fig):
        ax = fig.add_subplot(111)
        (line,) = ax.plot([0, 1], [0, 1], lw=10, color="black")
        line.set_animated(True)
        fig.canvas.draw()
        without = _buffer(fig)

        line.set_animated(False)
        fig.canvas.draw()
        with_line = _buffer(fig)
        assert not np.array_equal(without, with_line)

    def test_hiding_an_image_keeps_it_out_of_the_draw(self, fig):
        """The property save_video relies on for the grids and camera views.

        It hides images instead of flagging them animated because that is
        the one approach that holds either way: matplotlib 3.10's Axes.draw
        exempts AxesImage from its animated filter (so the flag alone would
        composite the grids twice), while 3.11 honours it. Hiding excludes
        them on both, so only that is asserted here.
        """
        ax = fig.add_subplot(111)
        image = ax.imshow(np.zeros((4, 4, 3)))
        fig.canvas.draw()
        visible = _buffer(fig)

        image.set_visible(False)
        fig.canvas.draw()
        hidden = _buffer(fig)
        assert not np.array_equal(visible, hidden)

        # ...and drawing it explicitly afterwards puts it back, which is how
        # the chrome layer composites it.
        image.set_visible(True)
        image.set_animated(True)
        fig.canvas.restore_region(fig.canvas.copy_from_bbox(fig.bbox))
        fig.draw_artist(image)
        assert np.array_equal(visible, _buffer(fig))


class TestRegionCaching:
    """copy_from_bbox/restore_region must round-trip the canvas exactly."""

    def test_restore_round_trips(self, fig):
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(fig.bbox)
        before = _buffer(fig)

        marker, = ax.plot([0.5], [0.5], "o", ms=20)
        marker.set_animated(True)
        fig.draw_artist(marker)
        assert not np.array_equal(before, _buffer(fig))

        fig.canvas.restore_region(background)
        assert np.array_equal(before, _buffer(fig))


class TestBufferMatchesWriterFrameSize:
    def test_buffer_nbytes_is_width_times_height_times_four(self, fig):
        """grab_frame's size check assumes this layout."""
        fig.set_dpi(100)
        FigureCanvasAgg(fig)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        width, height = fig.canvas.get_width_height()
        assert buf.nbytes == width * height * 4


def _have_ffmpeg() -> bool:
    from matplotlib.animation import FFMpegWriter
    return FFMpegWriter.isAvailable()


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not installed")
class TestSaveVideoEndToEnd:
    @pytest.fixture
    def dashboard(self):
        from railroad import operators
        from railroad._bindings import State
        from railroad.core import Fluent as F
        from railroad.dashboard import PlannerDashboard
        from railroad.environment import ObjectSearchEnvironment

        move_op = operators.construct_move_operator_blocking(lambda r, a, b: 10.0)
        env = ObjectSearchEnvironment(
            state=State(0.0, {F("at r1 A"), F("free r1")}, []),
            objects_by_type={"robot": {"r1"}, "location": {"A", "B"}},
            operators=[move_op],
        )
        db = PlannerDashboard(
            F("at r1 B"), env, force_interactive=False, print_on_exit=False,
        )
        db.known_robots = {"r1"}
        db._entity_positions = {"r1": [(0.0, "A", None), (10.0, "B", None)]}
        db._goal_time = 10.0
        db.actions_taken = [("move r1 A B", 0.0)]
        return db

    def test_writes_the_requested_frames_at_the_requested_size(
        self, dashboard, tmp_path,
    ):
        out = tmp_path / "trajectory.mp4"
        dashboard.save_video(
            str(out),
            location_coords={"A": (0.0, 0.0), "B": (10.0, 0.0)},
            fps=5, duration=1.0, figsize=(4.0, 3.0), dpi=50,
        )
        assert out.is_file() and out.stat().st_size > 0

        import subprocess
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries",
             "stream=width,height,nb_read_frames", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True, check=True,
        )
        width, height, frames = probe.stdout.strip().split(",")
        assert (int(width), int(height)) == (200, 150)
        # fps * duration, plus the leading poster frame
        assert int(frames) == 6


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not installed")
class TestSceneImageSurvivesCompositing:
    """The overhead image is static chrome, not a per-frame artist.

    ``save_video`` caches a "chrome" raster and replays only moving artists
    over it. The image is deliberately left out of the layered artist sets so
    the plain ``canvas.draw()`` bakes it into that cache. Wire it in as an
    animated artist instead and it drops out of the restored region, leaving
    the frames white behind the map.
    """

    PHOTO_COLOR = (200, 40, 160)

    @pytest.fixture
    def dashboard(self):
        from railroad import operators
        from railroad._bindings import State
        from railroad.core import Fluent as F
        from railroad.dashboard import PlannerDashboard
        from railroad.environment import ObjectSearchEnvironment
        from railroad.environment.types import TopDownView

        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:, :] = self.PHOTO_COLOR

        class Scene:
            def get_top_down_view(self):
                # Overhangs the grid on every side, as ProcTHOR's does.
                return TopDownView(image=image, min_x=-20.0, max_x=39.0,
                                   min_y=-20.0, max_y=39.0)

        env = ObjectSearchEnvironment(
            state=State(0.0, {F("at r1 A"), F("free r1")}, []),
            objects_by_type={"robot": {"r1"}, "location": {"A", "B"}},
            operators=[operators.construct_move_operator_blocking(lambda r, a, b: 10.0)],
        )
        env.occupancy_grid = np.zeros((20, 20))  # ty: ignore[unresolved-attribute]
        env.scene = Scene()  # ty: ignore[unresolved-attribute]

        db = PlannerDashboard(
            F("at r1 B"), env, force_interactive=False, print_on_exit=False,
        )
        db.known_robots = {"r1"}
        db._entity_positions = {"r1": [(0.0, "A", None), (10.0, "B", None)]}
        db._goal_time = 10.0
        db.actions_taken = [("move r1 A B", 0.0)]
        return db

    def test_the_image_is_present_and_unchanged_across_frames(
        self, dashboard, tmp_path,
    ):
        import subprocess

        import matplotlib.image as mpimg

        out = tmp_path / "trajectory.mp4"
        dashboard.save_video(
            str(out), location_coords={"A": (2.0, 2.0), "B": (17.0, 17.0)},
            fps=5, duration=1.0, figsize=(8.0, 6.0), dpi=100,
        )
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(out),
             str(frames_dir / "frame_%03d.png")],
            check=True, capture_output=True,
        )
        frames = sorted(frames_dir.glob("frame_*.png"))
        assert len(frames) >= 3

        def photo_pixels(path):
            # The image overhangs the grid, so its colour is the only thing
            # that can be in the corner of the axes.
            rgb = (mpimg.imread(str(path))[:, :, :3] * 255).astype(int)
            return (np.abs(rgb - np.array(self.PHOTO_COLOR)).sum(axis=2) < 60).sum()

        counts = [photo_pixels(p) for p in (frames[1], frames[-1])]
        assert counts[0] > 500, "scene image missing from the composited frames"
        # Static chrome: the same pixels every frame, modulo h264 noise.
        assert abs(counts[0] - counts[1]) < 0.05 * counts[0]
