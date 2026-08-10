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

    def test_default_antialias_takes_the_shortcut(self, fig):
        """Documents *why* the antialias list is set.

        Without it a single-point draw diverges. If this ever stops being
        true the workaround is obsolete, not broken -- but we want to know.
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
        assert not np.array_equal(shortcut, correct), (
            "matplotlib no longer renders single-element collections "
            "differently; the set_antialiased workaround can be removed"
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

    def test_animated_images_are_drawn_anyway(self, fig):
        """Why images get set_visible(False) rather than the animated flag.

        Axes.draw exempts AxesImage from its animated filter, so relying on
        the flag alone would composite the grids twice.
        """
        ax = fig.add_subplot(111)
        image = ax.imshow(np.zeros((4, 4, 3)))
        image.set_animated(True)
        fig.canvas.draw()
        animated = _buffer(fig)

        image.set_visible(False)
        fig.canvas.draw()
        hidden = _buffer(fig)
        assert not np.array_equal(animated, hidden), (
            "Axes.draw now honours the animated flag for AxesImage; the "
            "set_visible dance in save_video can be simplified"
        )


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
