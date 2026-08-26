"""How long a video comes out: `--video-time` and what it resolves to."""

import pytest

from railroad.dashboard.plotting import (
    DEFAULT_VIDEO_DURATION,
    parse_video_time,
    resolve_video_duration,
)

PLAN_TIME = 1500.0


class TestResolveVideoDuration:
    def test_nothing_asked_for_gives_the_default(self):
        assert resolve_video_duration(None, PLAN_TIME) == DEFAULT_VIDEO_DURATION

    @pytest.mark.parametrize(
        "spec", [15, 15.0, "15", "15.0", " 15 ", "15s", "15S", "15.0s", " 15 s"],
    )
    def test_a_number_is_seconds_of_video(self, spec):
        """With or without the unit: `15` and `15s` say the same thing."""
        assert resolve_video_duration(spec, PLAN_TIME) == 15.0

    def test_a_length_ignores_what_the_plan_cost(self):
        assert resolve_video_duration(15, 3.0) == resolve_video_duration(15, 9e9)

    @pytest.mark.parametrize("spec", ["100x", "100X", " 100x "])
    def test_a_speed_divides_the_plan_by_it(self, spec):
        """The worked example: 1500 of plan at 100x is 15 seconds of video."""
        assert resolve_video_duration(spec, PLAN_TIME) == 15.0

    def test_a_fractional_speed_runs_longer_than_the_plan(self):
        assert resolve_video_duration("0.5x", 30.0) == 60.0

    def test_doubling_the_speed_halves_the_video(self):
        assert (
            resolve_video_duration("200x", PLAN_TIME) * 2
            == resolve_video_duration("100x", PLAN_TIME)
        )

    @pytest.mark.parametrize(
        "spec",
        ["banana", "", "x", "s", "1e", "10x5", "--", "5sx", "5xs", "inf", "nan"],
    )
    def test_unreadable_specs_are_rejected(self, spec):
        with pytest.raises(ValueError, match="video time"):
            resolve_video_duration(spec, PLAN_TIME)

    def test_the_unit_does_not_turn_a_speed_into_a_length(self):
        """`s` and `x` are the whole of the grammar; only the last one counts."""
        assert resolve_video_duration("100x", PLAN_TIME) == 15.0
        assert resolve_video_duration("100s", PLAN_TIME) == 100.0

    def test_a_space_before_the_x_is_tolerated(self):
        """`float` ignores it, and nothing else could have been meant."""
        assert resolve_video_duration("100 x", PLAN_TIME) == 15.0

    @pytest.mark.parametrize("spec", ["0", "-5", "0s", "-5s", 0, -1.5])
    def test_a_length_of_no_seconds_is_rejected(self, spec):
        with pytest.raises(ValueError, match="length must be positive"):
            resolve_video_duration(spec, PLAN_TIME)

    @pytest.mark.parametrize("spec", ["0x", "-2x"])
    def test_a_speed_of_zero_or_less_is_rejected(self, spec):
        with pytest.raises(ValueError, match="speed must be positive"):
            resolve_video_duration(spec, PLAN_TIME)

    def test_a_speed_over_a_plan_that_cost_nothing_is_rejected(self):
        """Only reachable here: the spec itself is fine, the plan is not."""
        parse_video_time("100x")  # so the spec is not what is being rejected
        with pytest.raises(ValueError, match="nothing to play"):
            resolve_video_duration("100x", 0.0)


class TestShowPlotsForwardsIt:
    def test_video_time_reaches_save_video_unresolved(self, fetch_dashboard):
        """Unresolved, because only save_video knows what the plan cost."""
        seen = {}

        def record(path, **kwargs):
            seen.update(path=path, **kwargs)

        fetch_dashboard.save_video = record
        fetch_dashboard.show_plots(save_video="out.mp4", video_time="100x")
        assert seen["duration"] == "100x"

    def test_asking_for_no_video_time_leaves_the_default(self, fetch_dashboard):
        seen = {}
        fetch_dashboard.save_video = lambda path, **kwargs: seen.update(kwargs)
        fetch_dashboard.show_plots(save_video="out.mp4")
        assert seen["duration"] is None


def _have_ffmpeg() -> bool:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.animation import FFMpegWriter

    return FFMpegWriter.isAvailable()


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not installed")
class TestVideoIsAsLongAsAsked:
    FPS = 5

    def _frames(self, dashboard, path, duration, location_coords) -> int:
        import subprocess

        dashboard.save_video(
            str(path), location_coords=location_coords, duration=duration,
            fps=self.FPS, figsize=(4.0, 3.0), dpi=50,
        )
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        )
        return int(probe.stdout.strip().rstrip(","))

    def test_seconds_give_that_many_seconds_of_frames(
        self, fetch_dashboard, tmp_path, location_coords,
    ):
        frames = self._frames(
            fetch_dashboard, tmp_path / "secs.mp4", 2.0, location_coords,
        )
        # fps * duration, plus the leading poster frame
        assert frames == self.FPS * 2 + 1

    def test_twice_the_speed_writes_half_the_video(
        self, fetch_dashboard, tmp_path, location_coords,
    ):
        """Asserted as a ratio: the plan's own cost is not a public number."""
        slow = self._frames(
            fetch_dashboard, tmp_path / "slow.mp4", "4x", location_coords,
        )
        fast = self._frames(
            fetch_dashboard, tmp_path / "fast.mp4", "8x", location_coords,
        )
        assert slow > 3, "too few frames for the ratio to mean anything"
        assert (slow - 1) == pytest.approx(2 * (fast - 1), abs=1)
