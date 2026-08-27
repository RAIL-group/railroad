"""How long a video comes out: `--video-time` and what it resolves to."""

import pytest

from railroad.dashboard.plotting import DEFAULT_VIDEO_DURATION, resolve_video_duration

PLAN_TIME = 1500.0


@pytest.mark.parametrize(
    "spec, expected",
    [
        (None, DEFAULT_VIDEO_DURATION),
        (15, 15.0), ("15", 15.0), ("15s", 15.0), ("15S", 15.0),
        ("100x", 15.0), ("100X", 15.0), ("0.5x", 3000.0),
    ],
)
def test_a_video_time_is_seconds_unless_it_ends_in_x(spec, expected):
    """`15` and `15s` both ask for fifteen seconds; `100x` plays 1500 in 15."""
    assert resolve_video_duration(spec, PLAN_TIME) == expected


@pytest.mark.parametrize(
    "spec, plan, complaint",
    [
        ("banana", PLAN_TIME, "video time"),  # not a number at all
        ("", PLAN_TIME, "video time"),
        ("inf", PLAN_TIME, "video time"),  # a number, but not a length
        ("0", PLAN_TIME, "length must be positive"),
        ("-2x", PLAN_TIME, "speed must be positive"),
        ("100x", 0.0, "nothing to play"),  # the spec is fine, the plan is empty
    ],
)
def test_what_is_not_a_video_length_is_rejected(spec, plan, complaint):
    with pytest.raises(ValueError, match=complaint):
        resolve_video_duration(spec, plan)


def test_video_time_reaches_save_video_unresolved(fetch_dashboard):
    """Unresolved, because only `save_video` knows what the plan cost."""
    seen = {}
    fetch_dashboard.save_video = lambda path, **kwargs: seen.update(kwargs)
    fetch_dashboard.show_plots(save_video="out.mp4", video_time="100x")
    assert seen["duration"] == "100x"


def _have_ffmpeg() -> bool:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.animation import FFMpegWriter

    return FFMpegWriter.isAvailable()


@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not installed")
def test_the_video_is_as_long_as_it_was_asked_to_be(
    fetch_dashboard, tmp_path, location_coords,
):
    """That a resolved duration is what actually reaches the writer."""
    import subprocess

    fps, duration = 5, 2
    out = tmp_path / "secs.mp4"
    fetch_dashboard.save_video(
        str(out), location_coords=location_coords, duration=float(duration),
        fps=fps, figsize=(4.0, 3.0), dpi=50,
    )
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    )
    # fps * duration, plus the leading poster frame
    assert int(probe.stdout.strip().rstrip(",")) == fps * duration + 1
