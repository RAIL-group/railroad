"""Growing a headless X screen so AI2-THOR will render into it."""

import subprocess

from railroad.environment.procthor import _display

HEADLESS = "Screen 0: minimum 8 x 8, current 1024 x 768, maximum 32767 x 32767\nDP-0 disconnected\n"
WITH_MONITOR = "Screen 0: minimum 8 x 8, current 1024 x 768, maximum 32767 x 32767\nDP-0 connected primary 1024x768+0+0\n"


def _patch(monkeypatch, listing: str, calls: list):
    def run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, listing, "")

    monkeypatch.setattr(_display.subprocess, "run", run)
    monkeypatch.setattr(_display.shutil, "which", lambda _name: "/usr/bin/xrandr")
    monkeypatch.setenv("DISPLAY", ":0")


def test_a_headless_screen_grows_and_is_put_back(monkeypatch):
    calls: list = []
    _patch(monkeypatch, HEADLESS, calls)

    with _display.screen_at_least(2048, 2048) as grew:
        assert grew

    assert calls[1] == ["xrandr", "--fb", "2048x2048"]
    assert calls[2] == ["xrandr", "--fb", "1024x768"]


def test_a_screen_someone_is_looking_at_is_left_alone(monkeypatch):
    """Growing the framebuffer past the monitor makes the desktop pan."""
    calls: list = []
    _patch(monkeypatch, WITH_MONITOR, calls)

    with _display.screen_at_least(2048, 2048) as grew:
        assert not grew

    assert all("--fb" not in call for call in calls)


def test_no_display_is_not_an_error(monkeypatch):
    """macOS has no X screen to size, and no resolution check either."""
    monkeypatch.delenv("DISPLAY", raising=False)
    with _display.screen_at_least(2048, 2048) as grew:
        assert not grew
