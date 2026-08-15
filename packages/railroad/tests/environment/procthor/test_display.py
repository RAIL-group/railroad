"""Growing a headless X screen so AI2-THOR will render into it."""

import subprocess

import pytest

from railroad.environment.procthor import _display

SCREEN = "Screen 0: minimum 8 x 8, current 1024 x 768, maximum 32767 x 32767\n"


def _patch(monkeypatch, outputs: str) -> list:
    calls: list = []

    def run(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, SCREEN + outputs, "")

    monkeypatch.setattr(_display.subprocess, "run", run)
    monkeypatch.setattr(_display.shutil, "which", lambda _name: "/usr/bin/xrandr")
    return calls


@pytest.mark.parametrize("outputs, grows", [("DP-0 disconnected", True),
                                            ("DP-0 connected primary", False)])
def test_only_an_unwatched_screen_is_grown(outputs, grows, monkeypatch):
    """Growing the framebuffer past an attached monitor makes the desktop pan."""
    calls = _patch(monkeypatch, outputs)
    monkeypatch.setenv("DISPLAY", ":0")

    with _display.screen_at_least(2048, 2048) as grew:
        assert grew is grows

    assert [c for c in calls if "--fb" in c] == ([
        ["xrandr", "-d", ":0", "--fb", "2048x2048"],
        ["xrandr", "-d", ":0", "--fb", "1024x768"],
    ] if grows else [])


def test_an_unset_display_is_still_found(monkeypatch):
    """AI2-THOR globs the socket directory rather than reading DISPLAY, so an
    unset variable is no reason to leave the screen too small for it."""
    calls = _patch(monkeypatch, "DP-0 disconnected")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(_display.glob, "glob", lambda _p: ["/tmp/.X11-unix/X0"])

    with _display.screen_at_least(2048, 2048) as grew:
        assert grew

    assert ["xrandr", "-d", ":0", "--fb", "2048x2048"] in calls
