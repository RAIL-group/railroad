"""Fitting the render to the X screen AI2-THOR will draw it on."""

import subprocess
import warnings

import pytest

from railroad.environment.procthor import _display, thor_interface

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


@pytest.mark.parametrize("screens, render_px", [
    # Colab's Xvfb, fixed at 1024x768: the roomiest square that fits, squarely,
    # and loudly -- a stretched render would misplace every overlay on it.
    ([(800, 600), (1024, 768)], 768),
    ([(4096, 4096)], 2048),  # room to spare: full size, and nothing to say
    ([], 2048),  # nothing measurable: AI2-THOR's own message is the better one
])
def test_the_render_shrinks_only_to_a_screen_that_cannot_take_it(
    screens, render_px, monkeypatch
):
    monkeypatch.setattr(_display, "_usable_screens", lambda: screens)
    monkeypatch.setattr(thor_interface, "TOP_DOWN_RENDER_PX", 2048)
    thor = object.__new__(thor_interface.ThorInterface)  # __init__ loads a scene
    thor.seed = 7

    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always")
        assert thor._render_px() == render_px

    assert bool(warned) is (render_px < 2048)
