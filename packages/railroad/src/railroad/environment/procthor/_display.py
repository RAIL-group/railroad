"""Fitting the render to the X screen AI2-THOR will draw it on.

AI2-THOR's Linux64 platform refuses a render larger than the X screen, and an
X server with no outputs connected defaults to 1024x768 whatever the GPU can
do -- so asking for a 2048 top-down image fails before Unity starts.

An Xorg screen can simply be grown, since ``xrandr`` reports a maximum in the
tens of thousands; it just needs asking. An Xvfb one cannot: its framebuffer is
sized when the server starts and RandR offers that one size as both minimum and
maximum, which is what Colab's 1024x768 default leaves you with. So grow what
can be grown, then ask what is actually there and render no larger.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from typing import Iterator, Optional

_CURRENT = re.compile(r"current\s+(\d+)\s*x\s*(\d+)")


def _candidate_displays() -> list[str]:
    """Displays to consider, the way AI2-THOR finds them.

    It globs the socket directory rather than reading ``DISPLAY``, so an unset
    variable is no reason to skip: it will still find and use ``:0``.
    """
    if os.environ.get("DISPLAY"):
        return [os.environ["DISPLAY"]]
    return sorted(
        ":" + os.path.basename(socket)[1:]
        for socket in glob.glob("/tmp/.X11-unix/X[0-9]*")
    )


def _xrandr(display: str, *args: str) -> Optional[str]:
    if not shutil.which("xrandr"):
        return None
    try:
        done = subprocess.run(
            ["xrandr", "-d", display, *args],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def _growable(display: str, width: int, height: int) -> Optional[tuple[int, int]]:
    """The screen's size, if it is smaller than asked for and safe to resize."""
    listing = _xrandr(display)
    # Never resize a screen someone is looking at: with a monitor attached,
    # growing the framebuffer turns the desktop into a panning viewport.
    if listing is None or re.search(r"^\S+ connected", listing, re.MULTILINE):
        return None
    found = _CURRENT.search(listing)
    if not found:
        return None
    current = (int(found.group(1)), int(found.group(2)))
    return current if current[0] < width or current[1] < height else None


@contextmanager
def screen_at_least(width: int, height: int) -> Iterator[bool]:
    """Grow a headless X screen to fit *width* x *height*, then put it back.

    Yields whether it was grown. Does nothing -- and reports nothing -- when
    there is no display, no ``xrandr``, a monitor is connected, or the screen
    is already big enough; AI2-THOR then fails with its own clear message
    rather than this masking the cause.
    """
    grown: Optional[tuple[str, tuple[int, int]]] = None
    for display in _candidate_displays():
        current = _growable(display, width, height)
        if current and _xrandr(
            display, "--fb",
            f"{max(width, current[0])}x{max(height, current[1])}",
        ) is not None:
            grown = (display, current)
            break
    try:
        yield grown is not None
    finally:
        if grown is not None:
            _xrandr(grown[0], "--fb", f"{grown[1][0]}x{grown[1][1]}")


def _usable_screens() -> list[tuple[int, int]]:
    """The size of every screen AI2-THOR would accept, as it measures them.

    Mirrors ``ai2thor.platform.Linux64._validate_screen``: a screen counts only
    if it speaks GLX at 24-bit depth, so a screen this reports is one AI2-THOR
    will consider. Empty when there is no X at all, or no Xlib to ask with.
    """
    try:
        import Xlib.display
        import Xlib.error
    except ImportError:
        return []

    # A server that will not answer is a screen we cannot count, never a reason
    # to fail: AI2-THOR is about to make the same connection itself, and its
    # message about it is the better one. ConnectionClosedError is listed
    # because Xlib does not derive it from XError.
    unanswered = (
        Xlib.error.DisplayError,
        Xlib.error.XError,
        Xlib.error.ConnectionClosedError,
        OSError,
    )

    sizes = []
    for display in _candidate_displays():
        try:
            connection = Xlib.display.Display(display)
        except unanswered:
            continue
        try:
            for index in range(connection.screen_count()):
                # AI2-THOR connects per screen rather than indexing one
                # connection, since ``list_extensions`` is per display.
                screen_connection = Xlib.display.Display(f"{display}.{index}")
                try:
                    screen = screen_connection.screen()
                    if (
                        "GLX" in screen_connection.list_extensions()
                        and screen["root_depth"] == 24
                    ):
                        sizes.append(
                            (screen["width_in_pixels"], screen["height_in_pixels"])
                        )
                finally:
                    screen_connection.close()
        except unanswered:
            continue
        finally:
            connection.close()
    return sizes


def render_px_at_most(preferred: int) -> int:
    """The largest square render the X screens will take, capped at *preferred*.

    Call inside :func:`screen_at_least`, so a screen that could be grown has
    been. What is left is a screen that cannot: rendering smaller keeps the
    scene generating -- softer than intended, but generated -- where AI2-THOR
    would otherwise refuse to start.

    Returns *preferred* untouched when nothing can be measured (no X, no Xlib,
    macOS), leaving AI2-THOR to fail with its own clear message rather than
    this quietly shrinking a render for a reason it invented.
    """
    fits = [min(width, height) for width, height in _usable_screens()]
    # max, not min: AI2-THOR takes the first screen large enough, so it is the
    # roomiest one that has to fit, not every one of them.
    return min(preferred, max(fits)) if fits else preferred
