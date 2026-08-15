"""Making a headless X screen big enough for AI2-THOR to render into.

AI2-THOR's Linux64 platform refuses a render larger than the X screen, and an
X server with no outputs connected defaults to 1024x768 whatever the GPU can
do -- so asking for a 2048 top-down image fails before Unity starts. The screen
can simply be grown, since ``xrandr`` reports a maximum in the tens of
thousands; it just needs asking.
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
