"""Making a headless X screen big enough for AI2-THOR to render into.

AI2-THOR's Linux64 platform refuses a render larger than the X screen, and an
X server with no outputs connected defaults to 1024x768 whatever the GPU can
do -- so asking for a 2048 top-down image fails before Unity starts. The screen
can simply be grown, since ``xrandr`` reports a maximum in the tens of
thousands; it just needs asking.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from typing import Iterator, Optional

_CURRENT = re.compile(r"current\s+(\d+)\s*x\s*(\d+)")


def _screen_size() -> Optional[tuple[int, int]]:
    """Current X screen size, or None if it cannot be asked for."""
    if not os.environ.get("DISPLAY") or not shutil.which("xrandr"):
        return None
    try:
        out = subprocess.run(
            ["xrandr"], capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    # Never resize a screen someone is looking at: with a monitor attached,
    # growing the framebuffer turns the desktop into a panning viewport.
    if re.search(r"^\S+ connected", out.stdout, re.MULTILINE):
        return None
    found = _CURRENT.search(out.stdout)
    return (int(found.group(1)), int(found.group(2))) if found else None


def _resize(width: int, height: int) -> bool:
    try:
        return subprocess.run(
            ["xrandr", "--fb", f"{width}x{height}"],
            capture_output=True, timeout=10,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@contextmanager
def screen_at_least(width: int, height: int) -> Iterator[bool]:
    """Grow a headless X screen to fit *width* x *height*, then put it back.

    Yields whether it was grown. Does nothing -- and reports nothing -- when
    there is no display, no ``xrandr``, a monitor is connected, or the screen
    is already big enough; AI2-THOR then fails with its own clear message
    rather than this masking the cause.
    """
    original = _screen_size()
    grew = bool(
        original
        and (original[0] < width or original[1] < height)
        and _resize(max(width, original[0]), max(height, original[1]))
    )
    try:
        yield grew
    finally:
        if grew and original is not None:
            _resize(*original)
