"""A scrolling viewer for the step diffs, in place of an external pager.

``less`` was the obvious thing to reach for and it is the wrong shape for this.
Two reasons, both of which show up over ssh in the middle of a talk:

- the mouse wheel does nothing unless less is new enough for ``--mouse`` and the
  terminal agrees, which is not a thing to discover live; and
- a side-by-side diff is *laid out* for a width. less redraws when the window
  changes but it cannot re-flow two columns of code, so resizing leaves the
  layout wrong until you quit and run the command again.

Both are fixed by owning the loop: this renders through rich, keeps the lines,
and re-renders from scratch whenever the terminal changes size -- which is why
:func:`show` takes a *callable* of width rather than a finished renderable.

It is deliberately small. Scrolling, resizing and quitting, no search and no
horizontal scroll: rich folds long lines, so there is nothing off to the side to
scroll to. ``RAILROAD_TUTORIAL_VIEWER=off`` prints everything inline instead.
"""

from __future__ import annotations

import io
import os
import re
import select
import shutil
import sys
from typing import Any, Callable, List, Literal, Optional

from rich.console import Console

VIEWER_ENV = "RAILROAD_TUTORIAL_VIEWER"
"""Set to ``off`` to dump the whole thing and never take over the screen."""

WHEEL_LINES = 3
"""Lines per wheel notch. Three is what every other pager settled on."""

_ALT_SCREEN = ("\x1b[?1049h", "\x1b[?1049l")
_CURSOR = ("\x1b[?25l", "\x1b[?25h")
# 1000 is button tracking, 1006 asks for the SGR encoding of it. Terminals that
# do not know 1006 fall back to the legacy form, which is parsed below too.
_MOUSE = ("\x1b[?1000h\x1b[?1006h", "\x1b[?1006l\x1b[?1000l")
_HOME, _CLEAR_LINE = "\x1b[H", "\x1b[K"

_SGR_MOUSE = re.compile(rb"\x1b\[<(\d+);\d+;\d+[Mm]")

_KEYS = {
    b"q": "quit", b"\x1b": "quit", b"\x03": "quit",
    b"j": "down", b"\x1b[B": "down", b"\r": "down", b"\n": "down",
    b"k": "up", b"\x1b[A": "up",
    b" ": "page-down", b"\x1b[6~": "page-down", b"\x06": "page-down",
    b"b": "page-up", b"\x1b[5~": "page-up", b"\x02": "page-up",
    b"g": "top", b"\x1b[H": "top", b"\x1b[1~": "top",
    b"G": "bottom", b"\x1b[F": "bottom", b"\x1b[4~": "bottom",
}


def parse_key(data: bytes) -> str:
    """Map one burst of terminal input to an action, or ``""`` to ignore it."""
    if not data:
        return ""
    match = _SGR_MOUSE.match(data)
    if match:
        button = int(match.group(1))
        return {64: "wheel-up", 65: "wheel-down"}.get(button, "")
    if data.startswith(b"\x1b[M") and len(data) >= 6:
        return {64: "wheel-up", 65: "wheel-down"}.get(data[3] - 32, "")
    return _KEYS.get(data, "")


def scroll_to(top: int, action: str, total: int, page: int) -> int:
    """Where *action* moves the viewport, clamped to the document.

    Split out from the loop because it is the only part with arithmetic worth
    being wrong, and the only part worth testing without a terminal.
    """
    limit = max(total - page, 0)
    delta = {
        "down": 1, "up": -1,
        "wheel-down": WHEEL_LINES, "wheel-up": -WHEEL_LINES,
        "page-down": page, "page-up": -page,
    }
    if action == "top":
        return 0
    if action == "bottom":
        return limit
    return max(0, min(top + delta.get(action, 0), limit))


ColorSystem = Literal["standard", "256", "truecolor", "windows"]


def color_system(console: Console) -> ColorSystem:
    """What *console* detected, narrowed to what :class:`Console` accepts back.

    ``Console.color_system`` reports a plain string and the constructor wants
    one of a fixed set, so the round trip needs saying out loud. Anything
    unrecognised, including no colour at all, is treated as eight colours.
    """
    detected = console.color_system
    if detected in ("standard", "256", "truecolor", "windows"):
        return detected
    return "standard"


def render_lines(console: Console, build: Callable[[int], Any], width: int) -> List[str]:
    """Lay ``build(width)`` out at *width* and return it as styled lines.

    Rendered at the same colour depth as *console*, which is the whole of the
    diff's legibility: pinning this to "standard" rounds every colour to the
    nearest of eight, and the dark backgrounds that mark a changed line round
    to black -- leaving a diff with no marking on it at all.
    """
    buffer = io.StringIO()
    # A second console because the real one is fixed to the terminal's size and
    # we may be rendering for a size it has not caught up with yet.
    Console(file=buffer, width=width, force_terminal=True,
            color_system=color_system(console)).print(build(width))
    return buffer.getvalue().splitlines()


def terminal_size(fd: int) -> os.terminal_size:
    """The terminal's real size, asked of the terminal.

    Not :func:`shutil.get_terminal_size`, which consults ``$COLUMNS`` first: a
    shell that exports it would pin this viewer to a stale width and resizing
    would do nothing, which is the exact complaint the viewer exists to fix.
    """
    try:
        return os.get_terminal_size(fd)
    except OSError:  # pragma: no cover - not a terminal
        return shutil.get_terminal_size()


def _status(top: int, total: int, page: int, width: int) -> str:
    at_end = top >= max(total - page, 0)
    percent = "all" if total <= page else ("end" if at_end
                                           else f"{100 * top // max(total - page, 1)}%")
    keys = "wheel/jk scroll · space page · g/G ends · q quit"
    bar = f" {percent:>3}   {keys}"
    return f"\x1b[7m{bar[:width]:<{width}}\x1b[0m"


def _draw(out, lines: List[str], top: int, width: int, height: int) -> None:
    chunks = [_HOME]
    for row in range(height):
        index = top + row
        chunks.append((lines[index] if index < len(lines) else "") + _CLEAR_LINE + "\n")
    chunks.append(_status(top, len(lines), height, width))
    out.write("".join(chunks))
    out.flush()


def show(console: Console, build: Callable[[int], Any]) -> bool:
    """Scroll ``build(width)`` on the alternate screen. Returns whether it did.

    Falls back to printing inline whenever taking over the screen would be
    wrong or impossible: no terminal (a pipe, a recording console under test),
    no termios, the viewer switched off, content that already fits, or any
    failure at all on the way in. A viewer that cannot start must still show
    you the diff.
    """
    if os.environ.get(VIEWER_ENV, "").strip().lower() in {"off", "0", "none"}:
        console.print(build(console.width))
        return False
    if not (console.is_terminal and sys.stdin.isatty()):
        console.print(build(console.width))
        return False
    try:
        import termios
        import tty
    except ImportError:  # pragma: no cover - not POSIX
        console.print(build(console.width))
        return False

    size = terminal_size(sys.stdout.fileno())
    lines = render_lines(console, build, size.columns)
    if len(lines) < size.lines:
        # It fits. Taking over the screen for it would be theatre, and it would
        # vanish from the scrollback the moment you quit.
        console.print(build(console.width))
        return False

    fd = sys.stdin.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:  # pragma: no cover - not a real terminal
        console.print(build(console.width))
        return False

    out = sys.stdout
    top = 0
    failed = False
    try:
        tty.setcbreak(fd)
        out.write(_ALT_SCREEN[0] + _CURSOR[0] + _MOUSE[0])
        while True:
            current = terminal_size(fd)
            if current != size:
                # The one thing an external pager could not do: lay it out
                # again for the size the window is now.
                size = current
                lines = render_lines(console, build, size.columns)
            page = max(size.lines - 1, 1)
            top = min(top, max(len(lines) - page, 0))
            _draw(out, lines, top, size.columns, page)

            if not select.select([fd], [], [], 0.25)[0]:
                continue
            data = os.read(fd, 64)
            if not data:
                # stdin closed under us. Without this the select above stays
                # readable for ever and the loop spins at full tilt.
                break
            action = parse_key(data)
            if action == "quit":
                break
            top = scroll_to(top, action, len(lines), page)
    except KeyboardInterrupt:
        pass
    except Exception:
        failed = True
    finally:
        # Restore before anything else prints, or it prints onto a screen that
        # is about to be thrown away.
        out.write(_MOUSE[1] + _CURSOR[1] + _ALT_SCREEN[1])
        out.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    if failed:
        console.print(build(console.width))
        return False
    return True


__all__ = ["VIEWER_ENV", "color_system", "parse_key", "render_lines",
           "scroll_to", "show", "terminal_size"]
