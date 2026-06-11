"""Portable offscreen OpenGL context creation."""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any

import moderngl


def _create(backend: str | None) -> moderngl.Context:
    global _current_ctx
    kwargs: dict[str, Any] = {} if backend is None else {"backend": backend}
    ctx = moderngl.create_context(standalone=True, require=330, **kwargs)
    _current_ctx = ctx  # creation makes the new context current
    return ctx


def _create_software() -> moderngl.Context:
    """CPU-only rendering via Mesa's llvmpipe software rasterizer.

    Linux only: macOS has no software OpenGL implementation, so there the
    request degrades to the default (GPU) backend with a warning. The
    environment variable must be set before Mesa selects a driver, so
    forcing CPU only works reliably if no hardware context was created
    earlier in this process.
    """
    if sys.platform == "darwin":
        warnings.warn(
            "The 'cpu' backend needs Mesa (llvmpipe), which macOS does not "
            "ship -- there is no software OpenGL on macOS. Falling back to "
            "the default CGL backend.")
        return _create(None)
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    try:
        return _create(None)
    except Exception:
        return _create("egl")


def create_gl_context(backend: str | None = None) -> moderngl.Context:
    """Create a standalone (offscreen) GL 3.3 core context.

    Tries the platform default first (CGL on macOS, GLX/X11 on Linux with a
    display), then falls back to EGL for headless Linux. Set
    ``RAILSIM_GL_BACKEND`` (or pass ``backend``) to pin one explicitly:
    one of ``cgl``, ``egl``, ``glx``, or ``cpu`` to force Mesa's software
    rasterizer when no (working) GPU is available.
    """
    backend = backend or os.environ.get("RAILSIM_GL_BACKEND")
    if backend == "cpu":
        return _create_software()
    if backend:
        return _create(backend)

    try:
        return _create(None)
    except Exception:
        # Typical headless-Linux failure: no display for GLX. Try EGL.
        return _create("egl")


# The context railsim last made (or created) current. Context switches are
# not free (CGL flushes), so `make_current` skips redundant switches.
_current_ctx: moderngl.Context | None = None


def make_current(ctx: moderngl.Context) -> None:
    """Make ``ctx`` the thread's current GL context.

    GL commands go to whichever context is current, and creating or
    releasing *another* context (e.g. a second Simulator) can displace this
    one. Call before issuing GL work on a context that may not be current.
    """
    global _current_ctx
    if ctx is _current_ctx:
        return
    ctx.__enter__()
    _current_ctx = ctx


def release_context(ctx: moderngl.Context) -> None:
    """Release a context; releasing leaves no context current, so the
    `make_current` tracker is cleared."""
    global _current_ctx
    ctx.release()
    if _current_ctx is ctx:
        _current_ctx = None
