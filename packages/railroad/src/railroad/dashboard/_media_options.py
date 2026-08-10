"""The plot/video command-line options, described once.

``railroad example <name>`` has offered ``--save-plot``, ``--save-video`` and
their friends since the examples existed, and they map one-to-one onto
:meth:`PlannerDashboard.show_plots` keywords. Anything else that plans and then
draws wants exactly the same five, spelled exactly the same way -- so they are
described here as data rather than declared twice.

``railroad.cli`` turns them into click options for every example; the tutorial
turns them into argparse arguments for its ``demo.py``. Adding a sixth here
gives it to both.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TypedDict


class MediaOption(TypedDict, total=False):
    """One flag, in the shape ``railroad.examples.OptionInfo`` already uses."""

    name: str
    """The flag as typed, e.g. ``--save-plot``."""

    param_name: str
    """The ``show_plots`` keyword it sets."""

    is_flag: bool
    type: Any
    default: Any
    help: str

    is_path: bool
    """True when the value names a file, so a caller may relocate bare names."""


MEDIA_OPTIONS: List[MediaOption] = [
    {
        "name": "--save-plot",
        "param_name": "save_plot",
        "default": None,
        "help": "Save trajectory plot to file (e.g. out.png)",
        "is_path": True,
    },
    {
        "name": "--show-plot",
        "param_name": "show_plot",
        "is_flag": True,
        "default": False,
        "help": "Show trajectory plot interactively",
    },
    {
        "name": "--save-video",
        "param_name": "save_video",
        "default": None,
        "help": "Save trajectory animation to file (e.g. out.mp4)",
        "is_path": True,
    },
    {
        "name": "--video-fps",
        "param_name": "video_fps",
        "type": int,
        "default": 60,
        "help": "Video frames per second",
    },
    {
        "name": "--video-dpi",
        "param_name": "video_dpi",
        "type": int,
        "default": 150,
        "help": "Video resolution in dots per inch",
    },
]

MEDIA_OPTION_NAMES: List[str] = [option["name"] for option in MEDIA_OPTIONS]


def media_kwargs(
    values: Dict[str, Any], *, relocate: Optional[Any] = None
) -> Dict[str, Any]:
    """Pick the ``show_plots`` keywords out of *values*.

    Only what was actually asked for: an option left at its default is dropped,
    so the caller's own defaults win and ``show_plots`` returns immediately
    when nothing was requested.

    *relocate* is an optional ``str -> str`` applied to path-valued options,
    for callers that want bare filenames to land somewhere particular.
    """
    kwargs: Dict[str, Any] = {}
    for option in MEDIA_OPTIONS:
        key = option["param_name"]
        value = values.get(key)
        if value is None or value == option.get("default"):
            continue
        if relocate is not None and option.get("is_path"):
            value = relocate(value)
        kwargs[key] = value
    return kwargs


def add_to_argparse(parser: Any, only: Optional[Sequence[str]] = None) -> None:
    """Declare these options on an ``argparse`` parser.

    Used by scripts that are their own command line rather than a click
    subcommand -- the tutorial's ``demo.py``, for one.
    """
    for option in MEDIA_OPTIONS:
        if only is not None and option["name"] not in only:
            continue
        kwargs: Dict[str, Any] = {
            "dest": option["param_name"],
            "default": option.get("default"),
            "help": option.get("help", ""),
        }
        if option.get("is_flag"):
            kwargs["action"] = "store_true"
        else:
            kwargs["metavar"] = "PATH" if option.get("is_path") else "N"
            if "type" in option:
                kwargs["type"] = option["type"]
        parser.add_argument(option["name"], **kwargs)
