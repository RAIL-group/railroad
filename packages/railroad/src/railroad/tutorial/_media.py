"""Plot and video paths, passed to ``demo.py`` on its own command line.

The later steps render a trajectory plot and an MP4. Rather than reach for
environment variables, the tutorial forwards ``--video``/``--plot`` straight
through to the script, so ``python demo.py --video out.mp4`` by hand does
exactly what pressing the key does.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence, TypedDict


class MediaArgs(TypedDict):
    """Exactly the ``show_plots`` keywords the steps set, so ``**`` type-checks."""

    save_plot: Optional[str]
    save_video: Optional[str]


def media_args(argv: Optional[Sequence[str]] = None) -> MediaArgs:
    """Parse ``--video PATH`` / ``--plot PATH`` into ``show_plots`` kwargs.

    Unknown arguments are ignored: this is a convenience for the step scripts,
    not a real CLI, and a demo should never die on an argument it did not
    expect mid-talk.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    parsed: MediaArgs = {"save_plot": None, "save_video": None}
    if "--video" in args and args.index("--video") + 1 < len(args):
        parsed["save_video"] = args[args.index("--video") + 1]
    if "--plot" in args and args.index("--plot") + 1 < len(args):
        parsed["save_plot"] = args[args.index("--plot") + 1]
    return parsed
