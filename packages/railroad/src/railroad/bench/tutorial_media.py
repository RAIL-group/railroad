"""Shared location for tutorial plots/videos.

The tutorial script (single/TUI mode) writes trajectory plots and videos here;
the benchmark dashboard serves the same directory at ``/media/`` so they can
be viewed remotely on the dashboard's port.

Default is ``./tutorial_media`` relative to the current working directory,
matching the cwd-relative convention already used for ``mlflow.db``. Override
with the ``RAILROAD_TUTORIAL_MEDIA_DIR`` environment variable. Both the script
and the dashboard must run with the same cwd (or the same override) to agree.
"""

import os
from pathlib import Path

ENV_VAR = "RAILROAD_TUTORIAL_MEDIA_DIR"
DEFAULT_DIRNAME = "tutorial_media"


def media_dir() -> Path:
    """Return the media directory, creating it if necessary."""
    d = Path(
        os.environ.get(ENV_VAR, DEFAULT_DIRNAME)
    ).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d
