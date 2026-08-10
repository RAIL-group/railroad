"""Serving plots and videos beside the benchmark results.

The dashboard renders each run's ``plot.jpg`` from MLflow, but a video is not
an MLflow artifact -- it is written to a directory by whoever ran the demo.
This adds a plain index over that directory on the dashboard's own port, so a
talk needs one browser tab rather than a browser and a video player.

The directory is ``./tutorial-media`` relative to the working directory, which
means the dashboard and whatever wrote the file have to agree on it -- the same
convention ``mlflow.db`` and the ProcTHOR scene cache already follow. Override
with ``RAILROAD_TUTORIAL_MEDIA_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_DIR = "RAILROAD_TUTORIAL_MEDIA_DIR"
DEFAULT_DIRNAME = "tutorial-media"

VIDEO_SUFFIXES = {".mp4", ".webm", ".mov"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif"}


def media_dir() -> Path:
    """Where demo plots and videos are looked for."""
    return Path(os.environ.get(ENV_DIR) or Path.cwd() / DEFAULT_DIRNAME)


def _page(body: str) -> str:
    return (
        "<!doctype html><meta charset='utf-8'><title>tutorial media</title>"
        "<style>body{background:#1e1e2e;color:#cdd6f4;font-family:monospace;"
        "margin:2rem}h1{font-size:1rem}figure{margin:0 0 2rem}"
        "figcaption{margin-bottom:.4rem}video,img{max-width:min(100%,720px);"
        "border:1px solid #45475a}a{color:#89b4fa}</style>" + body
    )


def register_media_routes(app) -> None:
    """Add ``/media/`` and ``/media/<file>`` to the dashboard's Flask server."""
    from flask import send_from_directory
    from markupsafe import escape

    @app.server.route("/media/")
    def media_index():  # pragma: no cover - exercised by hand during a talk
        directory = media_dir()
        if not directory.is_dir():
            return _page(
                f"<h1>no media yet</h1><p>Nothing in <code>{escape(str(directory))}"
                "</code>. Render some with <code>railroad tutorial run "
                "--video house.mp4</code>.</p>"
            )
        files = sorted(
            (p for p in directory.iterdir() if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not files:
            return _page(f"<h1>no media yet</h1><p>{escape(str(directory))}</p>")

        parts = [f"<h1>{escape(str(directory))}</h1>"]
        for path in files:
            name = escape(path.name)
            suffix = path.suffix.lower()
            if suffix in VIDEO_SUFFIXES:
                element = f"<video controls src='/media/{name}'></video>"
            elif suffix in IMAGE_SUFFIXES:
                element = f"<img src='/media/{name}'>"
            else:
                element = f"<a href='/media/{name}'>{name}</a>"
            parts.append(
                f"<figure><figcaption>{name}</figcaption>{element}</figure>"
            )
        return _page("".join(parts))

    @app.server.route("/media/<path:filename>")
    def serve_media(filename: str):  # pragma: no cover - as above
        # send_from_directory rejects paths that escape the directory.
        return send_from_directory(str(media_dir()), filename)
