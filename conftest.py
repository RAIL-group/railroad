# conftest.py
import pathlib

DATA_DIR = pathlib.Path(__file__).parent / "resources"

def pytest_configure(config):
    # This hook runs in both controller and workers, but only
    # the controller process has no 'workerinput' attribute.
    if not hasattr(config, "workerinput"):
        # controller process
        if not (DATA_DIR / ".download_complete").exists():
            from railroad.environment.procthor import ensure_all_resources
            ensure_all_resources()
            (DATA_DIR / ".download_complete").touch()
        # The glyph assets are not part of the procthor extra, so they are
        # fetched even where that is unavailable. Tests never download; this
        # warms the cache once, in the controller, before any worker looks.
        from railroad.dashboard._sprites.resources import (
            ensure_emoji_font, ensure_emoji_sbert_model,
        )
        ensure_emoji_font()
        ensure_emoji_sbert_model()
