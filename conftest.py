# conftest.py
import pathlib
import pytest

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
