# conftest.py
import pathlib
import pytest
from filelock import FileLock
from procthor import ensure_all_resources

DATA_DIR = pathlib.Path(__file__).parent / "resources"

def pytest_configure(config):
    # This hook runs in both controller and workers, but only
    # the controller process has no 'workerinput' attribute.
    if not hasattr(config, "workerinput"):
        # controller process
        if not (DATA_DIR / ".download_complete").exists():
            ensure_all_resources()
            (DATA_DIR / ".download_complete").touch()
