"""Auto-skip bench tests when the bench extra is not installed."""

import pytest

try:
    import railroad.bench  # noqa: F401
except ImportError:
    pytest.skip("bench extra not installed", allow_module_level=True)
