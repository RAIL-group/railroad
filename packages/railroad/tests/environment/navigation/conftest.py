"""Auto-skip navigation tests when the navigation extra is not installed."""

import pytest

from railroad.environment.navigation import is_available

if not is_available():
    pytest.skip("navigation extra not installed", allow_module_level=True)
