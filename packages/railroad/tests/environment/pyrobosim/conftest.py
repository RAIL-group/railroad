"""Auto-skip pyrobosim tests when the pyrobosim extra is not installed."""

import pytest

from railroad.environment.pyrobosim import is_available

if not is_available():
    pytest.skip("pyrobosim extra not installed", allow_module_level=True)
