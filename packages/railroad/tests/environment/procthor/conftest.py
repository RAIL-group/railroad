"""Auto-skip procthor tests when the procthor extra is not installed."""

import pytest

from railroad.environment.procthor import is_available

if not is_available():
    pytest.skip("procthor extra not installed", allow_module_level=True)
