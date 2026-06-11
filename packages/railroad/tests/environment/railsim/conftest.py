"""Auto-skip railsim tests when the railsim extra is not installed."""

import pytest

from railroad.environment.railsim import is_available

if not is_available():
    pytest.skip("railsim extra not installed", allow_module_level=True)
