"""Keep the emoji module's process-wide caches from leaking between tests.

`railroad.plotting.emoji` memoizes fonts, matches and providers by path, and
these tests repoint `SYSTEM_FONT_PATHS` and `DEFAULT_RESOURCES_BASE` at
temporary directories. `monkeypatch` restores the attributes; only this
restores what was cached while they were patched.
"""

import pytest

from railroad.plotting import emoji


@pytest.fixture(autouse=True)
def reset_emoji_caches():
    emoji._reset_caches()
    yield
    emoji._reset_caches()
