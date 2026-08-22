"""Keep the emoji module's process-wide caches from leaking between tests.

`railroad.plotting.emoji` memoizes fonts, matches and providers by path, and
`test_emoji_glyphs.py` repoints `SYSTEM_FONT_PATHS` and `DEFAULT_RESOURCES_BASE`
at temporary directories. `monkeypatch` restores the attributes; only this
restores what was cached while they were patched.

Deliberately *not* autouse. `_MODELS` holds the SentenceTransformer, so clearing
it around every test in this directory forced a full model reload per test --
4.6s of the 5.3s suite went to `test_emoji_matching.py`, which never patches
anything. Only `test_emoji_glyphs.py` needs the reset, and it requests this via
a module-level `usefixtures`; running before *and* after each of its tests keeps
the pollution it creates from reaching the rest of the directory.
"""

import pytest

from railroad.plotting import emoji


@pytest.fixture
def reset_emoji_caches():
    emoji._reset_caches()
    yield
    emoji._reset_caches()
