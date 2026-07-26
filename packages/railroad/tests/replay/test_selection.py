"""The cross-trial policy-selection layer is a deferred, explicit stub."""

from __future__ import annotations

import pytest

from railroad.experimental.unknown_search import FixedObjectFind
from railroad.replay import select_policy


def test_select_policy_is_not_implemented() -> None:
    """The selection layer is deferred; calling it fails loudly, not silently."""
    with pytest.raises(NotImplementedError):
        select_policy([], [FixedObjectFind(0.5), FixedObjectFind(0.9)])
