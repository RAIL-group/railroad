"""The cross-trial policy-selection layer is a deferred, explicit stub."""

from __future__ import annotations

import pytest

from railroad.replay import CandidatePolicy, select_policy


def test_select_policy_is_not_implemented() -> None:
    """The selection layer is deferred; calling it fails loudly, not silently."""
    with pytest.raises(NotImplementedError):
        select_policy([], [CandidatePolicy(name="a"), CandidatePolicy(name="b")])
