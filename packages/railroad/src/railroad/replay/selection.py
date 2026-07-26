"""Cross-trial policy selection over offline replay — **deferred stub**.

Offline replay reduces *one* deployment plus *one* candidate policy to a
:class:`~railroad.replay.types.ReplayResult` carrying lower-bound costs
(:func:`railroad.replay.run_replay`). *Selection* is the layer on top: replay a
set of candidates over the same recording — and, across many recordings,
aggregate their bound evidence — to pick the policy that is data-efficiently
justified.

The single-recording comparison already works today with the shipped API: replay
each candidate over one log and rank by bound. Callers can do that directly, and
``scripts/replay/point_goal_nav.py`` demonstrates it::

    ranked = sorted(
        ((p, run_replay(build_replay_env(log), p)) for p in candidates),
        key=lambda kv: kv[1].bounds.simply_connected_lb,
    )

What is **not** implemented is the *cross-trial* aggregator that decides a winner
from bound evidence spanning many recordings (the accept/reject rule and its
bulk-run harness). :func:`select_policy` is a placeholder that raises
:class:`NotImplementedError` so the seam is explicit rather than silently missing;
it will be filled in when the selection work lands.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .types import RolloutLog


def select_policy(
    logs: Iterable[RolloutLog],
    candidates: Sequence[Any],
    **replay_kwargs: object,
) -> Any:
    """Pick the best candidate from bound evidence across many recordings.

    **Not implemented.** This is the cross-trial selection layer: replay every
    *candidate* over every recording in *logs*, aggregate the resulting bounds,
    and return the policy whose lower-bound evidence dominates. Deferred to a
    later change; for now, compare candidates over a single recording by calling
    :func:`railroad.replay.run_replay` on a per-candidate
    :func:`railroad.replay.build_replay_env` arena and ranking by bound (see this
    module's docstring and ``point_goal_nav.py``).
    """
    raise NotImplementedError(
        "cross-trial policy selection is not implemented yet; replay each "
        "candidate over one recording with run_replay(build_replay_env(log), p) "
        "and rank by bound (see railroad.replay.selection docstring)."
    )
