"""Network-gated IPC compatibility sweep (design doc §7.2 and §7.3).

The converter README publishes per-domain compatibility tables and a set of
"solved optimally" claims. Those are the converter's contract with the IPC
collections, but they were produced by hand, so nothing re-checks them when
grounding or the planner changes. These tests encode them.

Both are gated on ``RAILROAD_PDDL_NETWORK_TESTS`` because they need the
GitHub API. Anonymous requests are limited to 60/hour and a full sweep needs
roughly 130, so populate the cache incrementally (it is permanent) or set
``GITHUB_TOKEN``::

    RAILROAD_PDDL_NETWORK_TESTS=1 uv run pytest -q -k ipc_sweep
    RAILROAD_PDDL_NETWORK_TESTS=1 uv run pytest -q -k ipc_solves -m slow

A drifted status is reported for the whole collection at once rather than
failing on the first domain, so one run tells you everything that moved.
"""

import os
import urllib.error

import pytest

from railroad import pddl_converter as pc


class _Unavailable(Exception):
    """The domain could not be fetched, so this run has no verdict on it.

    Kept distinct from a conversion outcome so rate limiting cannot masquerade
    as agreement: a run that could not reach some domains skips rather than
    reporting a green tick that checked less than it appears to.
    """

pytestmark = pytest.mark.skipif(
    not os.environ.get("RAILROAD_PDDL_NETWORK_TESTS"),
    reason="set RAILROAD_PDDL_NETWORK_TESTS=1 to hit the GitHub API",
)


# Status per domain, first instance: "ok", or "<error-kind>:<reason slug>".
# Transcribed from the converter README's compatibility tables.
EXPECTED_STATUS = {
    "ipc-2000": {
        "blocks-strips-typed": "ok",
        "blocks-strips-untyped": "ok",
        "elevator-adl-full-typed": "unsupported:imply-conditions",
        "elevator-adl-simple-typed": "ok",
        "elevator-strips-simple-typed": "ok",
        "elevator-strips-simple-untyped": "ok",
        "freecell-strips-typed": "ok",
        "freecell-strips-untyped": "ok",
        "logistics-strips-typed": "ok",
        "logistics-strips-untyped": "ok",
        "schedule-adl-typed": "ok",
        "schedule-adl-untyped": "ok",
    },
    "ippc-2006": {
        "blocksworld": "ok",
        "drive": "unsupported:disjunctive-preconditions",
        "drive-unrolled": "ok",
        "elevators": "ok",
        "ex-blocksworld": "ok",
        "pitchcatch": "ok",
        "random": "ok",
        "schedule": "ok",
        "tireworld": "ok",
        "zenotravel": "ok",
    },
    "ippc-2008": {
        "2-tireworlds": "parse-error",
        "blocksworld": "ok",
        "boxworld": "ok",
        "ex-blocksworld": "ok",
        "rectangle-tireworld": "unsupported:numeric-effects",
        "schedule": "ok",
        "search-and-rescue": "unsupported:imply-conditions",
        "sysadmin-slp": "unsupported:rewards",
        "triangle-tireworld": "ok",
        "zenotravel": "unsupported:numeric-effects",
    },
}


def _fetch(collection: str, domain: str):
    try:
        return pc.fetch_domain(collection, domain, max_instances=1)
    except urllib.error.HTTPError as exc:
        raise _Unavailable(f"HTTP {exc.code}: {exc.reason}") from exc
    except OSError as exc:  # URLError, DNS, connection reset
        raise _Unavailable(str(exc)) from exc


def _status(collection: str, domain: str) -> str:
    """Convert the domain's first instance and classify the outcome."""
    fetched = _fetch(collection, domain)
    if not fetched.instances:
        return "fetch-error:no instances found"
    instance = fetched.instances[0]
    try:
        pc.load_problem(fetched.domain_for(instance), instance)
    except pc.UnsupportedPDDLError as exc:
        return f"unsupported:{exc.reason}"
    except pc.PDDLParseError:
        return "parse-error"
    return "ok"


@pytest.mark.parametrize("collection", sorted(EXPECTED_STATUS))
def test_ipc_sweep_matches_published_table(collection):
    """Every domain converts (or fails) exactly as the README claims.

    Passing means *every* domain in the collection was reached and agreed. A
    confirmed disagreement fails even if other domains were unreachable; an
    incomplete run skips.
    """
    expected = EXPECTED_STATUS[collection]
    drift, unavailable = {}, {}
    for domain in sorted(expected):
        try:
            actual = _status(collection, domain)
        except _Unavailable as exc:
            unavailable[domain] = str(exc)
            continue
        if actual != expected[domain]:
            drift[domain] = (expected[domain], actual)

    if drift:
        pytest.fail(
            f"{collection} diverges from the README table:\n"
            + "\n".join(
                f"  {d}: expected {want!r}, got {got!r}"
                for d, (want, got) in sorted(drift.items())
            )
        )
    if unavailable:
        pytest.skip(
            f"{len(unavailable)}/{len(expected)} {collection} domains "
            f"unreachable (GitHub API rate limit? set GITHUB_TOKEN): "
            + ", ".join(sorted(unavailable))
        )


# Instances the README reports as solved, with the planner and, where a cost
# is pinned, the value this repo actually produces under seed 0. `sim_time` is
# total cost under the cost->duration mapping and plan length under the unit
# mapping. A pinned number is a regression guard on *our* output, not a claim
# about the instance's true optimum.
SOLVED_INSTANCES = [
    ("ipc-2000", "blocks-strips-typed", "mcts", None),
    # 21 steps, stable across seeds 0-2 and independent of relevance
    # projection. The README previously recorded "20 steps, optimal"; that
    # does not reproduce, so the README now records 21.
    ("ipc-2000", "logistics-strips-typed", "mcts", 21.0),
    ("ipc-2000", "elevator-adl-simple-typed", "mcts", None),
    ("ippc-2006", "blocksworld", "greedy", None),
    ("ippc-2008", "triangle-tireworld", "greedy", None),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "collection, domain, planner, expected_cost",
    SOLVED_INSTANCES,
    ids=[f"{c}-{d}" for c, d, _, _ in SOLVED_INSTANCES],
)
def test_ipc_solves_first_instance(collection, domain, planner, expected_cost):
    """Solved-instance regression (design doc §7.3), under a fixed seed."""
    try:
        fetched = _fetch(collection, domain)
    except _Unavailable as exc:
        pytest.skip(f"{collection}/{domain} unreachable: {exc}")
    if not fetched.instances:
        pytest.skip(f"{collection}/{domain} has no instances")

    instance = fetched.instances[0]
    problem = pc.load_problem(fetched.domain_for(instance), instance)
    result = pc.solve(problem, seed=0, planner=planner, max_iterations=4000)

    assert result.success, result.failure_reason
    if expected_cost is not None:
        assert result.sim_time == pytest.approx(expected_cost)
