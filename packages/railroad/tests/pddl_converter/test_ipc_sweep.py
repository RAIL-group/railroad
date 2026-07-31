"""Network-gated IPC compatibility sweep.

The table below is the converter's contract with the IPC collections: the
authoritative per-domain status, which the converter README summarises and
defers to. This test re-derives every status from the source repos and fails
if one moves, so grounding changes cannot silently invalidate it.

The scope is deliberately *conversion*, not solving. Whether a planner
happens to reach the goal on a given instance is a planner property, and
pinning it here only couples the converter's test suite to planner quality
(see the feature-example benchmarks for optional end-to-end demos).

Gated on ``RAILROAD_PDDL_NETWORK_TESTS`` because it needs the GitHub API.
Anonymous requests are limited to 60/hour and a full sweep needs roughly 130,
so populate the cache incrementally (it is permanent) or set ``GITHUB_TOKEN``::

    RAILROAD_PDDL_NETWORK_TESTS=1 uv run pytest -q -k ipc_sweep

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
        "sysAdmin-SLP": "unsupported:rewards",  # upstream name is case-sensitive
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

    # Reconcile against the upstream listing first. A domain name that does
    # not exist upstream 404s, which would otherwise land in `unavailable`
    # and skip -- indistinguishable from a rate limit. Names are
    # case-sensitive (ippc-2008 ships `sysAdmin-SLP`).
    try:
        upstream = set(pc.list_domains(collection))
    except OSError as exc:
        pytest.skip(f"cannot list {collection}: {exc}")
    missing = sorted(upstream - set(expected))
    unknown = sorted(set(expected) - upstream)
    assert not (missing or unknown), (
        f"{collection} table is out of sync with the source repo:\n"
        f"  absent from the table: {missing}\n"
        f"  no longer upstream:    {unknown}"
    )

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
