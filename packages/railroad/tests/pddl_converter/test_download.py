import json

import pytest

from railroad.pddl_converter.download import (
    CACHE_DIR_ENV,
    _natural_key,
    fetch_domain,
    list_domains,
)


def test_unknown_collection_rejected():
    with pytest.raises(KeyError):
        list_domains("ipc-1893")


def test_natural_instance_ordering():
    names = ["instance-10.pddl", "instance-2.pddl", "instance-1.pddl"]
    assert sorted(names, key=_natural_key) == [
        "instance-1.pddl", "instance-2.pddl", "instance-10.pddl",
    ]


def _write_listing(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))


def test_fetch_domain_offline_from_cache(monkeypatch, tmp_path):
    """A fully-populated cache is used without any network access."""
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
    root = tmp_path / "ipc-2000" / "blocks"
    _write_listing(
        root / ".listing.json",
        [
            {"name": "domain.pddl", "type": "file", "git_url": None},
            {"name": "instances", "type": "dir", "git_url": None},
        ],
    )
    _write_listing(
        root / "instances" / ".listing.json",
        [
            {"name": "instance-2.pddl", "type": "file", "git_url": None},
            {"name": "instance-1.pddl", "type": "file", "git_url": None},
        ],
    )
    (root / "domain.pddl").write_text("(define (domain blocks))")
    (root / "instances" / "instance-1.pddl").write_text("(define (problem p1))")
    (root / "instances" / "instance-2.pddl").write_text("(define (problem p2))")

    fetched = fetch_domain("ipc-2000", "blocks")
    assert fetched.domain_file == root / "domain.pddl"
    assert [p.name for p in fetched.instances] == [
        "instance-1.pddl", "instance-2.pddl",
    ]
    assert fetched.domain_for(fetched.instances[0]) == root / "domain.pddl"


def test_fetch_domain_loose_instance_layout(monkeypatch, tmp_path):
    """ppddl-benchmarks layout: pNN.pddl files next to domain.pddl."""
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
    root = tmp_path / "ippc-2008" / "tireworld"
    _write_listing(
        root / ".listing.json",
        [
            {"name": "domain.pddl", "type": "file", "git_url": None},
            {"name": "p01.pddl", "type": "file", "git_url": None},
            {"name": "p02.pddl", "type": "file", "git_url": None},
        ],
    )
    (root / "domain.pddl").write_text("(define (domain t))")
    (root / "p01.pddl").write_text("(define (problem p1))")
    (root / "p02.pddl").write_text("(define (problem p2))")

    fetched = fetch_domain("ippc-2008", "tireworld", max_instances=1)
    assert [p.name for p in fetched.instances] == ["p01.pddl"]


def test_fetch_domain_per_instance_domains(monkeypatch, tmp_path):
    """potassco layout variant: per-instance domain files in domains/."""
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
    root = tmp_path / "ipc-2000" / "weird"
    _write_listing(
        root / ".listing.json",
        [
            {"name": "domains", "type": "dir", "git_url": None},
            {"name": "instances", "type": "dir", "git_url": None},
        ],
    )
    _write_listing(
        root / "instances" / ".listing.json",
        [{"name": "instance-1.pddl", "type": "file", "git_url": None}],
    )
    _write_listing(
        root / "domains" / ".listing.json",
        [{"name": "domain-1.pddl", "type": "file", "git_url": None}],
    )
    (root / "instances" / "instance-1.pddl").write_text("(define (problem p1))")
    (root / "domains" / "domain-1.pddl").write_text("(define (domain d1))")

    fetched = fetch_domain("ipc-2000", "weird")
    assert fetched.domain_file is None
    instance = fetched.instances[0]
    assert fetched.domain_for(instance) == root / "domains" / "domain-1.pddl"


# No test here reaches the network. The download layer is exercised against
# monkeypatched listings and the on-disk cache; checking that GitHub still
# serves a given IPC domain tests github.com, not railroad, and pays for it
# with a 60/hour anonymous rate limit. Use `railroad pddl check <collection>`
# when you want to know what upstream is currently shipping.
