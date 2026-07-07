"""Download IPC benchmark problems from GitHub mirrors, with local caching.

Sources (fetched through the GitHub API, so only ``api.github.com`` needs to
be reachable; set ``GITHUB_TOKEN`` to raise the 60 requests/hour anonymous
rate limit):

- ``ipc-<year>`` (deterministic, 1998-2014): ``potassco/pddl-instances``
- ``ippc-2006`` / ``ippc-2008`` (PPDDL probabilistic tracks):
  ``probfd/ppddl-benchmarks``

Files are cached under ``$RAILROAD_PDDL_CACHE_DIR`` (default
``~/.cache/railroad/pddl``) and never re-downloaded once present.
"""

import base64
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

CACHE_DIR_ENV = "RAILROAD_PDDL_CACHE_DIR"


@dataclass(frozen=True)
class CollectionSpec:
    repo: str
    root: str  # repo-relative directory whose children are domain directories


COLLECTIONS: Dict[str, CollectionSpec] = {
    **{
        f"ipc-{year}": CollectionSpec("potassco/pddl-instances", f"ipc-{year}/domains")
        for year in (1998, 2000, 2002, 2004, 2006, 2008, 2011, 2014)
    },
    "ippc-2006": CollectionSpec("probfd/ppddl-benchmarks", "ippc06/problems"),
    "ippc-2008": CollectionSpec("probfd/ppddl-benchmarks", "ippc08/domains"),
}


@dataclass
class FetchedDomain:
    collection: str
    domain: str
    directory: Path
    # Shared domain file, if the domain has a single domain.pddl.
    domain_file: Optional[Path]
    instances: List[Path] = field(default_factory=list)
    # For domains with per-instance domain files: instance path -> domain path.
    per_instance_domains: Dict[Path, Path] = field(default_factory=dict)

    def domain_for(self, instance: Path) -> Path:
        """The domain file to use with the given instance file."""
        if instance in self.per_instance_domains:
            return self.per_instance_domains[instance]
        if self.domain_file is None:
            raise FileNotFoundError(
                f"No domain file for instance {instance.name} in "
                f"{self.collection}/{self.domain}"
            )
        return self.domain_file


def cache_dir() -> Path:
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override)
    return Path.home() / ".cache" / "railroad" / "pddl"


def _spec(collection: str) -> CollectionSpec:
    if collection not in COLLECTIONS:
        known = ", ".join(sorted(COLLECTIONS))
        raise KeyError(f"Unknown collection {collection!r}; known: {known}")
    return COLLECTIONS[collection]


# ============================================================================
#  GitHub API plumbing
# ============================================================================


def _api_request(url: str) -> dict | list:
    request = urllib.request.Request(url)
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("User-Agent", "railroad-pddl-converter")
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode())


def _list_dir(repo: str, path: str, cache_file: Path) -> List[dict]:
    """List a repo directory, caching the (name, type, url) listing locally."""
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    entries = _api_request(f"https://api.github.com/repos/{repo}/contents/{path}")
    assert isinstance(entries, list)
    slim = [
        {"name": e["name"], "type": e["type"], "git_url": e.get("git_url")}
        for e in entries
    ]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(slim))
    tmp.replace(cache_file)
    return slim


def _fetch_file(repo: str, path: str, git_url: Optional[str], dest: Path) -> Path:
    """Download one file (base64 via the contents or blob API) into the cache."""
    if dest.exists():
        return dest
    data = _api_request(f"https://api.github.com/repos/{repo}/contents/{path}")
    assert isinstance(data, dict)
    if data.get("content"):
        content = base64.b64decode(data["content"])
    else:
        # Files over ~1 MB come back with empty content; use the blob API.
        blob_url = data.get("git_url") or git_url
        if not blob_url:
            raise IOError(f"No content or blob URL for {repo}/{path}")
        blob = _api_request(blob_url)
        assert isinstance(blob, dict)
        content = base64.b64decode(blob["content"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(dest)
    return dest


def _natural_key(name: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


# ============================================================================
#  Public API
# ============================================================================


def list_domains(collection: str) -> List[str]:
    """List domain names available in a collection."""
    spec = _spec(collection)
    listing = _list_dir(
        spec.repo, spec.root, cache_dir() / collection / ".listing.json"
    )
    return sorted(e["name"] for e in listing if e["type"] == "dir")


def fetch_domain(
    collection: str, domain: str, max_instances: Optional[int] = None
) -> FetchedDomain:
    """Download (or reuse cached) domain + instance files for one domain."""
    spec = _spec(collection)
    local_dir = cache_dir() / collection / domain
    repo_dir = f"{spec.root}/{domain}"
    listing = _list_dir(spec.repo, repo_dir, local_dir / ".listing.json")
    by_name = {e["name"]: e for e in listing}

    result = FetchedDomain(collection, domain, local_dir, domain_file=None)

    if "domain.pddl" in by_name:
        result.domain_file = _fetch_file(
            spec.repo, f"{repo_dir}/domain.pddl", None, local_dir / "domain.pddl"
        )

    # Instance files: either an instances/ subdirectory (potassco layout) or
    # loose .pddl files next to domain.pddl (ppddl-benchmarks layout).
    if "instances" in by_name and by_name["instances"]["type"] == "dir":
        instance_dir = f"{repo_dir}/instances"
        instance_listing = _list_dir(
            spec.repo, instance_dir, local_dir / "instances" / ".listing.json"
        )
        instance_names = sorted(
            (e["name"] for e in instance_listing
             if e["type"] == "file" and e["name"].endswith(".pddl")),
            key=_natural_key,
        )
        if max_instances is not None:
            instance_names = instance_names[:max_instances]
        for name in instance_names:
            result.instances.append(
                _fetch_file(
                    spec.repo,
                    f"{instance_dir}/{name}",
                    None,
                    local_dir / "instances" / name,
                )
            )
    else:
        loose = sorted(
            (e["name"] for e in listing
             if e["type"] == "file"
             and e["name"].endswith(".pddl")
             and e["name"] != "domain.pddl"),
            key=_natural_key,
        )
        if max_instances is not None:
            loose = loose[:max_instances]
        for name in loose:
            result.instances.append(
                _fetch_file(spec.repo, f"{repo_dir}/{name}", None, local_dir / name)
            )

    # Potassco's per-instance-domain layout: a domains/ subdir with files
    # named like the instances (instance-N.pddl <-> domain-N.pddl).
    if result.domain_file is None and "domains" in by_name:
        domains_dir = f"{repo_dir}/domains"
        domain_listing = _list_dir(
            spec.repo, domains_dir, local_dir / "domains" / ".listing.json"
        )
        available = {e["name"] for e in domain_listing if e["type"] == "file"}
        for instance in result.instances:
            candidate = instance.name.replace("instance", "domain")
            if candidate in available:
                result.per_instance_domains[instance] = _fetch_file(
                    spec.repo,
                    f"{domains_dir}/{candidate}",
                    None,
                    local_dir / "domains" / candidate,
                )
    return result
