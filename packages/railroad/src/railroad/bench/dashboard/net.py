"""Working out which URLs the dashboard is actually reachable at.

The server has always bound every interface, so viewing it from another
machine already worked -- it just printed ``127.0.0.1`` and left you to guess
the rest. This finds the addresses that will work and prints them, which is the
whole of the "how do I see this remotely" problem for anyone on a VPN or a LAN.

Tailscale gets named specially because its addresses come out of the shared
CGNAT range (100.64.0.0/10) and are meaningless to look at otherwise, but
nothing here is Tailscale-specific: any address the host answers on is listed.
"""

from __future__ import annotations

import ipaddress
import shutil
import socket
import subprocess
from typing import List, Optional, Tuple

DEFAULT_PORT = 8050
ALL_INTERFACES = "0.0.0.0"

TAILSCALE_RANGE = ipaddress.ip_network("100.64.0.0/10")
"""Carrier-grade NAT space, which is where Tailscale hands out addresses."""


def tailscale_address() -> Optional[str]:
    """This host's tailnet IPv4 address, if it is on a tailnet."""
    exe = shutil.which("tailscale")
    if exe is None:
        return None
    try:
        result = subprocess.run(
            [exe, "ip", "-4"], capture_output=True, text=True, timeout=3.0
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return None


def _routable_address() -> Optional[str]:
    """The address this host would use to reach the outside world.

    Opening a UDP socket sends nothing; it just asks the routing table which
    local address it would pick. That is the usual way to find the LAN address
    without enumerating interfaces, which the standard library cannot do
    portably.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 1))  # TEST-NET-1; no packets are sent
            return sock.getsockname()[0]
    except OSError:
        return None


def _interface_addresses() -> List[str]:
    """Every IPv4 address the host holds, via ``ip``, where that exists.

    The standard library cannot enumerate interfaces portably, and the
    tailscale CLI is not always installed even when the interface is up, so
    this is the second way of finding a tailnet address.
    """
    exe = shutil.which("ip")
    if exe is None:
        return []
    try:
        result = subprocess.run(
            [exe, "-4", "-o", "addr", "show"],
            capture_output=True, text=True, timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    addresses = []
    for line in result.stdout.splitlines():
        fields = line.split()
        # "3: tailscale0    inet 100.x.y.z/32 scope global ..."
        if "inet" in fields:
            value = fields[fields.index("inet") + 1]
            addresses.append(value.split("/")[0])
    return addresses


def _hostname_addresses() -> List[str]:
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return []
    # sockaddr is (host, port) for AF_INET; keep only the host.
    return [str(info[4][0]) for info in infos]


def _label(address: str) -> str:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return "other"
    if parsed.is_loopback:
        return "this machine"
    if parsed in TAILSCALE_RANGE:
        return "tailnet"
    if parsed.is_private:
        return "local network"
    return "public"


def reachable_addresses() -> List[Tuple[str, str]]:
    """``(address, label)`` for everywhere the dashboard can be reached.

    Ordered most useful first: loopback, then the tailnet, then the LAN.
    """
    found: List[str] = ["127.0.0.1"]
    tailnet = tailscale_address()
    if tailnet:
        found.append(tailnet)
    found.extend(_interface_addresses())
    routable = _routable_address()
    if routable:
        found.append(routable)
    found.extend(_hostname_addresses())

    seen, ordered = set(), []
    for address in found:
        label = _label(address)
        # 127.0.0.1 is listed once; other loopback aliases (a hostname mapped
        # to 127.0.1.1, say) tell the reader nothing.
        if address in seen or (label == "this machine" and address != "127.0.0.1"):
            continue
        seen.add(address)
        ordered.append((address, label))
    rank = {"this machine": 0, "tailnet": 1, "local network": 2}
    ordered.sort(key=lambda pair: rank.get(pair[1], 3))
    return ordered


def resolve_host(requested: str) -> str:
    """Turn a ``--host`` value into an address to bind.

    ``auto`` binds every interface, which is what it has always done and what
    makes a remote view work without thinking about it. ``tailscale`` binds
    only the tailnet address, for when you would rather it not answer on the
    coffee shop's wifi.
    """
    if requested == "auto":
        return ALL_INTERFACES
    if requested == "tailscale":
        address = tailscale_address()
        if address is None:
            raise RuntimeError(
                "no tailnet address found. Is tailscale running, and is its CLI "
                "on PATH? Use --host 0.0.0.0 to bind every interface instead."
            )
        return address
    return requested


def is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in ("localhost", "")


def url_lines(host: str, port: int) -> List[str]:
    """Human-readable URLs for a server bound to *host*."""
    if host != ALL_INTERFACES:
        return [f"  http://{host}:{port}/"]
    return [
        f"  http://{address}:{port}/".ljust(38) + f"({label})"
        for address, label in reachable_addresses()
    ]
