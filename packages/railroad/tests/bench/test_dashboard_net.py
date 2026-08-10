"""Tests for working out where the dashboard can be reached.

The point of this module is that someone on a tailnet is told the URL that
works, instead of the ``127.0.0.1`` that does not. Address discovery itself is
environment-specific, so the sources are stubbed and what is tested is the
classification, ordering and de-duplication built on top of them.
"""

import pytest

from railroad.bench.dashboard import net


@pytest.fixture
def sources(monkeypatch):
    """Stub every address source; each test fills in what it needs."""
    state = {"tailscale": None, "interfaces": [], "routable": None, "hostname": []}
    monkeypatch.setattr(net, "tailscale_address", lambda: state["tailscale"])
    monkeypatch.setattr(net, "_interface_addresses", lambda: state["interfaces"])
    monkeypatch.setattr(net, "_routable_address", lambda: state["routable"])
    monkeypatch.setattr(net, "_hostname_addresses", lambda: state["hostname"])
    return state


def test_labels_a_tailnet_address(sources):
    sources["tailscale"] = "100.101.102.103"
    assert ("100.101.102.103", "tailnet") in net.reachable_addresses()


def test_finds_the_tailnet_address_without_the_cli(sources):
    """The interface is often up when the tailscale CLI is not installed."""
    sources["interfaces"] = ["192.168.1.20", "100.90.80.70"]
    labels = dict(net.reachable_addresses())
    assert labels["100.90.80.70"] == "tailnet"
    assert labels["192.168.1.20"] == "local network"


def test_orders_loopback_then_tailnet_then_lan(sources):
    sources["tailscale"] = "100.100.1.2"
    sources["routable"] = "192.168.0.5"
    assert [label for _address, label in net.reachable_addresses()] == [
        "this machine", "tailnet", "local network",
    ]


def test_drops_duplicates_and_extra_loopback_aliases(sources):
    """Debian maps the hostname to 127.0.1.1; listing it helps nobody."""
    sources["tailscale"] = "100.100.1.2"
    sources["interfaces"] = ["100.100.1.2", "127.0.0.1"]
    sources["hostname"] = ["127.0.1.1", "127.0.0.2"]
    addresses = [address for address, _label in net.reachable_addresses()]
    assert addresses == ["127.0.0.1", "100.100.1.2"]


def test_resolve_host_auto_binds_everything():
    assert net.resolve_host("auto") == net.ALL_INTERFACES


def test_resolve_host_tailscale_picks_the_tailnet_address(sources):
    sources["tailscale"] = "100.101.5.6"
    assert net.resolve_host("tailscale") == "100.101.5.6"


def test_resolve_host_tailscale_explains_itself_when_absent(sources):
    with pytest.raises(RuntimeError, match="tailscale running"):
        net.resolve_host("tailscale")


def test_resolve_host_passes_an_explicit_address_through():
    assert net.resolve_host("127.0.0.1") == "127.0.0.1"


@pytest.mark.parametrize(
    "host,expected",
    [("127.0.0.1", True), ("localhost", True), ("", True),
     ("0.0.0.0", False), ("100.100.1.2", False)],
)
def test_is_loopback_decides_whether_the_debugger_is_safe(host, expected):
    """Only loopback keeps the interactive debugger, which runs code."""
    assert net.is_loopback(host) is expected


def test_url_lines_for_a_specific_bind_names_only_that_host():
    assert net.url_lines("100.100.1.2", 9000) == ["  http://100.100.1.2:9000/"]


def test_url_lines_for_every_interface_lists_them_all(sources):
    sources["tailscale"] = "100.100.1.2"
    lines = net.url_lines(net.ALL_INTERFACES, 8050)
    assert any("100.100.1.2:8050" in line and "tailnet" in line for line in lines)
    assert any("127.0.0.1:8050" in line for line in lines)
