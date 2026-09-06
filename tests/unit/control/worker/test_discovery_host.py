"""Host capability probe tests (Phase 1 spec §21).

Network and OS discovery are driven through monkeypatched psutil/platform
fixtures so the assertions are deterministic on any development host; the
Docker probe runs against an injected fake client factory.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from edgeshard.cluster.capability import MemoryModel
from edgeshard.cluster.identity import DeviceKind
from edgeshard.control.worker.agent import assemble_capability
from edgeshard.control.worker.discovery.host import (
    _LINK_FAMILIES,
    HOST_MEMORY_POOL_ID,
    HostCapabilityProbe,
    canonical_architecture,
)
from edgeshard.control.worker.identity import derive_cpu_device_id

WORKER_ID = "11111111-2222-3333-4444-555555555555"


class FakeDockerClient:
    def __init__(self, version: dict[str, Any], info: dict[str, Any]) -> None:
        self._version = version
        self._info = info

    def version(self) -> dict[str, Any]:
        return self._version

    def info(self) -> dict[str, Any]:
        return self._info


def _probe(docker_client_factory: Any = None) -> HostCapabilityProbe:
    return HostCapabilityProbe(WORKER_ID, docker_client_factory=docker_client_factory)


@pytest.mark.parametrize(
    ("machine", "expected"),
    [
        ("x86_64", "x86_64"),
        ("AMD64", "x86_64"),
        ("amd64", "x86_64"),
        ("aarch64", "aarch64"),
        ("arm64", "aarch64"),
        ("riscv64", "riscv64"),
    ],
)
def test_canonical_architecture_normalizes(machine: str, expected: str) -> None:
    assert canonical_architecture(machine) == expected


def test_discover_reports_cpu_device_and_host_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.machine", lambda: "x86_64")

    fragment = _probe(docker_client_factory=lambda: _raise_docker()).discover()

    assert fragment.architecture == "x86_64"
    cpu = fragment.devices
    assert len(cpu) == 1
    assert cpu[0].identity.kind is DeviceKind.CPU
    assert cpu[0].identity.device_id == derive_cpu_device_id(WORKER_ID)
    assert cpu[0].memory_pool_id == HOST_MEMORY_POOL_ID
    assert cpu[0].vendor
    assert cpu[0].model

    assert len(fragment.memory_pools) == 1
    pool = fragment.memory_pools[0]
    assert pool.memory_pool_id == HOST_MEMORY_POOL_ID
    assert pool.model is MemoryModel.SHARED
    assert pool.total_bytes > 0


def _raise_docker() -> Any:
    raise RuntimeError("docker unavailable")


def test_discover_container_runtime_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeDockerClient(
        version={"Version": "27.0.1"},
        info={"Runtimes": {"nvidia": {}, "runc": {}}},
    )
    fragment = _probe(docker_client_factory=lambda: client).discover()
    assert fragment.container_runtime is not None
    assert fragment.container_runtime.runtime == "docker"
    assert fragment.container_runtime.version == "27.0.1"
    assert fragment.container_runtime.nvidia_runtime_available is True


def test_discover_container_runtime_no_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeDockerClient(version={"Version": "24.0"}, info={"Runtimes": {"runc": {}}})
    fragment = _probe(docker_client_factory=lambda: client).discover()
    assert fragment.container_runtime is not None
    assert fragment.container_runtime.nvidia_runtime_available is False


def test_discover_container_runtime_absent_when_daemon_unavailable() -> None:
    fragment = _probe(docker_client_factory=lambda: _raise_docker()).discover()
    assert fragment.container_runtime is None


class Addr:
    def __init__(self, family: int, address: str) -> None:
        self.family = family
        self.address = address


class Stats:
    def __init__(self, mtu: int) -> None:
        self.mtu = mtu


MAC_FAMILY = next(iter(_LINK_FAMILIES), -1)


def patch_interfaces(
    monkeypatch: pytest.MonkeyPatch,
    addrs: dict[str, list[Addr]],
    mtus: dict[str, int] | None = None,
) -> None:
    monkeypatch.setattr("psutil.net_if_addrs", lambda: addrs)
    monkeypatch.setattr(
        "psutil.net_if_stats", lambda: {name: Stats(mtu) for name, mtu in (mtus or {}).items()}
    )


def mac(name: str, address: str) -> tuple[str, list[Addr]]:
    return name, [Addr(MAC_FAMILY, address)]


def test_discover_network_interfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interfaces come from psutil addrs/stats; MAC normalizes dashes (spec §58).

    Loopback is a transient/virtual interface, not a static host fact: it is
    filtered out of the capability entirely (§15-16).
    """
    addrs = {
        "eth0": [
            Addr(MAC_FAMILY, "AA-BB-CC-DD-EE-FF"),
            Addr(socket.AF_INET, "192.168.1.100"),
            Addr(socket.AF_INET6, "fe80::1"),
        ],
        "lo": [Addr(socket.AF_INET, "127.0.0.1")],
    }
    patch_interfaces(monkeypatch, addrs, {"eth0": 1500, "lo": 65536})

    fragment = _probe(docker_client_factory=lambda: _raise_docker()).discover()
    interfaces = {nic.name: nic for nic in fragment.network_interfaces}

    assert set(interfaces) == {"eth0"}  # loopback never reaches capability
    eth0 = interfaces["eth0"]
    assert eth0.interface_id == "aa:bb:cc:dd:ee:ff"
    assert eth0.mtu == 1500
    assert "192.168.1.100" in eth0.addresses
    assert "fe80::1" in eth0.addresses


# -- transient container/Pod interfaces are not static capability (§16) -----


@pytest.mark.parametrize(
    "name",
    [
        "lo",
        "lo0",
        "Loopback Pseudo-Interface 1",  # Windows loopback
        "docker0",  # Docker NAT bridge
        "br-9f3k2jx8a1b2",  # Docker bridge network
        "veth7ac1f3b",  # container veth pair (Linux)
        "vEthernet (WSL (Hyper-V firewall))",  # Windows virtual switch
        "cali1234abcd",  # Calico CNI (Kubernetes)
        "tunl0",  # Calico IPIP tunnel
        "cni0",  # CNI bridge (containerd/podman)
        "flannel.1",  # Flannel overlay
        "kube-bridge",
        "podman0",
        "virbr0",  # libvirt NAT bridge
        "vnet3",  # libvirt/KVM tap
    ],
)
def test_transient_container_interfaces_are_filtered(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    addrs = {
        "eth0": [Addr(MAC_FAMILY, "AA-BB-CC-DD-EE-FF")],
        name: [Addr(socket.AF_INET, "10.244.0.1")],
    }
    patch_interfaces(monkeypatch, addrs)

    fragment = _probe(docker_client_factory=lambda: _raise_docker()).discover()

    assert [nic.name for nic in fragment.network_interfaces] == ["eth0"]


@pytest.mark.parametrize(
    "name",
    [
        "eth0",
        "enp3s0",
        "wlan0",
        "Wi-Fi",  # Windows wireless
        "Ethernet",  # Windows wired
        "br0",  # plain LAN bridge: NOT a Docker "br-*" bridge
        "bond0",
        "ztly7mk5dq",  # ZeroTier host overlay
        "tailscale0",  # Tailscale host overlay
        "wg0",  # WireGuard host overlay
        "utun3",  # macOS tunnel (Tailscale et al.)
    ],
)
def test_physical_and_host_overlay_interfaces_are_kept(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    patch_interfaces(monkeypatch, {name: [Addr(MAC_FAMILY, "AA-BB-CC-DD-EE-FF")]})

    fragment = _probe(docker_client_factory=lambda: _raise_docker()).discover()

    assert [nic.name for nic in fragment.network_interfaces] == [name]


def test_capability_revision_stable_under_transient_interface_churn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§16: containers starting/stopping must not churn the capability revision."""
    baseline = dict(
        [
            mac("eth0", "AA-BB-CC-DD-EE-FF"),
            mac("wlan0", "11-22-33-44-55-66"),
            ("tailscale0", [Addr(socket.AF_INET, "100.64.0.1")]),
        ]
    )
    churned = {
        **baseline,
        "docker0": [Addr(MAC_FAMILY, "02:42:9c:1f:2a:0b"), Addr(socket.AF_INET, "172.17.0.1")],
        "br-9f3k2jx8a1b2": [Addr(MAC_FAMILY, "02:42:0d:3e:5f:70")],
        "veth7ac1f3b": [Addr(MAC_FAMILY, "fe:dc:ba:98:76:54")],
        "cali1234abcd": [Addr(MAC_FAMILY, "ee:ee:ee:ee:ee:ee")],
    }

    def revision_for(addrs: dict[str, list[Addr]]) -> str:
        patch_interfaces(monkeypatch, addrs)
        fragment = _probe(docker_client_factory=lambda: _raise_docker()).discover()
        return assemble_capability([fragment]).capability_revision

    before = revision_for(baseline)
    during = revision_for(churned)  # containers up: veth/bridges appear
    after = revision_for(baseline)  # containers down: they vanish again
    assert before == during == after

    # Sanity: the revision still tracks *real* host changes — a new physical
    # NIC with a different MAC is not noise, it must change the fingerprint.
    real_change = dict(baseline)
    real_change["eth1"] = [Addr(MAC_FAMILY, "99-88-77-66-55-44")]
    assert revision_for(real_change) != before


# -- stable, unique interface ids (§11) --------------------------------------


def test_all_zero_mac_is_never_used_as_interface_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two kept interfaces with null MACs must not collide on one id."""
    addrs = {
        "wg0": [Addr(MAC_FAMILY, "00:00:00:00:00:00"), Addr(socket.AF_INET, "10.8.0.1")],
        "tailscale0": [Addr(MAC_FAMILY, "00-00-00-00-00-00")],
    }
    patch_interfaces(monkeypatch, addrs)

    fragment = _probe(docker_client_factory=lambda: _raise_docker()).discover()
    interfaces = {nic.name: nic for nic in fragment.network_interfaces}

    assert set(interfaces) == {"wg0", "tailscale0"}
    assert interfaces["wg0"].interface_id == "if-wg0"
    assert interfaces["tailscale0"].interface_id == "if-tailscale0"
    ids = [nic.interface_id for nic in fragment.network_interfaces]
    assert len(set(ids)) == len(ids)
    assert "00:00:00:00:00:00" not in ids


def test_shared_mac_interfaces_get_unique_name_based_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bonded/aliased NICs legitimately share one MAC: no enumeration-order
    winner, every sharer falls back to its name — in any order."""
    shared = "AA-BB-CC-DD-EE-FF"
    forward = dict([mac("bond0", shared), mac("eth0", shared), mac("eth1", shared)])
    backward = dict(reversed(list(forward.items())))

    ids_by_order = []
    for addrs in (forward, backward):
        patch_interfaces(monkeypatch, addrs)
        fragment = _probe(docker_client_factory=lambda: _raise_docker()).discover()
        ids_by_order.append({nic.name: nic.interface_id for nic in fragment.network_interfaces})

    for mapping in ids_by_order:
        assert mapping == {"bond0": "if-bond0", "eth0": "if-eth0", "eth1": "if-eth1"}
        assert len(set(mapping.values())) == 3  # unique ids, order-independent
    assert ids_by_order[0] == ids_by_order[1]


def test_malformed_mac_falls_back_to_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """E.g. InfiniBand link-layer addresses are 20 bytes: not a MAC id."""
    infiniband = "80:00:02:08:fe:80:00:00:00:00:00:00:f4:52:14:03:00:75:e5:18"
    patch_interfaces(monkeypatch, {"ib0": [Addr(MAC_FAMILY, infiniband)]})

    fragment = _probe(docker_client_factory=lambda: _raise_docker()).discover()

    (nic,) = fragment.network_interfaces
    assert nic.name == "ib0"
    assert nic.interface_id == "if-ib0"


def test_no_mac_reports_name_based_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interfaces exposing no link-layer address at all keep the old fallback."""
    patch_interfaces(monkeypatch, {"tun0": [Addr(socket.AF_INET, "10.0.0.2")]})

    fragment = _probe(docker_client_factory=lambda: _raise_docker()).discover()

    (nic,) = fragment.network_interfaces
    assert nic.interface_id == "if-tun0"


def test_discover_os_uses_freedesktop_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "platform.freedesktop_os_release",
        lambda: {"ID": "ubuntu", "VERSION_ID": "24.04"},
    )
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.release", lambda: "6.8.0")

    fragment = _probe(docker_client_factory=lambda: _raise_docker()).discover()
    assert fragment.os is not None
    assert fragment.os.name == "ubuntu"
    assert fragment.os.version == "24.04"
    assert fragment.os.kernel == "6.8.0"


def test_discover_os_falls_back_without_freedesktop(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_freedesktop() -> Any:
        raise OSError("not available")

    monkeypatch.setattr("platform.freedesktop_os_release", _no_freedesktop)
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("platform.release", lambda: "11")

    fragment = _probe(docker_client_factory=lambda: _raise_docker()).discover()
    assert fragment.os is not None
    assert fragment.os.name == "Windows"
    assert fragment.os.version is None
    assert fragment.os.kernel == "11"
