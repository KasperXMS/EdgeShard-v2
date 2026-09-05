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


def test_discover_network_interfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interfaces come from psutil addrs/stats; MAC normalizes dashes (spec §58)."""

    class Addr:
        def __init__(self, family: int, address: str) -> None:
            self.family = family
            self.address = address

    mac_family = next(iter(_LINK_FAMILIES), -1)
    addrs = {
        "eth0": [
            Addr(mac_family, "AA-BB-CC-DD-EE-FF"),
            Addr(socket.AF_INET, "192.168.1.100"),
            Addr(socket.AF_INET6, "fe80::1"),
        ],
        "lo": [Addr(socket.AF_INET, "127.0.0.1")],
    }

    class Stats:
        def __init__(self, mtu: int) -> None:
            self.mtu = mtu

    monkeypatch.setattr("psutil.net_if_addrs", lambda: addrs)
    monkeypatch.setattr("psutil.net_if_stats", lambda: {"eth0": Stats(1500), "lo": Stats(65536)})

    fragment = _probe(docker_client_factory=lambda: _raise_docker()).discover()
    interfaces = {nic.name: nic for nic in fragment.network_interfaces}

    assert set(interfaces) == {"eth0", "lo"}
    eth0 = interfaces["eth0"]
    assert eth0.interface_id == "aa:bb:cc:dd:ee:ff"
    assert eth0.mtu == 1500
    assert "192.168.1.100" in eth0.addresses
    assert "fe80::1" in eth0.addresses

    lo = interfaces["lo"]
    assert lo.mtu == 65536
    assert lo.addresses == ("127.0.0.1",)
    assert lo.interface_id == "if-lo"  # no MAC exposed: fall back to name


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
