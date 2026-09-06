"""Network profiling domain (Phase 2 spec §30-32, §35-36).

The three-level hierarchy Endpoint → PathClass → Pair (§1.5):

* ``NetworkEndpointProfile`` references Phase 1 network facts (worker,
  interface, addresses, MTU) and adds profiling-relevant nominal values —
  Phase 2 measures empirically, it never replaces Phase 1 discovery (§30);
* ``NetworkPathClass`` is the cheap classification that makes bandwidth
  profiling sparse: one or few representative directed pairs per class
  instead of a full pairwise matrix (§31, §34);
* ``NetworkPair`` is directed — A→B and B→A are distinct (§32).

Phase 2 v1 measures idle single-flow baselines only. The regime is part of
the domain so Phase 3 can never mistake baseline throughput for guaranteed
concurrent bandwidth (§35, §52.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from edgeshard.profiling.domain.hashing import canonical_sha256


class NetworkPathClass(StrEnum):
    """Coarse classification of a network path (spec §31).

    Classified from interface metadata, subnet information, overlay
    metadata, and host identity — no LLDP/SNMP/SDN dependencies in v1.
    """

    SAME_HOST = "same_host"
    WIRED_LAN = "wired_lan"
    WIFI_LAN = "wifi_lan"
    OVERLAY = "overlay"
    CROSS_SUBNET = "cross_subnet"
    OTHER = "other"


class ProbeKind(StrEnum):
    """Which empirical network probe a case requests (spec §33-34)."""

    RTT = "rtt"
    BANDWIDTH = "bandwidth"


class NetworkTransport(StrEnum):
    """Transport of a bandwidth probe (iperf3 ``--tcp``/``--udp``)."""

    TCP = "tcp"
    UDP = "udp"


class NetworkDirection(StrEnum):
    """Flow direction relative to the pair (iperf3 ``-R``)."""

    FORWARD = "forward"
    REVERSE = "reverse"


class NetworkMeasurementRegime(StrEnum):
    """Load regime a network measurement was taken under (spec §35).

    v1 only ever measures idle single-flow baselines; the regime is
    recorded explicitly so later phases cannot misread the numbers.
    """

    IDLE_SINGLE_FLOW = "idle_single_flow"


@dataclass(frozen=True)
class NetworkEndpointProfile:
    """Profiling view of one Worker network endpoint (spec §30.1).

    Identity fields (``worker_id``/``interface_id``) reference the Phase 1
    ``NetworkInterfaceCapability``; the remaining fields are the nominal
    endpoint characterization Phase 2 adds on top. Anything unknown is
    ``None`` (§52.2).
    """

    worker_id: str
    interface_id: str
    addresses: tuple[str, ...] = ()
    mtu: int | None = None
    link_speed_mbps: float | None = None
    interface_type: str | None = None
    overlay_type: str | None = None

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id must not be empty")
        if not self.interface_id:
            raise ValueError("interface_id must not be empty")
        for address in self.addresses:
            if not address:
                raise ValueError("addresses must not contain empty entries")
        if self.mtu is not None and self.mtu <= 0:
            raise ValueError(f"mtu must be positive, got {self.mtu}")
        if self.link_speed_mbps is not None and self.link_speed_mbps <= 0:
            raise ValueError(
                f"link_speed_mbps must be positive, got {self.link_speed_mbps}"
            )
        if self.interface_type is not None and not self.interface_type:
            raise ValueError("interface_type must not be empty when present")
        if self.overlay_type is not None and not self.overlay_type:
            raise ValueError("overlay_type must not be empty when present")


@dataclass(frozen=True)
class NetworkPair:
    """Directed source → destination measurement pair (spec §32).

    The pair identity is the endpoint tuple only; ``PathClass`` is a
    derived classification that may change (reclassification must not
    change which pair a measurement belongs to), so it never appears here.
    Interface ids may be ``None`` when a probe uses the default route.
    """

    source_worker_id: str
    destination_worker_id: str
    source_interface_id: str | None = None
    destination_interface_id: str | None = None

    def __post_init__(self) -> None:
        if not self.source_worker_id:
            raise ValueError("source_worker_id must not be empty")
        if not self.destination_worker_id:
            raise ValueError("destination_worker_id must not be empty")
        if self.source_interface_id is not None and not self.source_interface_id:
            raise ValueError("source_interface_id must not be empty when present")
        if (
            self.destination_interface_id is not None
            and not self.destination_interface_id
        ):
            raise ValueError("destination_interface_id must not be empty when present")
        if (
            self.source_worker_id == self.destination_worker_id
            and self.source_interface_id == self.destination_interface_id
        ):
            raise ValueError(
                "a pair must not connect an endpoint to itself "
                f"(worker {self.source_worker_id!r}, "
                f"interface {self.source_interface_id!r})"
            )


def network_pair_signature_id(pair: NetworkPair) -> str:
    """Canonical SHA-256 identity of a directed pair (spec §7, §32).

    Directional by construction: the A→B and B→A digests differ.
    """
    return canonical_sha256(("network_pair_signature", pair))
