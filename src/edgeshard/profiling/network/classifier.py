"""Network endpoint characterization and path classification (spec §30-§32).

Phase 2 reuses the Phase 1 discovery facts instead of re-inventing them:
endpoint profiles reference ``NetworkInterfaceCapability`` identities and
add only the profiling-relevant characterization. Anything Phase 1 does
not record (link speed) stays ``None`` (§52.2) — the classifier never
extends facts by guessing.

Interface kinds and path classes are derived from documented, explicit
policies:

* interface kind from the interface *name* using fixed prefix tables
  (wireless ``wl*``/``wifi*``, overlay ``zt*``/``tailscale*``/``wg*``/
  ``utun*``/``tun*``/``tap*``/``vxlan*`` — matching the Phase 1 host
  discovery conventions — wired ``eth*``/``en*``); an unmatched name is
  ``UNKNOWN``, never a silent default;
* real probes require an explicit interface on multi-NIC workers; a sole
  IPv4-capable interface may be selected without ambiguity;
* same-subnet detection compares the first ``same_subnet_prefix_length``
  bits (default /24) — Phase 1 records no netmasks, so the prefix length
  is an explicit configuration knob, not a hidden guess (§31).

Classification precedence (§31): SAME_HOST (identical worker or hostname)
> OVERLAY (probed interface is an overlay on either side) > WIFI_LAN /
WIRED_LAN (same subnet, media known) > CROSS_SUBNET (both IPv4, different
subnets) > OTHER (unknown media or no probeable IPv4). No LLDP/SNMP/SDN.

``NetworkPair`` is directed (§32): A→B and B→A are distinct pairs. Pairs
here are worker-level with ``None`` interface ids (default route); the
bandwidth selection implements the §34 sparse policy — one or few
deterministic representatives per ``PathClass`` plus optional explicit
pairs.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from edgeshard.cluster.snapshot import WorkerSnapshot
from edgeshard.profiling.domain.network import (
    NetworkEndpointProfile,
    NetworkPair,
    NetworkPathClass,
)

DEFAULT_SAME_SUBNET_PREFIX_LENGTH = 24
"""IPv4 prefix length treated as "same subnet" (explicit policy, §31)."""


class InterfaceKind(StrEnum):
    """Coarse medium classification of one interface (spec §30.1)."""

    WIRED = "wired"
    WIRELESS = "wireless"
    OVERLAY = "overlay"
    UNKNOWN = "unknown"


_WIRELESS_PREFIXES = ("wl", "wifi")
_WIRED_PREFIXES = ("eth", "en")
# Overlay names double as the overlay type label; longest prefixes first so
# more specific conventions win (e.g. a future "tunxyz" stays a tunnel).
_OVERLAY_PREFIXES: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (
            ("tailscale", "tailscale"),
            ("vxlan", "vxlan"),
            ("utun", "tunnel"),
            ("tun", "tunnel"),
            ("tap", "tunnel"),
            ("zt", "zerotier"),
            ("wg", "wireguard"),
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def classify_interface(name: str) -> tuple[InterfaceKind, str | None]:
    """Kind (and overlay type) of an interface from its name.

    A fixed, documented name-prefix policy — no LLDP/SNMP/SDN lookups
    (§31). Unmatched names classify as ``UNKNOWN`` so downstream policy can
    refuse to claim a medium it did not observe (§52.2).
    """
    lowered = name.lower()
    for prefix, overlay_type in _OVERLAY_PREFIXES:
        if lowered.startswith(prefix):
            return InterfaceKind.OVERLAY, overlay_type
    if lowered.startswith(_WIRELESS_PREFIXES):
        return InterfaceKind.WIRELESS, None
    if lowered.startswith(_WIRED_PREFIXES):
        return InterfaceKind.WIRED, None
    return InterfaceKind.UNKNOWN, None


@dataclass(frozen=True)
class InterfaceFacts:
    """Phase 1 interface facts plus the derived classification (§30.1)."""

    interface_id: str
    name: str
    kind: InterfaceKind
    overlay_type: str | None
    addresses: tuple[str, ...]
    mtu: int | None


@dataclass(frozen=True)
class WorkerNetworkFacts:
    """Everything network profiling knows about one worker (facts only)."""

    worker_id: str
    hostname: str
    interfaces: tuple[InterfaceFacts, ...]

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id must not be empty")
        if not self.hostname:
            raise ValueError("hostname must not be empty")


def worker_network_facts(worker: WorkerSnapshot) -> WorkerNetworkFacts:
    """Derive profiling facts from a Phase 1 ``WorkerSnapshot`` (§30).

    Reuses the Worker-reported identity (hostname) and interface
    capability; classification is derived from those facts, never from a
    fresh scan.
    """
    interfaces = []
    for capability in worker.capability.network_interfaces:
        kind, overlay_type = classify_interface(capability.name)
        interfaces.append(
            InterfaceFacts(
                interface_id=capability.interface_id,
                name=capability.name,
                kind=kind,
                overlay_type=overlay_type,
                addresses=capability.addresses,
                mtu=capability.mtu,
            )
        )
    return WorkerNetworkFacts(
        worker_id=worker.identity.worker_id,
        hostname=worker.identity.hostname,
        interfaces=tuple(interfaces),
    )


def endpoint_profiles(
    facts: Iterable[WorkerNetworkFacts],
) -> tuple[NetworkEndpointProfile, ...]:
    """One endpoint profile per worker interface (§30.1).

    ``link_speed_mbps`` stays ``None``: Phase 1 capability does not record
    it and Phase 2 never guesses missing facts (§52.2). ``interface_type``
    carries the documented classification, including the explicit
    ``unknown`` label for unmatched names.
    """
    return tuple(
        NetworkEndpointProfile(
            worker_id=worker.worker_id,
            interface_id=interface.interface_id,
            addresses=interface.addresses,
            mtu=interface.mtu,
            link_speed_mbps=None,
            interface_type=interface.kind.value,
            overlay_type=interface.overlay_type,
        )
        for worker in facts
        for interface in worker.interfaces
    )


def _ipv4_addresses(addresses: Sequence[str]) -> tuple[ipaddress.IPv4Address, ...]:
    parsed: list[ipaddress.IPv4Address] = []
    for address in addresses:
        try:
            candidate = ipaddress.ip_address(address)
        except ValueError:
            continue  # non-IP or malformed entries are not probe targets
        if isinstance(candidate, ipaddress.IPv4Address):
            parsed.append(candidate)
    return tuple(parsed)


def primary_ipv4_address(facts: WorkerNetworkFacts) -> str | None:
    """Deterministic probe target: the numerically smallest IPv4 address.

    Phase 1 records no routing table, so this explicit policy approximates
    the default-route address of the worker. ``None`` when the worker has
    no IPv4 address at all — callers fail typed rather than probe a
    guessed target (§52.2).
    """
    candidates = [
        address
        for interface in facts.interfaces
        for address in _ipv4_addresses(interface.addresses)
    ]
    return str(min(candidates)) if candidates else None


def selected_interface(
    facts: WorkerNetworkFacts, interface_id: str | None
) -> InterfaceFacts | None:
    """Resolve an explicit interface, or the sole probeable interface.

    A multi-NIC host without an explicit selection is intentionally
    ambiguous. Real probes must not guess a route from address ordering.
    """
    if interface_id is not None:
        return next(
            (
                interface
                for interface in facts.interfaces
                if interface.interface_id == interface_id
            ),
            None,
        )
    candidates = tuple(
        interface
        for interface in facts.interfaces
        if _ipv4_addresses(interface.addresses)
    )
    return candidates[0] if len(candidates) == 1 else None


def selected_ipv4_address(
    facts: WorkerNetworkFacts, interface_id: str | None
) -> str | None:
    interface = selected_interface(facts, interface_id)
    if interface is None:
        return None
    addresses = _ipv4_addresses(interface.addresses)
    return str(addresses[0]) if addresses else None


def probed_interface(facts: WorkerNetworkFacts) -> InterfaceFacts | None:
    """The interface owning the primary IPv4 address (classification input)."""
    primary = primary_ipv4_address(facts)
    if primary is None:
        return None
    for interface in facts.interfaces:
        if primary in {str(address) for address in _ipv4_addresses(interface.addresses)}:
            return interface
    return None  # unreachable: the primary address came from some interface


def _same_subnet(left: str, right: str, prefix_length: int) -> bool:
    left_bits = int(ipaddress.IPv4Address(left)) >> (32 - prefix_length)
    right_bits = int(ipaddress.IPv4Address(right)) >> (32 - prefix_length)
    return left_bits == right_bits


def classify_path(
    source: WorkerNetworkFacts,
    destination: WorkerNetworkFacts,
    *,
    same_subnet_prefix_length: int = DEFAULT_SAME_SUBNET_PREFIX_LENGTH,
    source_interface_id: str | None = None,
    destination_interface_id: str | None = None,
) -> NetworkPathClass:
    """Cheap path classification from facts only (§31).

    Inputs are interface metadata (name-derived kind), subnet membership
    (configurable prefix policy), overlay metadata, and host identity —
    no LLDP/SNMP/SDN. The classification is over the *probed* interfaces:
    explicit ones when supplied, otherwise the legacy classification-only
    primary address. Real probe execution never uses that legacy guess.
    """
    if not 1 <= same_subnet_prefix_length <= 32:
        raise ValueError(
            f"same_subnet_prefix_length must be within [1, 32], "
            f"got {same_subnet_prefix_length}"
        )
    if source.worker_id == destination.worker_id or source.hostname == destination.hostname:
        return NetworkPathClass.SAME_HOST
    source_interface = (
        selected_interface(source, source_interface_id)
        if source_interface_id is not None
        else probed_interface(source)
    )
    destination_interface = (
        selected_interface(destination, destination_interface_id)
        if destination_interface_id is not None
        else probed_interface(destination)
    )
    if source_interface is None or destination_interface is None:
        return NetworkPathClass.OTHER
    kinds = (source_interface.kind, destination_interface.kind)
    if InterfaceKind.OVERLAY in kinds:
        return NetworkPathClass.OVERLAY
    source_address = (
        selected_ipv4_address(source, source_interface_id)
        if source_interface_id is not None
        else primary_ipv4_address(source)
    )
    destination_address = (
        selected_ipv4_address(destination, destination_interface_id)
        if destination_interface_id is not None
        else primary_ipv4_address(destination)
    )
    assert source_address is not None and destination_address is not None
    if not _same_subnet(source_address, destination_address, same_subnet_prefix_length):
        return NetworkPathClass.CROSS_SUBNET
    if InterfaceKind.UNKNOWN in kinds:
        return NetworkPathClass.OTHER  # same subnet, but the medium is unobserved
    if InterfaceKind.WIRELESS in kinds:
        return NetworkPathClass.WIFI_LAN
    return NetworkPathClass.WIRED_LAN


def enumerate_pairs(facts: Iterable[WorkerNetworkFacts]) -> tuple[NetworkPair, ...]:
    """All directed worker-level pairs (§32), default route (``None`` ids).

    A→B and B→A are distinct; a worker is never paired with itself.
    """
    workers = list(facts)
    worker_ids = [worker.worker_id for worker in workers]
    if len(set(worker_ids)) != len(worker_ids):
        raise ValueError("worker facts contain duplicate worker_id entries")
    return tuple(
        NetworkPair(
            source_worker_id=source.worker_id,
            destination_worker_id=destination.worker_id,
        )
        for source in workers
        for destination in workers
        if source.worker_id != destination.worker_id
    )


def enumerate_probe_paths(
    facts: Iterable[WorkerNetworkFacts],
) -> tuple[NetworkPair, ...]:
    """All directed, explicitly bound IPv4 interface paths.

    Multi-NIC Workers produce one case per source/destination interface
    combination. This may include unreachable combinations, which become
    typed probe failures; it never chooses an interface from address order.
    """
    workers = tuple(facts)
    worker_ids = [worker.worker_id for worker in workers]
    if len(set(worker_ids)) != len(worker_ids):
        raise ValueError("worker facts contain duplicate worker_id entries")
    return tuple(
        NetworkPair(
            source_worker_id=source.worker_id,
            destination_worker_id=destination.worker_id,
            source_interface_id=source_interface.interface_id,
            destination_interface_id=destination_interface.interface_id,
        )
        for source in workers
        for destination in workers
        if source.worker_id != destination.worker_id
        for source_interface in source.interfaces
        if _ipv4_addresses(source_interface.addresses)
        for destination_interface in destination.interfaces
        if _ipv4_addresses(destination_interface.addresses)
    )


@dataclass(frozen=True)
class ClassifiedPair:
    """A directed pair with its derived classification (§31-§32).

    The class is *not* part of pair identity (see ``NetworkPair``); it is
    attached here for the sparse bandwidth policy and metadata only.
    """

    pair: NetworkPair
    path_class: NetworkPathClass


def classify_pairs(
    facts: Iterable[WorkerNetworkFacts],
    *,
    same_subnet_prefix_length: int = DEFAULT_SAME_SUBNET_PREFIX_LENGTH,
) -> tuple[ClassifiedPair, ...]:
    """The dense directed pair list with a ``PathClass`` per pair (§31)."""
    workers = tuple(facts)
    by_id = {worker.worker_id: worker for worker in workers}
    return tuple(
        ClassifiedPair(
            pair=pair,
            path_class=classify_path(
                by_id[pair.source_worker_id],
                by_id[pair.destination_worker_id],
                same_subnet_prefix_length=same_subnet_prefix_length,
            ),
        )
        for pair in enumerate_pairs(workers)
    )


def _pair_order_key(pair: NetworkPair) -> tuple[str, str, str, str]:
    return (
        pair.source_worker_id,
        pair.destination_worker_id,
        pair.source_interface_id or "",
        pair.destination_interface_id or "",
    )


def select_bandwidth_pairs(
    classified: Iterable[ClassifiedPair],
    *,
    representatives_per_class: int = 1,
    extra_pairs: Iterable[NetworkPair] = (),
) -> tuple[NetworkPair, ...]:
    """Sparse bandwidth selection: representatives per class (§34).

    Bandwidth probing is expensive, so the policy measures one (or a few)
    *deterministic* representative directed pairs per ``PathClass`` —
    sorted by (source, destination) so the selection never depends on
    input order — plus optional explicitly requested pairs (the §34
    "explicit pair profiling" knob). Explicit pairs are appended in the
    order given and deduplicated against the representatives.
    """
    if representatives_per_class < 1:
        raise ValueError(
            f"representatives_per_class must be positive, got {representatives_per_class}"
        )
    by_class: dict[NetworkPathClass, list[NetworkPair]] = {}
    for entry in classified:
        by_class.setdefault(entry.path_class, []).append(entry.pair)
    selected: list[NetworkPair] = []
    seen: set[tuple[str, str, str, str]] = set()
    for path_class in sorted(by_class):
        candidates = sorted(by_class[path_class], key=_pair_order_key)
        for pair in candidates[:representatives_per_class]:
            key = _pair_order_key(pair)
            if key not in seen:
                seen.add(key)
                selected.append(pair)
    for pair in extra_pairs:
        key = _pair_order_key(pair)
        if key not in seen:
            seen.add(key)
            selected.append(pair)
    return tuple(selected)


__all__ = [
    "DEFAULT_SAME_SUBNET_PREFIX_LENGTH",
    "ClassifiedPair",
    "InterfaceFacts",
    "InterfaceKind",
    "WorkerNetworkFacts",
    "classify_interface",
    "classify_pairs",
    "classify_path",
    "endpoint_profiles",
    "enumerate_pairs",
    "enumerate_probe_paths",
    "primary_ipv4_address",
    "probed_interface",
    "select_bandwidth_pairs",
    "worker_network_facts",
]
