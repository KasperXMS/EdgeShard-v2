"""Network profiling domain (spec §30-32, §35)."""

from __future__ import annotations

import pytest

from edgeshard.profiling.domain.network import (
    NetworkDirection,
    NetworkEndpointProfile,
    NetworkMeasurementRegime,
    NetworkPair,
    NetworkPathClass,
    NetworkTransport,
    ProbeKind,
    network_pair_signature_id,
)


def test_path_class_vocabulary_matches_spec() -> None:
    assert {path_class.value for path_class in NetworkPathClass} == {
        "same_host",
        "wired_lan",
        "wifi_lan",
        "overlay",
        "cross_subnet",
        "other",
    }
    assert {probe.value for probe in ProbeKind} == {"rtt", "bandwidth"}
    assert {t.value for t in NetworkTransport} == {"tcp", "udp"}
    assert {d.value for d in NetworkDirection} == {"forward", "reverse"}
    assert NetworkMeasurementRegime.IDLE_SINGLE_FLOW.value == "idle_single_flow"


def test_endpoint_profile_validation() -> None:
    with pytest.raises(ValueError, match="worker_id"):
        NetworkEndpointProfile(worker_id="", interface_id="iface-1")
    with pytest.raises(ValueError, match="mtu"):
        NetworkEndpointProfile(worker_id="w", interface_id="i", mtu=0)
    with pytest.raises(ValueError, match="link_speed_mbps"):
        NetworkEndpointProfile(worker_id="w", interface_id="i", link_speed_mbps=-1.0)
    with pytest.raises(ValueError, match="empty"):
        NetworkEndpointProfile(worker_id="w", interface_id="i", addresses=("",))


def test_pair_rejects_self_loop() -> None:
    with pytest.raises(ValueError, match="itself"):
        NetworkPair(source_worker_id="w-a", destination_worker_id="w-a")
    # Same worker but different interfaces (two runtimes on one host) is fine.
    assert NetworkPair(
        source_worker_id="w-a",
        destination_worker_id="w-a",
        source_interface_id="iface-1",
        destination_interface_id="iface-2",
    )


def test_pair_signature_is_directional() -> None:
    """§32: A→B and B→A are distinct pairs."""
    forward = NetworkPair(source_worker_id="w-a", destination_worker_id="w-b")
    reverse = NetworkPair(source_worker_id="w-b", destination_worker_id="w-a")
    assert network_pair_signature_id(forward) != network_pair_signature_id(reverse)


def test_pair_signature_is_deterministic_and_interface_sensitive() -> None:
    pair = NetworkPair(
        source_worker_id="w-a",
        destination_worker_id="w-b",
        source_interface_id="iface-1",
    )
    assert network_pair_signature_id(pair) == network_pair_signature_id(
        NetworkPair(
            source_worker_id="w-a",
            destination_worker_id="w-b",
            source_interface_id="iface-1",
        )
    )
    assert network_pair_signature_id(pair) != network_pair_signature_id(
        NetworkPair(
            source_worker_id="w-a",
            destination_worker_id="w-b",
            source_interface_id="iface-2",
        )
    )
