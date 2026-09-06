"""P2F endpoint characterization and path classification tests (spec §30-§32).

Pinned: interface kinds come from the documented name-prefix policy only,
endpoint profiles reuse Phase 1 facts with unknowns left as ``None``
(§52.2), path classification follows the §31 precedence (SAME_HOST >
OVERLAY > WIFI_LAN/WIRED_LAN > CROSS_SUBNET > OTHER), pairs are directed
(§32), and the sparse bandwidth selection (§34) is deterministic
regardless of input order.
"""

from __future__ import annotations

import dataclasses

import pytest

from edgeshard.cluster.capability import NetworkInterfaceCapability
from edgeshard.cluster.snapshot import WorkerSnapshot
from edgeshard.cluster.state import WorkerStatus
from edgeshard.profiling.domain.network import NetworkPair, NetworkPathClass
from edgeshard.profiling.network.classifier import (
    DEFAULT_SAME_SUBNET_PREFIX_LENGTH,
    InterfaceFacts,
    InterfaceKind,
    WorkerNetworkFacts,
    classify_interface,
    classify_pairs,
    classify_path,
    endpoint_profiles,
    enumerate_pairs,
    primary_ipv4_address,
    probed_interface,
    select_bandwidth_pairs,
    worker_network_facts,
)
from factories import make_rtx_capability, make_worker_identity, make_worker_state


def _snapshot(
    worker_id: str,
    hostname: str,
    interfaces: tuple[tuple[str, tuple[str, ...]], ...],
) -> WorkerSnapshot:
    """A WorkerSnapshot whose interfaces are (name, addresses) pairs."""
    capability = dataclasses.replace(
        make_rtx_capability(),
        network_interfaces=tuple(
            NetworkInterfaceCapability(
                interface_id=f"if-{name}", name=name, addresses=addresses, mtu=1500
            )
            for name, addresses in interfaces
        ),
    )
    return WorkerSnapshot(
        identity=dataclasses.replace(
            make_worker_identity(worker_id), hostname=hostname
        ),
        capability=capability,
        state=make_worker_state(worker_id),
        status=WorkerStatus.ONLINE,
        session_id="session-1",
        last_seen_at=None,
    )


def _facts(
    worker_id: str,
    hostname: str | None = None,
    interfaces: tuple[tuple[str, tuple[str, ...]], ...] = (("eth0", ("192.168.1.10",)),),
) -> WorkerNetworkFacts:
    # Default hostnames are worker-specific; tests that want SAME_HOST pass
    # one shared hostname explicitly.
    return worker_network_facts(_snapshot(worker_id, hostname or f"host-{worker_id}", interfaces))


class TestClassifyInterface:
    @pytest.mark.parametrize(
        ("name", "kind", "overlay"),
        [
            ("eth0", InterfaceKind.WIRED, None),
            ("enp3s0", InterfaceKind.WIRED, None),
            ("eno1", InterfaceKind.WIRED, None),
            ("wlan0", InterfaceKind.WIRELESS, None),
            ("wlp3s0", InterfaceKind.WIRELESS, None),
            ("wifi0", InterfaceKind.WIRELESS, None),
            ("zt7sd3rfw7", InterfaceKind.OVERLAY, "zerotier"),
            ("tailscale0", InterfaceKind.OVERLAY, "tailscale"),
            ("wg0", InterfaceKind.OVERLAY, "wireguard"),
            ("utun3", InterfaceKind.OVERLAY, "tunnel"),
            ("tun0", InterfaceKind.OVERLAY, "tunnel"),
            ("tap0", InterfaceKind.OVERLAY, "tunnel"),
            ("vxlan1", InterfaceKind.OVERLAY, "vxlan"),
            ("mystery0", InterfaceKind.UNKNOWN, None),
            ("br-private", InterfaceKind.UNKNOWN, None),
        ],
    )
    def test_name_prefix_policy(self, name: str, kind: InterfaceKind, overlay: str | None) -> None:
        assert classify_interface(name) == (kind, overlay)

    def test_case_insensitive(self) -> None:
        assert classify_interface("WLAN0") == (InterfaceKind.WIRELESS, None)
        assert classify_interface("Eth0") == (InterfaceKind.WIRED, None)

    def test_unmatched_names_never_default_to_a_medium(self) -> None:
        """§52.2: an unobserved medium stays explicit, never a silent guess."""
        kind, overlay = classify_interface("if-not-in-any-table")
        assert kind is InterfaceKind.UNKNOWN
        assert overlay is None


class TestWorkerNetworkFacts:
    def test_derives_from_phase1_snapshot(self) -> None:
        snapshot = _snapshot(
            "w1",
            "host-a",
            (("eth0", ("192.168.1.10", "fe80::1")), ("zt7abc", ("100.64.1.5",))),
        )
        facts = worker_network_facts(snapshot)
        assert facts.worker_id == "w1"
        assert facts.hostname == "host-a"
        assert [interface.name for interface in facts.interfaces] == ["eth0", "zt7abc"]
        eth = facts.interfaces[0]
        assert eth.interface_id == "if-eth0"  # Phase 1 identity reused verbatim
        assert eth.kind is InterfaceKind.WIRED
        assert eth.mtu == 1500
        assert eth.addresses == ("192.168.1.10", "fe80::1")
        zt = facts.interfaces[1]
        assert zt.kind is InterfaceKind.OVERLAY
        assert zt.overlay_type == "zerotier"

    def test_empty_worker_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="worker_id"):
            WorkerNetworkFacts("", "host", ())
        with pytest.raises(ValueError, match="hostname"):
            WorkerNetworkFacts("w1", "", ())


class TestEndpointProfiles:
    def test_one_profile_per_interface_with_phase1_unknowns_none(self) -> None:
        facts = _facts(
            "w1",
            interfaces=(
                ("eth0", ("192.168.1.10",)),
                ("zt7abc", ("100.64.1.5",)),
                ("mystery0", ()),
            ),
        )
        profiles = endpoint_profiles([facts])
        assert len(profiles) == 3
        eth = profiles[0]
        assert (eth.worker_id, eth.interface_id) == ("w1", "if-eth0")
        assert eth.addresses == ("192.168.1.10",)
        assert eth.mtu == 1500
        assert eth.interface_type == "wired"
        assert eth.overlay_type is None
        # Phase 1 capability records no link speed -> None, never guessed (§52.2).
        assert all(profile.link_speed_mbps is None for profile in profiles)
        assert profiles[1].interface_type == "overlay"
        assert profiles[1].overlay_type == "zerotier"
        assert profiles[2].interface_type == "unknown"
        assert profiles[2].overlay_type is None

    def test_no_interfaces_no_profiles(self) -> None:
        assert endpoint_profiles([_facts("w1", interfaces=())]) == ()


class TestPrimaryAddress:
    def test_smallest_numeric_ipv4_wins(self) -> None:
        facts = _facts(
            "w1", interfaces=(("eth0", ("192.168.1.10",)), ("zt7abc", ("100.64.1.5",)))
        )
        assert primary_ipv4_address(facts) == "100.64.1.5"

    def test_numeric_not_lexicographic_ordering(self) -> None:
        facts = _facts("w1", interfaces=(("eth0", ("10.0.0.9", "9.0.0.10")),))
        assert primary_ipv4_address(facts) == "9.0.0.10"

    def test_ipv6_only_worker_has_no_probe_target(self) -> None:
        facts = _facts("w1", interfaces=(("eth0", ("fe80::1",)),))
        assert primary_ipv4_address(facts) is None
        assert probed_interface(facts) is None

    def test_probed_interface_owns_the_primary_address(self) -> None:
        facts = _facts(
            "w1", interfaces=(("eth0", ("192.168.1.10",)), ("wlan0", ("192.168.0.5",)))
        )
        interface = probed_interface(facts)
        assert interface is not None
        assert interface.name == "wlan0"


class TestClassifyPath:
    def test_same_hostname_is_same_host_even_across_worker_ids(self) -> None:
        source = _facts("w1", hostname="box", interfaces=(("eth0", ("192.168.1.10",)),))
        destination = _facts("w2", hostname="box", interfaces=(("eth0", ("10.0.0.5",)),))
        assert classify_path(source, destination) is NetworkPathClass.SAME_HOST

    def test_same_worker_id_is_same_host(self) -> None:
        facts = _facts("w1")
        assert classify_path(facts, facts) is NetworkPathClass.SAME_HOST

    def test_overlay_beats_subnet_reasoning(self) -> None:
        source = _facts("w1", interfaces=(("zt7abc", ("100.64.1.5",)),))
        destination = _facts("w2", interfaces=(("zt7def", ("100.64.1.6",)),))
        assert classify_path(source, destination) is NetworkPathClass.OVERLAY

    def test_either_side_overlay_is_overlay(self) -> None:
        source = _facts("w1", interfaces=(("eth0", ("192.168.1.10",)),))
        destination = _facts("w2", interfaces=(("wg0", ("192.168.1.11",)),))
        assert classify_path(source, destination) is NetworkPathClass.OVERLAY

    def test_same_subnet_wired_both_sides(self) -> None:
        source = _facts("w1", interfaces=(("eth0", ("192.168.1.10",)),))
        destination = _facts("w2", interfaces=(("enp3s0", ("192.168.1.20",)),))
        assert classify_path(source, destination) is NetworkPathClass.WIRED_LAN

    def test_same_subnet_one_wireless_side_is_wifi(self) -> None:
        source = _facts("w1", interfaces=(("wlan0", ("192.168.1.10",)),))
        destination = _facts("w2", interfaces=(("eth0", ("192.168.1.20",)),))
        assert classify_path(source, destination) is NetworkPathClass.WIFI_LAN

    def test_different_subnets_cross_subnet(self) -> None:
        source = _facts("w1", interfaces=(("eth0", ("192.168.1.10",)),))
        destination = _facts("w2", interfaces=(("eth0", ("192.168.2.10",)),))
        assert classify_path(source, destination) is NetworkPathClass.CROSS_SUBNET

    def test_prefix_length_is_a_policy_knob(self) -> None:
        """Phase 1 records no netmasks: the prefix length is explicit config."""
        source = _facts("w1", interfaces=(("eth0", ("10.0.1.5",)),))
        destination = _facts("w2", interfaces=(("eth0", ("10.0.2.5",)),))
        assert classify_path(source, destination) is NetworkPathClass.CROSS_SUBNET
        assert (
            classify_path(source, destination, same_subnet_prefix_length=16)
            is NetworkPathClass.WIRED_LAN
        )

    def test_same_subnet_unknown_medium_is_other(self) -> None:
        """Same subnet but an unobserved medium: OTHER, not a guessed WIRED_LAN."""
        source = _facts("w1", interfaces=(("mystery0", ("192.168.1.10",)),))
        destination = _facts("w2", interfaces=(("eth0", ("192.168.1.20",)),))
        assert classify_path(source, destination) is NetworkPathClass.OTHER

    def test_missing_ipv4_is_other(self) -> None:
        source = _facts("w1", interfaces=(("eth0", ("fe80::1",)),))
        destination = _facts("w2", interfaces=(("eth0", ("192.168.1.20",)),))
        assert classify_path(source, destination) is NetworkPathClass.OTHER
        assert classify_path(destination, source) is NetworkPathClass.OTHER

    def test_same_host_beats_overlay(self) -> None:
        source = _facts("w1", hostname="box", interfaces=(("zt7abc", ("100.64.1.5",)),))
        destination = _facts("w2", hostname="box", interfaces=(("zt7def", ("100.64.1.6",)),))
        assert classify_path(source, destination) is NetworkPathClass.SAME_HOST

    def test_invalid_prefix_length_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[1, 32\]"):
            classify_path(_facts("w1"), _facts("w2"), same_subnet_prefix_length=0)
        with pytest.raises(ValueError, match=r"\[1, 32\]"):
            classify_path(_facts("w1"), _facts("w2"), same_subnet_prefix_length=33)

    def test_default_prefix_length_is_24(self) -> None:
        assert DEFAULT_SAME_SUBNET_PREFIX_LENGTH == 24


class TestPairEnumeration:
    def test_directed_pairs_both_directions_no_self(self) -> None:
        pairs = enumerate_pairs([_facts("w1"), _facts("w2"), _facts("w3")])
        assert len(pairs) == 6
        directed = {(pair.source_worker_id, pair.destination_worker_id) for pair in pairs}
        assert ("w1", "w2") in directed and ("w2", "w1") in directed
        assert all(pair.source_worker_id != pair.destination_worker_id for pair in pairs)
        # Worker-level pairs probe the default route: interface ids stay None (§32).
        assert all(
            pair.source_interface_id is None and pair.destination_interface_id is None
            for pair in pairs
        )

    def test_single_worker_no_pairs(self) -> None:
        assert enumerate_pairs([_facts("w1")]) == ()

    def test_duplicate_worker_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            enumerate_pairs([_facts("w1"), _facts("w1")])

    def test_classify_pairs_attaches_path_class(self) -> None:
        workers = [
            _facts("w1", hostname="host-a", interfaces=(("eth0", ("192.168.1.10",)),)),
            _facts("w2", hostname="host-b", interfaces=(("eth0", ("192.168.1.20",)),)),
            _facts("w3", hostname="host-c", interfaces=(("zt7abc", ("100.64.1.9",)),)),
        ]
        classified = classify_pairs(workers)
        assert len(classified) == 6
        classes = {
            (entry.pair.source_worker_id, entry.pair.destination_worker_id): entry.path_class
            for entry in classified
        }
        assert classes[("w1", "w2")] is NetworkPathClass.WIRED_LAN
        assert classes[("w1", "w3")] is NetworkPathClass.OVERLAY
        assert classes[("w3", "w1")] is NetworkPathClass.OVERLAY  # directed pairs, symmetric class


class TestSelectBandwidthPairs:
    def _workers(self) -> tuple[WorkerNetworkFacts, ...]:
        return (
            _facts("w1", hostname="a", interfaces=(("eth0", ("192.168.1.10",)),)),
            _facts("w2", hostname="b", interfaces=(("eth0", ("192.168.1.20",)),)),
            _facts("w3", hostname="c", interfaces=(("wlan0", ("192.168.1.30",)),)),
            _facts("w4", hostname="d", interfaces=(("eth0", ("10.0.0.5",)),)),
            _facts("w5", hostname="e", interfaces=(("zt7abc", ("100.64.1.5",)),)),
        )

    def test_one_deterministic_representative_per_class(self) -> None:
        workers = self._workers()
        classified = classify_pairs(workers)
        selected = select_bandwidth_pairs(classified)
        classes = {entry.pair: entry.path_class for entry in classified}
        selected_classes = [classes[pair] for pair in selected]
        assert len(selected) == len(set(selected_classes))  # exactly one per class
        assert set(selected_classes) == {
            NetworkPathClass.WIRED_LAN,
            NetworkPathClass.WIFI_LAN,
            NetworkPathClass.CROSS_SUBNET,
            NetworkPathClass.OVERLAY,
        }

    def test_selection_independent_of_input_order(self) -> None:
        workers = self._workers()
        forward = select_bandwidth_pairs(classify_pairs(workers))
        reversed_order = select_bandwidth_pairs(classify_pairs(workers[::-1]))
        assert forward == reversed_order

    def test_representatives_per_class_knob(self) -> None:
        workers = self._workers()
        classified = classify_pairs(workers)
        selected = select_bandwidth_pairs(classified, representatives_per_class=2)
        classes = [
            {entry.pair: entry.path_class for entry in classified}[pair] for pair in selected
        ]
        # Each class has at least 2 directed pairs here, so 2 per class.
        assert len(selected) == 8
        assert classes.count(NetworkPathClass.WIRED_LAN) == 2

    def test_extra_pairs_appended_and_deduplicated(self) -> None:
        """§34 explicit pair profiling knob: requested pairs always included."""
        workers = self._workers()
        classified = classify_pairs(workers)
        explicit = NetworkPair(source_worker_id="w4", destination_worker_id="w5")
        baseline = select_bandwidth_pairs(classified)
        selected = select_bandwidth_pairs(classified, extra_pairs=(explicit, explicit))
        assert selected.count(explicit) == 1  # duplicates collapsed
        assert len(selected) == len(baseline) + (0 if explicit in baseline else 1)
        assert explicit in selected

    def test_invalid_representatives_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            select_bandwidth_pairs([], representatives_per_class=0)

    def test_empty_classification_empty_selection(self) -> None:
        assert select_bandwidth_pairs([]) == ()

    def test_selected_pairs_are_subset_of_classified(self) -> None:
        workers = self._workers()
        classified = classify_pairs(workers)
        selected = select_bandwidth_pairs(classified)
        all_pairs = {entry.pair for entry in classified}
        assert all(pair in all_pairs for pair in selected)


class TestInterfaceFactsValidation:
    def test_facts_are_plain_values(self) -> None:
        facts = InterfaceFacts("if-eth0", "eth0", InterfaceKind.WIRED, None, ("10.0.0.1",), 9000)
        assert facts.mtu == 9000
        assert facts.overlay_type is None
