"""P2F NetworkProfiler and case-builder tests (spec §33-§36, P2F DoD).

Probes run against injected fake runners — the tests pin record assembly
(RttMetrics/BandwidthMetrics conventions), the §35 regime metadata, the
dense RTT matrix (§33), the sparse per-class bandwidth selection with
both directions (§34), and the §36 payload-size derivation from model
characterization dimensions.
"""

from __future__ import annotations

import asyncio

import pytest

from edgeshard.profiling.domain.experiment import (
    ModelCaseSpec,
    NetworkCaseSpec,
    ProfilingCase,
    ProfilingErrorCategory,
)
from edgeshard.profiling.domain.measurement import (
    MeasurementRecord,
    TimeUnit,
    summarize_samples,
)
from edgeshard.profiling.domain.network import (
    NetworkDirection,
    NetworkMeasurementRegime,
    NetworkPair,
    NetworkPathClass,
    NetworkTransport,
    ProbeKind,
)
from edgeshard.profiling.domain.signature import (
    GemmSignature,
    OperatorKind,
    OperatorSignature,
    ProfilingGranularity,
)
from edgeshard.profiling.dtypes import dtype_byte_size
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.network.classifier import (
    InterfaceFacts,
    WorkerNetworkFacts,
    classify_interface,
)
from edgeshard.profiling.network.iperf import Iperf3Observation, Iperf3Probe
from edgeshard.profiling.network.ping import PingObservation
from edgeshard.profiling.network.profiler import (
    DEFAULT_PAYLOAD_SEQUENCE_LENGTHS,
    NetworkProfiler,
    bandwidth_cases,
    hidden_state_payload_bytes,
    pipeline_payload_sizes,
    require_network_spec,
    rtt_matrix_cases,
)


def _worker(worker_id: str, *addresses: str, name: str = "eth0") -> WorkerNetworkFacts:
    kind, overlay = classify_interface(name)
    return WorkerNetworkFacts(
        worker_id=worker_id,
        hostname=f"host-{worker_id}",
        interfaces=(
            InterfaceFacts(f"if-{name}", name, kind, overlay, addresses, 1500),
        ),
    )


def _facts_map(*workers: WorkerNetworkFacts) -> dict[str, WorkerNetworkFacts]:
    return {worker.worker_id: worker for worker in workers}


def _model_case_spec() -> ModelCaseSpec:
    """A model-side spec used to prove the network path rejects it."""
    return ModelCaseSpec(
        granularity=ProfilingGranularity.OPERATOR,
        device_ids=("cpu-0",),
        dtype="fp32",
        operator_signature=OperatorSignature(
            kind=OperatorKind.GEMM,
            parameters=GemmSignature(m=8, n=8, k=8, dtype="fp32"),
            backend_family="torch",
        ),
    )


def _rtt_case(
    source: str = "w1",
    destination: str = "w2",
    *,
    packet_count: int | None = None,
    path_class: NetworkPathClass | None = NetworkPathClass.WIRED_LAN,
) -> ProfilingCase:
    return ProfilingCase.for_spec(
        source,
        NetworkCaseSpec(
            probe_kind=ProbeKind.RTT,
            source_worker_id=source,
            destination_worker_id=destination,
            path_class=path_class,
            packet_count=packet_count,
        ),
    )


def _bandwidth_case(
    source: str = "w1",
    destination: str = "w2",
    *,
    direction: NetworkDirection = NetworkDirection.FORWARD,
    payload_bytes: int | None = None,
) -> ProfilingCase:
    return ProfilingCase.for_spec(
        source,
        NetworkCaseSpec(
            probe_kind=ProbeKind.BANDWIDTH,
            source_worker_id=source,
            destination_worker_id=destination,
            path_class=NetworkPathClass.WIRED_LAN,
            transport=NetworkTransport.TCP,
            direction=direction,
            duration_s=2.0,
            payload_bytes=payload_bytes,
        ),
    )


class FakePingRunner:
    """Injected PingRunner stand-in returning a fixed observation."""

    def __init__(self, samples: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8)) -> None:
        self.samples = samples
        self.calls: list[tuple[str, int | None]] = []
        self.active = 0
        self.max_active = 0
        self.delay = 0.01

    async def probe(self, target: str, *, packet_count: int | None = None) -> PingObservation:
        self.calls.append((target, packet_count))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        sent = packet_count if packet_count is not None else 10
        return PingObservation(
            target=target,
            packets_sent=sent,
            packets_received=len(self.samples),
            samples_ms=self.samples,
            summary=summarize_samples(self.samples),
        )


class FakeIperfRunner:
    """Injected Iperf3Runner stand-in returning a fixed observation."""

    def __init__(self, bits_per_second: float = 9.3e9, retransmits: int | None = 7) -> None:
        self.observation = Iperf3Observation(
            bits_per_second=bits_per_second, retransmits=retransmits
        )
        self.probes: list[Iperf3Probe] = []
        self.active = 0
        self.max_active = 0
        self.delay = 0.01

    async def run_client(self, probe: Iperf3Probe) -> Iperf3Observation:
        self.probes.append(probe)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        return self.observation


class TestDtypeByteSize:
    @pytest.mark.parametrize(
        ("label", "size"),
        [
            ("fp64", 8),
            ("fp32", 4),
            ("fp16", 2),
            ("bf16", 2),
            ("fp8_e4m3", 1),
            ("fp8_e5m2", 1),
            ("int8", 1),
            ("uint8", 1),
            ("bool", 1),
        ],
    )
    def test_known_labels(self, label: str, size: int) -> None:
        assert dtype_byte_size(label) == size

    def test_unknown_label_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown dtype label"):
            dtype_byte_size("unknown")


class TestPayloadSizes:
    def test_hidden_state_formula(self) -> None:
        """§36: batch * seq * hidden * bytes_per_element."""
        assert (
            hidden_state_payload_bytes(
                batch_size=1, sequence_length=512, hidden_size=4096, dtype="bf16"
            )
            == 512 * 4096 * 2
        )
        assert (
            hidden_state_payload_bytes(
                batch_size=4, sequence_length=16, hidden_size=64, dtype="fp32"
            )
            == 4 * 16 * 64 * 4
        )

    def test_invalid_dimensions_rejected(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            hidden_state_payload_bytes(batch_size=0, sequence_length=1, hidden_size=1, dtype="bf16")
        with pytest.raises(ValueError, match="sequence_length"):
            hidden_state_payload_bytes(batch_size=1, sequence_length=0, hidden_size=1, dtype="bf16")
        with pytest.raises(ValueError, match="hidden_size"):
            hidden_state_payload_bytes(batch_size=1, sequence_length=1, hidden_size=0, dtype="bf16")
        with pytest.raises(ValueError, match="unknown dtype"):
            hidden_state_payload_bytes(
                batch_size=1, sequence_length=1, hidden_size=1, dtype="unknown"
            )

    def test_default_sizes_are_small_sorted_deduped_set(self) -> None:
        sizes = pipeline_payload_sizes(hidden_size=4096, dtype="bf16")
        assert sizes == (8192, 4_194_304, 16_777_216)  # seq 1 / 512 / 2048
        assert DEFAULT_PAYLOAD_SEQUENCE_LENGTHS == (1, 512, 2048)

    def test_duplicate_lengths_dedup(self) -> None:
        sizes = pipeline_payload_sizes(
            hidden_size=64, dtype="fp32", sequence_lengths=(8, 8, 16)
        )
        assert sizes == (8 * 64 * 4, 16 * 64 * 4)

    def test_batch_size_knob(self) -> None:
        assert pipeline_payload_sizes(
            hidden_size=64, dtype="fp32", batch_size=2, sequence_lengths=(8,)
        ) == (2 * 8 * 64 * 4,)

    def test_empty_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            pipeline_payload_sizes(hidden_size=64, dtype="fp32", sequence_lengths=())


class TestRequireNetworkSpec:
    def test_network_spec_passes_through(self) -> None:
        case = _rtt_case()
        assert require_network_spec(case) is case.spec

    def test_model_case_rejected(self) -> None:
        with pytest.raises(ValueError, match="network case spec"):
            require_network_spec(ProfilingCase.for_spec("w1", _model_case_spec()))


class TestCaseBuilders:
    def test_rtt_matrix_is_dense_and_directed(self) -> None:
        """§33: a full pairwise RTT matrix, one case per directed pair."""
        workers = [
            _worker("w1", "192.168.1.10"),
            _worker("w2", "192.168.1.20"),
            _worker("w3", "10.0.0.5"),
        ]
        cases = rtt_matrix_cases(workers)
        assert len(cases) == 6
        assert all(isinstance(case.spec, NetworkCaseSpec) for case in cases)
        # Network cases execute on the source worker (§8.2).
        assert all(case.worker_id == case.spec.source_worker_id for case in cases)
        directed = {
            (case.spec.source_worker_id, case.spec.destination_worker_id) for case in cases
        }
        assert ("w1", "w2") in directed and ("w2", "w1") in directed
        assert all(case.spec.probe_kind is ProbeKind.RTT for case in cases)

    def test_rtt_matrix_attaches_derived_path_class(self) -> None:
        workers = [
            _worker("w1", "192.168.1.10"),
            _worker("w2", "192.168.1.20"),
            _worker("w3", "10.0.0.5"),
        ]
        cases = rtt_matrix_cases(workers)
        classes = {
            (case.spec.source_worker_id, case.spec.destination_worker_id): case.spec.path_class
            for case in cases
        }
        assert classes[("w1", "w2")] is NetworkPathClass.WIRED_LAN
        assert classes[("w1", "w3")] is NetworkPathClass.CROSS_SUBNET

    def test_rtt_matrix_packet_count_knob(self) -> None:
        cases = rtt_matrix_cases(
            [_worker("w1", "10.0.0.1"), _worker("w2", "10.0.0.2")], packet_count=7
        )
        assert all(case.spec.packet_count == 7 for case in cases)

    def test_rtt_matrix_case_ids_are_canonical(self) -> None:
        workers = [_worker("w1", "10.0.0.1"), _worker("w2", "10.0.0.2")]
        first = rtt_matrix_cases(workers)
        second = rtt_matrix_cases(workers)
        assert [case.case_id for case in first] == [case.case_id for case in second]

    def test_bandwidth_cases_are_sparse_with_both_directions(self) -> None:
        """§34: representatives per class; forward and reverse are distinct cases."""
        workers = [
            _worker("w1", "192.168.1.10"),
            _worker("w2", "192.168.1.20"),
            _worker("w3", "192.168.1.30", name="wlan0"),
            _worker("w4", "10.0.0.5"),
            _worker("w5", "100.64.1.5", name="zt7abc"),
        ]
        cases = bandwidth_cases(workers)
        classes = {case.spec.path_class for case in cases}
        assert classes == {
            NetworkPathClass.WIRED_LAN,
            NetworkPathClass.WIFI_LAN,
            NetworkPathClass.CROSS_SUBNET,
            NetworkPathClass.OVERLAY,
        }
        assert len(cases) == 8  # one representative pair per class, both directions
        for path_class in classes:
            directions = [
                case.spec.direction for case in cases if case.spec.path_class is path_class
            ]
            assert NetworkDirection.FORWARD in directions
            assert NetworkDirection.REVERSE in directions
        assert all(case.spec.probe_kind is ProbeKind.BANDWIDTH for case in cases)
        assert all(case.spec.regime is NetworkMeasurementRegime.IDLE_SINGLE_FLOW for case in cases)

    def test_bandwidth_forward_only(self) -> None:
        workers = [_worker("w1", "10.0.0.1"), _worker("w2", "10.0.0.2")]
        cases = bandwidth_cases(workers, include_reverse=False)
        assert len(cases) == 1
        assert cases[0].spec.direction is NetworkDirection.FORWARD

    def test_bandwidth_knobs_flow_into_specs(self) -> None:
        workers = [_worker("w1", "10.0.0.1"), _worker("w2", "10.0.0.2")]
        cases = bandwidth_cases(
            workers,
            transport=NetworkTransport.UDP,
            duration_s=3.0,
            payload_bytes=4_194_304,
        )
        for case in cases:
            assert case.spec.transport is NetworkTransport.UDP
            assert case.spec.duration_s == 3.0
            assert case.spec.payload_bytes == 4_194_304

    def test_bandwidth_explicit_extra_pairs(self) -> None:
        """§34 optional explicit pair profiling knob."""
        workers = [
            _worker("w1", "192.168.1.10"),
            _worker("w2", "192.168.1.20"),
            _worker("w3", "192.168.1.30"),
        ]
        explicit = NetworkPair(source_worker_id="w2", destination_worker_id="w3")
        baseline = bandwidth_cases(workers, include_reverse=False)
        cases = bandwidth_cases(workers, extra_pairs=(explicit,), include_reverse=False)
        assert len(cases) == len(baseline) + 1
        pairs = {
            NetworkPair(
                source_worker_id=case.spec.source_worker_id,
                destination_worker_id=case.spec.destination_worker_id,
            )
            for case in cases
        }
        assert explicit in pairs

    def test_bandwidth_path_class_filter(self) -> None:
        """§49 ``--path-class`` knob: restricts the class-driven selection."""
        workers = [
            _worker("w1", "192.168.1.10"),
            _worker("w2", "192.168.1.20"),
            _worker("w3", "192.168.1.30", name="wlan0"),
        ]
        cases = bandwidth_cases(workers, path_classes=[NetworkPathClass.WIFI_LAN])
        assert cases
        assert {case.spec.path_class for case in cases} == {NetworkPathClass.WIFI_LAN}

    def test_bandwidth_path_class_filter_keeps_explicit_pairs(self) -> None:
        """The explicit-pair knob wins over the class filter (§34)."""
        workers = [
            _worker("w1", "192.168.1.10"),
            _worker("w2", "192.168.1.20"),
        ]
        explicit = NetworkPair(source_worker_id="w2", destination_worker_id="w1")
        cases = bandwidth_cases(
            workers,
            extra_pairs=(explicit,),
            path_classes=[NetworkPathClass.OVERLAY],
            include_reverse=False,
        )
        assert len(cases) == 1
        directed = (cases[0].spec.source_worker_id, cases[0].spec.destination_worker_id)
        assert directed == ("w2", "w1")

    def test_bandwidth_unknown_worker_in_extra_pairs_rejected(self) -> None:
        workers = [_worker("w1", "10.0.0.1"), _worker("w2", "10.0.0.2")]
        with pytest.raises(ValueError, match="unknown worker"):
            bandwidth_cases(
                workers,
                extra_pairs=(NetworkPair(source_worker_id="w1", destination_worker_id="ghost"),),
            )

    def test_single_worker_cluster_has_no_cases(self) -> None:
        assert rtt_matrix_cases([_worker("w1", "10.0.0.1")]) == ()
        assert bandwidth_cases([_worker("w1", "10.0.0.1")]) == ()


class TestNetworkProfilerRtt:
    async def test_rtt_record_end_to_end(self) -> None:
        ping = FakePingRunner(samples=(0.4, 0.5, 0.6, 0.7, 0.8))
        profiler = NetworkProfiler(ping_runner=ping)
        facts = _facts_map(_worker("w1", "192.168.1.10"), _worker("w2", "192.168.1.20"))
        case = _rtt_case(packet_count=5)
        record = await profiler.profile(
            case, network_facts=facts, environment_fingerprint="env-test"
        )
        assert ping.calls == [("192.168.1.20", 5)]
        assert record.case_id == case.case_id
        assert record.environment_fingerprint == "env-test"
        assert record.measurement_id
        assert record.sample_count == 5
        assert record.samples == (0.4, 0.5, 0.6, 0.7, 0.8)
        rtt = record.metrics.rtt
        assert rtt is not None
        assert rtt.unit is TimeUnit.MILLISECONDS
        assert rtt.packets_sent == 5
        assert rtt.packets_received == 5
        assert rtt.summary.median == 0.6
        assert rtt.summary.p95 == 0.8
        # Loss is derived from sent/received; jitter is the stddev (§33).
        assert rtt.summary.stddev > 0.0
        assert record.metrics.bandwidth is None
        assert record.finished_at >= record.started_at
        metadata = record.metadata_mapping
        assert metadata["probe_kind"] == "rtt"
        assert metadata["source_worker_id"] == "w1"
        assert metadata["destination_worker_id"] == "w2"
        assert metadata["target"] == "192.168.1.20"
        assert metadata["regime"] == "idle_single_flow"
        assert metadata["path_class"] == "wired_lan"
        assert metadata["timing_unit"] == "ms"
        assert metadata["packet_count"] == 5

    async def test_partial_loss_is_recorded_not_failed(self) -> None:
        ping = FakePingRunner(samples=(0.5, 0.6))
        profiler = NetworkProfiler(ping_runner=ping)
        facts = _facts_map(_worker("w1", "192.168.1.10"), _worker("w2", "192.168.1.20"))
        record = await profiler.profile(
            _rtt_case(packet_count=5), network_facts=facts, environment_fingerprint="e"
        )
        rtt = record.metrics.rtt
        assert rtt is not None
        assert rtt.packets_sent == 5
        assert rtt.packets_received == 2  # loss survives as a derived fact
        assert record.sample_count == 2

    async def test_default_packet_count_from_runner(self) -> None:
        ping = FakePingRunner()
        profiler = NetworkProfiler(ping_runner=ping)
        facts = _facts_map(_worker("w1", "192.168.1.10"), _worker("w2", "192.168.1.20"))
        await profiler.profile(
            _rtt_case(packet_count=None), network_facts=facts, environment_fingerprint="e"
        )
        assert ping.calls == [("192.168.1.20", None)]  # runner policy decides

    async def test_missing_path_class_records_none(self) -> None:
        """Unknown classification stays None — never a guessed label (§52.2)."""
        profiler = NetworkProfiler(ping_runner=FakePingRunner())
        facts = _facts_map(_worker("w1", "192.168.1.10"), _worker("w2", "192.168.1.20"))
        record = await profiler.profile(
            _rtt_case(path_class=None), network_facts=facts, environment_fingerprint="e"
        )
        assert record.metadata_mapping["path_class"] is None


class TestNetworkProfilerBandwidth:
    async def test_bandwidth_record_end_to_end(self) -> None:
        iperf = FakeIperfRunner(bits_per_second=9.3e9, retransmits=7)
        profiler = NetworkProfiler(iperf_runner=iperf)
        facts = _facts_map(_worker("w1", "192.168.1.10"), _worker("w2", "192.168.1.20"))
        case = _bandwidth_case(payload_bytes=4_194_304)
        record = await profiler.profile(
            case, network_facts=facts, environment_fingerprint="env-test"
        )
        probe = iperf.probes[0]
        assert probe.target == "192.168.1.20"
        assert probe.transport is NetworkTransport.TCP
        assert probe.direction is NetworkDirection.FORWARD
        assert probe.duration_s == 2.0
        assert probe.payload_bytes == 4_194_304
        bandwidth = record.metrics.bandwidth
        assert bandwidth is not None
        assert bandwidth.bits_per_second == pytest.approx(9.3e9)
        assert bandwidth.payload_bytes == 4_194_304
        assert bandwidth.retransmits == 7
        assert bandwidth.duration_s == 2.0
        assert record.metrics.rtt is None
        assert record.sample_count == 1
        assert record.samples == (9.3e9,)
        metadata = record.metadata_mapping
        assert metadata["probe_kind"] == "bandwidth"
        assert metadata["transport"] == "tcp"
        assert metadata["direction"] == "forward"
        assert metadata["duration_s"] == 2.0
        assert metadata["sample_unit"] == "bits_per_second"
        assert metadata["payload_bytes"] == 4_194_304
        # §35: the regime is recorded so Phase 3 never mistakes the baseline
        # for guaranteed concurrent bandwidth.
        assert metadata["regime"] == "idle_single_flow"

    async def test_reverse_direction_flows_into_probe(self) -> None:
        iperf = FakeIperfRunner()
        profiler = NetworkProfiler(iperf_runner=iperf)
        facts = _facts_map(_worker("w1", "192.168.1.10"), _worker("w2", "192.168.1.20"))
        record = await profiler.profile(
            _bandwidth_case(direction=NetworkDirection.REVERSE),
            network_facts=facts,
            environment_fingerprint="e",
        )
        assert iperf.probes[0].direction is NetworkDirection.REVERSE
        assert record.metadata_mapping["direction"] == "reverse"

    async def test_missing_payload_bytes_stays_absent(self) -> None:
        iperf = FakeIperfRunner(retransmits=None)
        profiler = NetworkProfiler(iperf_runner=iperf)
        facts = _facts_map(_worker("w1", "192.168.1.10"), _worker("w2", "192.168.1.20"))
        record = await profiler.profile(
            _bandwidth_case(), network_facts=facts, environment_fingerprint="e"
        )
        bandwidth = record.metrics.bandwidth
        assert bandwidth is not None
        assert bandwidth.payload_bytes is None
        assert bandwidth.retransmits is None  # UDP-style absence stays None
        assert "payload_bytes" not in record.metadata_mapping


class TestNetworkProfilerFactResolution:
    async def test_model_case_rejected(self) -> None:
        with pytest.raises(ValueError, match="network case spec"):
            await NetworkProfiler().profile(
                ProfilingCase.for_spec("w1", _model_case_spec()),
                network_facts={},
                environment_fingerprint="e",
            )

    async def test_unknown_destination_worker_rejected(self) -> None:
        with pytest.raises(ValueError, match="no network facts"):
            await NetworkProfiler(ping_runner=FakePingRunner()).profile(
                _rtt_case(destination="ghost"),
                network_facts=_facts_map(_worker("w1", "192.168.1.10")),
                environment_fingerprint="e",
            )

    async def test_destination_without_ipv4_fails_typed(self) -> None:
        """No probe target → typed failure, never a guessed address (§52.2)."""
        facts = _facts_map(_worker("w1", "192.168.1.10"), _worker("w2", "fe80::2"))
        with pytest.raises(ProfilingError) as excinfo:
            await NetworkProfiler(ping_runner=FakePingRunner()).profile(
                _rtt_case(), network_facts=facts, environment_fingerprint="e"
            )
        assert excinfo.value.category is ProfilingErrorCategory.NETWORK_UNREACHABLE
        assert "no IPv4 address" in str(excinfo.value)

    async def test_typed_probe_failure_propagates(self) -> None:
        class FailingPing(FakePingRunner):
            async def probe(self, target, *, packet_count=None):
                raise ProfilingError(
                    ProfilingErrorCategory.NETWORK_UNREACHABLE, "all packets lost"
                )

        facts = _facts_map(_worker("w1", "192.168.1.10"), _worker("w2", "192.168.1.20"))
        with pytest.raises(ProfilingError) as excinfo:
            await NetworkProfiler(ping_runner=FailingPing()).profile(
                _rtt_case(), network_facts=facts, environment_fingerprint="e"
            )
        assert excinfo.value.category is ProfilingErrorCategory.NETWORK_UNREACHABLE


class TestBatchProbing:
    async def test_probe_rtt_bounded_and_ordered(self) -> None:
        """§33: the dense matrix fan-out MUST stay within the concurrency bound."""
        ping = FakePingRunner()
        profiler = NetworkProfiler(ping_runner=ping)
        workers = [_worker(f"w{i}", f"192.168.1.{i + 10}") for i in range(5)]
        facts = _facts_map(*workers)
        cases = rtt_matrix_cases(workers, packet_count=5)
        assert len(cases) == 20
        records = await profiler.probe_rtt(
            cases, network_facts=facts, environment_fingerprint="e", concurrency=3
        )
        assert len(records) == 20
        assert [record.case_id for record in records] == [case.case_id for case in cases]
        assert ping.max_active <= 3

    async def test_probe_rtt_invalid_concurrency_rejected(self) -> None:
        profiler = NetworkProfiler(ping_runner=FakePingRunner())
        with pytest.raises(ValueError, match="concurrency"):
            await profiler.probe_rtt(
                [], network_facts={}, environment_fingerprint="e", concurrency=0
            )

    async def test_probe_bandwidth_is_strictly_sequential(self) -> None:
        """§35: one flow at a time — concurrency would contaminate the baseline."""
        iperf = FakeIperfRunner()
        profiler = NetworkProfiler(iperf_runner=iperf)
        facts = _facts_map(_worker("w1", "192.168.1.10"), _worker("w2", "192.168.1.20"))
        cases = [
            _bandwidth_case(direction=NetworkDirection.FORWARD),
            _bandwidth_case(direction=NetworkDirection.REVERSE),
        ]
        records = await profiler.probe_bandwidth(
            cases, network_facts=facts, environment_fingerprint="e"
        )
        assert len(records) == 2
        assert [record.case_id for record in records] == [case.case_id for case in cases]
        assert iperf.max_active == 1

    async def test_records_carry_distinct_measurement_ids(self) -> None:
        profiler = NetworkProfiler(ping_runner=FakePingRunner())
        facts = _facts_map(_worker("w1", "192.168.1.10"), _worker("w2", "192.168.1.20"))
        records = await profiler.probe_rtt(
            [_rtt_case(), _rtt_case()], network_facts=facts, environment_fingerprint="e"
        )
        ids = [record.measurement_id for record in records]
        assert len(set(ids)) == 2
        # Same request twice: identical canonical case identity (§7, §20).
        assert records[0].case_id == records[1].case_id
        assert isinstance(records[0], MeasurementRecord)
