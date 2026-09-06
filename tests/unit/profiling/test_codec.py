"""P2G JSON payload codec tests (spec §41, §43-46).

The codec carries domain DTOs across the profiling wire protocol and into
ProfileStore payload columns. Pinned here: lossless round-trips for every
variant union the domain declares (case specs, operator parameters, metric
objects), deterministic canonical output, and strict typed decoding —
missing/unknown fields, tag mismatches, and numeric-type violations fail
loudly instead of being repaired, and the domain ``__post_init__``
validation re-runs at every decode boundary.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from edgeshard.profiling.codec import (
    TYPE_TAG,
    PayloadCodecError,
    decode_json,
    decode_payload,
    encode_json,
    encode_payload,
)
from edgeshard.profiling.domain.environment import (
    DevicePerformanceClass,
    EnvironmentFingerprint,
    MemoryModel,
)
from edgeshard.profiling.domain.experiment import (
    CaseOutcome,
    ModelCaseSpec,
    NetworkCaseSpec,
    ProfilingCase,
    ProfilingErrorCategory,
    ProfilingExperiment,
    ProfilingFailure,
)
from edgeshard.profiling.domain.measurement import (
    AllocatorMemoryMetrics,
    BandwidthMetrics,
    LatencyMetrics,
    MeasurementMetrics,
    MeasurementRecord,
    PhysicalMemoryMetrics,
    RttMetrics,
    TelemetryContextMetrics,
    TelemetrySample,
    TimeUnit,
    summarize_samples,
)
from edgeshard.profiling.domain.model import (
    ModelCharacterization,
    ModelReference,
    ModelStage,
    StageKind,
)
from edgeshard.profiling.domain.network import (
    NetworkDirection,
    NetworkPathClass,
    NetworkTransport,
    ProbeKind,
)
from edgeshard.profiling.domain.session import (
    LayerEntry,
    ModelSessionFacts,
    ModuleEntry,
    ProfilingSessionKind,
    ProfilingSessionRequest,
)
from edgeshard.profiling.domain.signature import (
    AttentionSignature,
    CustomOperatorParameters,
    EmbeddingSignature,
    GemmSignature,
    GenericOperatorParameters,
    InferencePhase,
    KvCopySignature,
    ModuleKind,
    ModuleSignature,
    NormSignature,
    NormVariant,
    OperatorKind,
    OperatorSignature,
    ProfilingGranularity,
    RotarySignature,
    TransformerLayerSignature,
)
from edgeshard.profiling.domain.snapshot import ProfileSnapshot
from edgeshard.profiling.network.classifier import (
    InterfaceFacts,
    InterfaceKind,
    WorkerNetworkFacts,
)

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
LATER = datetime(2026, 9, 7, 12, 5, tzinfo=UTC)

LAYER_SIG = TransformerLayerSignature(
    architecture_family="llama",
    layer_type="standard_decoder",
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=8,
    dtype="fp32",
    quantization=None,
)
MODULE_SIG = ModuleSignature(
    kind=ModuleKind.MLP,
    architecture_family="llama",
    structural_parameters=(("hidden_size", 32), ("intermediate_size", 64)),
    dtype="fp32",
    quantization=None,
)
MODEL = ModelReference(model_id="tiny/llama", revision="local")


def _operator(kind: OperatorKind, parameters: object) -> OperatorSignature:
    return OperatorSignature(
        kind=kind, parameters=parameters, backend_family="torch"  # type: ignore[arg-type]
    )


OPERATOR_VARIANTS = [
    _operator(OperatorKind.GEMM, GemmSignature(m=512, n=64, k=32, dtype="bf16", transpose_a=True)),
    _operator(
        OperatorKind.ATTENTION,
        AttentionSignature(
            batch_size=1,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            q_len=1,
            kv_len=512,
            dtype="fp32",
            phase=InferencePhase.DECODE,
        ),
    ),
    _operator(
        OperatorKind.NORM,
        NormSignature(
            batch_size=2,
            sequence_length=512,
            hidden_size=32,
            dtype="fp32",
            variant=NormVariant.RMS,
        ),
    ),
    _operator(
        OperatorKind.EMBEDDING,
        EmbeddingSignature(
            batch_size=1, sequence_length=512, vocab_size=64, embedding_dim=32, dtype="fp32"
        ),
    ),
    _operator(
        OperatorKind.ROTARY,
        RotarySignature(batch_size=1, sequence_length=512, num_heads=4, head_dim=8, dtype="fp32"),
    ),
    _operator(
        OperatorKind.KV_COPY,
        KvCopySignature(batch_size=1, num_kv_heads=2, head_dim=8, context_length=512, dtype="fp32"),
    ),
    _operator(
        OperatorKind.ELEMENTWISE,
        GenericOperatorParameters(operation="silu", input_shapes=((2, 64),), dtype="fp32"),
    ),
    _operator(
        OperatorKind.CUSTOM,
        CustomOperatorParameters(
            raw_name="my_kernel",
            input_shapes=((2, 64), (64, 32)),
            input_dtypes=("fp32", "fp32"),
            metadata=(("count", 3), ("source", "graph")),
        ),
    ),
]


def _record(metrics: MeasurementMetrics, **overrides: object) -> MeasurementRecord:
    base: dict[str, object] = {
        "measurement_id": "m-1",
        "case_id": "c-1",
        "environment_fingerprint": "fp-1",
        "started_at": NOW,
        "finished_at": LATER,
        "sample_count": 5,
        "samples": (1.0, 2.0, 3.0, 4.0, 5.0),
        "metrics": metrics,
        "metadata": (
            ("batch_size", 1),
            ("device_id", "gpu-0"),
            ("note", None),
            ("ok", True),
            ("ratio", 0.5),
        ),
    }
    base.update(overrides)
    return MeasurementRecord(**base)  # type: ignore[arg-type]


LATENCY_RECORD = _record(
    MeasurementMetrics(
        latency=LatencyMetrics(
            summary=summarize_samples((1.0, 2.0, 3.0, 4.0, 5.0)),
            unit=TimeUnit.MILLISECONDS,
        )
    )
)
ALLOCATOR_RECORD = _record(
    MeasurementMetrics(
        latency=LatencyMetrics(
            summary=summarize_samples((1.0,)), unit=TimeUnit.MILLISECONDS
        ),
        allocator_memory=AllocatorMemoryMetrics(
            allocated_before=1024,
            reserved_before=2048,
            peak_allocated=4096,
            peak_reserved=8192,
            allocated_after=1024,
            reserved_after=8192,
        ),
    ),
    sample_count=1,
    samples=(1.0,),
)
PHYSICAL_RECORD = _record(
    MeasurementMetrics(
        latency=LatencyMetrics(
            summary=summarize_samples((1.0,)), unit=TimeUnit.MILLISECONDS
        ),
        physical_memory=PhysicalMemoryMetrics(
            pool_id="cuda-0", used_before=1 << 30, used_peak=None, used_after=1 << 30
        ),
    ),
    sample_count=1,
    samples=(1.0,),
)
TELEMETRY_RECORD = _record(
    MeasurementMetrics(
        latency=LatencyMetrics(
            summary=summarize_samples((1.0,)), unit=TimeUnit.MILLISECONDS
        ),
        telemetry=TelemetryContextMetrics(
            initial=TelemetrySample(device_id="gpu-0", utilization=3.0, temperature_c=41.0),
            final=TelemetrySample(device_id="gpu-0", utilization=97.0, memory_used_bytes=1 << 30),
            contaminated=False,
        ),
    ),
    sample_count=1,
    samples=(1.0,),
)
RTT_RECORD = _record(
    MeasurementMetrics(
        rtt=RttMetrics(
            summary=summarize_samples((0.4, 0.5, 0.6, 0.7, 0.8)),
            unit=TimeUnit.MILLISECONDS,
            packets_sent=5,
            packets_received=5,
        )
    )
)
BANDWIDTH_RECORD = _record(
    MeasurementMetrics(
        bandwidth=BandwidthMetrics(
            bits_per_second=9.3e9, payload_bytes=4194304, retransmits=7, duration_s=5.0
        )
    )
)

FAILURE = ProfilingFailure(
    category=ProfilingErrorCategory.DEVICE_BUSY,
    message="device is busy",
    details=(("device_id", "gpu-0"), ("utilization", 97.5)),
)

CHARACTERIZATION = ModelCharacterization(
    model=MODEL,
    architecture_family="llama",
    num_layers=2,
    hidden_size=32,
    intermediate_size=64,
    vocab_size=64,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=8,
    dtype="fp32",
    quantization=None,
    tied_word_embeddings=False,
    stages=(
        ModelStage(kind=StageKind.EMBEDDING),
        ModelStage(kind=StageKind.TRANSFORMER_LAYER_GROUP, layer_count=2),
        ModelStage(kind=StageKind.LM_HEAD),
    ),
)

SESSION_FACTS = ModelSessionFacts(
    characterization=CHARACTERIZATION,
    layer_entries=(
        LayerEntry(0, "model.layers.0", LAYER_SIG),
        LayerEntry(1, "model.layers.1", LAYER_SIG),
    ),
    module_entries=(
        ModuleEntry("mlp", "model.layers.0.mlp", ModuleKind.MLP, 0, MODULE_SIG),
    ),
    operator_signatures=(OPERATOR_VARIANTS[0],),
)

NETWORK_FACTS = WorkerNetworkFacts(
    worker_id="w-1",
    hostname="host-a",
    interfaces=(
        InterfaceFacts(
            interface_id="if-eth0",
            name="eth0",
            kind=InterfaceKind.WIRED,
            overlay_type=None,
            addresses=("192.168.1.10",),
            mtu=1500,
        ),
    ),
)


class TestRoundTrips:
    @pytest.mark.parametrize("signature", OPERATOR_VARIANTS, ids=lambda s: s.kind.value)
    def test_operator_signature_variants(self, signature: OperatorSignature) -> None:
        """Every OperatorParameters variant dispatches on its __type__ tag."""
        assert decode_json(OperatorSignature, encode_json(signature)) == signature

    def test_model_case_specs(self) -> None:
        layer_case = ProfilingCase.for_spec(
            "w-1",
            ModelCaseSpec(
                granularity=ProfilingGranularity.TRANSFORMER_LAYER,
                device_ids=("gpu-0",),
                dtype="fp32",
                model=MODEL,
                layer_signature=LAYER_SIG,
                layer_index=3,
            ),
        )
        module_case = ProfilingCase.for_spec(
            "w-1",
            ModelCaseSpec(
                granularity=ProfilingGranularity.MODULE,
                device_ids=("gpu-0",),
                dtype="fp32",
                model=MODEL,
                module_signature=MODULE_SIG,
            ),
        )
        operator_case = ProfilingCase.for_spec(
            "w-1",
            ModelCaseSpec(
                granularity=ProfilingGranularity.OPERATOR,
                device_ids=("gpu-0",),
                dtype="fp32",
                operator_signature=OPERATOR_VARIANTS[0],
            ),
        )
        decode_case = ProfilingCase.for_spec(
            "w-1",
            ModelCaseSpec(
                granularity=ProfilingGranularity.MODULE,
                device_ids=("gpu-0",),
                dtype="fp32",
                model=MODEL,
                module_signature=MODULE_SIG,
                phase=InferencePhase.DECODE,
                sequence_length=1,
                context_length=512,
            ),
        )
        for case in (layer_case, module_case, operator_case, decode_case):
            assert decode_json(ProfilingCase, encode_json(case)) == case

    def test_network_case_specs(self) -> None:
        rtt_case = ProfilingCase.for_spec(
            "w-1",
            NetworkCaseSpec(
                probe_kind=ProbeKind.RTT,
                source_worker_id="w-1",
                destination_worker_id="w-2",
                packet_count=5,
            ),
        )
        bandwidth_case = ProfilingCase.for_spec(
            "w-1",
            NetworkCaseSpec(
                probe_kind=ProbeKind.BANDWIDTH,
                source_worker_id="w-1",
                destination_worker_id="w-2",
                path_class=NetworkPathClass.WIRED_LAN,
                transport=NetworkTransport.TCP,
                direction=NetworkDirection.FORWARD,
                duration_s=5.0,
                payload_bytes=4194304,
            ),
        )
        for case in (rtt_case, bandwidth_case):
            assert decode_json(ProfilingCase, encode_json(case)) == case

    @pytest.mark.parametrize(
        "record",
        [
            LATENCY_RECORD,
            ALLOCATOR_RECORD,
            PHYSICAL_RECORD,
            TELEMETRY_RECORD,
            RTT_RECORD,
            BANDWIDTH_RECORD,
        ],
    )
    def test_measurement_record_metric_variants(
        self, record: MeasurementRecord
    ) -> None:
        assert decode_json(MeasurementRecord, encode_json(record)) == record

    def test_measurement_metadata_scalar_fidelity(self) -> None:
        """JsonScalar keeps JSON type fidelity: int stays int, bool stays bool."""
        decoded = decode_json(MeasurementRecord, encode_json(LATENCY_RECORD))
        mapping = decoded.metadata_mapping
        assert mapping["batch_size"] == 1 and type(mapping["batch_size"]) is int
        assert mapping["ok"] is True
        assert mapping["ratio"] == 0.5 and type(mapping["ratio"]) is float
        assert mapping["note"] is None

    def test_case_outcomes(self) -> None:
        for outcome in (
            CaseOutcome.from_record(LATENCY_RECORD),
            CaseOutcome.from_failure(FAILURE),
        ):
            assert decode_json(CaseOutcome, encode_json(outcome)) == outcome

    def test_experiment_and_failure(self) -> None:
        experiment = ProfilingExperiment(
            experiment_id="exp-1",
            strategy_id="default-v1",
            created_at=NOW,
            requested_by=None,
            case_ids=("c-1", "c-2"),
        )
        assert decode_json(ProfilingExperiment, encode_json(experiment)) == experiment
        assert decode_json(ProfilingFailure, encode_json(FAILURE)) == FAILURE

    def test_environment_types(self) -> None:
        fingerprint = EnvironmentFingerprint(
            backend="torch",
            profiling_implementation_revision="r1",
            device_performance_class_id="dpc-1",
            torch_version="2.13.0",
            worker_id="w-1",
            device_id="gpu-0",
        )
        performance_class = DevicePerformanceClass(
            vendor="nvidia",
            accelerator_model="rtx4090",
            memory_model=MemoryModel.DISCRETE,
            backend_family="torch",
            architecture="ada",
            software_versions=(("cuda", "12.4"),),
        )
        assert decode_json(EnvironmentFingerprint, encode_json(fingerprint)) == fingerprint
        assert (
            decode_json(DevicePerformanceClass, encode_json(performance_class))
            == performance_class
        )

    def test_session_types(self) -> None:
        request = ProfilingSessionRequest(
            kind=ProfilingSessionKind.MODEL,
            device_ids=("gpu-0",),
            model=MODEL,
            dtype="fp32",
        )
        assert decode_json(ProfilingSessionRequest, encode_json(request)) == request
        assert decode_json(ModelSessionFacts, encode_json(SESSION_FACTS)) == SESSION_FACTS

    def test_network_facts(self) -> None:
        assert decode_json(WorkerNetworkFacts, encode_json(NETWORK_FACTS)) == NETWORK_FACTS

    def test_profile_snapshot(self) -> None:
        snapshot = ProfileSnapshot(
            snapshot_id="snap-1",
            created_at=NOW,
            model_characterizations=(CHARACTERIZATION,),
            measurements=(replace(LATENCY_RECORD, measurement_id="m-latency"),),
            network_measurements=(
                replace(RTT_RECORD, measurement_id="m-rtt"),
                replace(BANDWIDTH_RECORD, measurement_id="m-bandwidth"),
            ),
        )
        assert decode_json(ProfileSnapshot, encode_json(snapshot)) == snapshot


class TestEncoding:
    def test_output_is_deterministic_and_canonical(self) -> None:
        text = encode_json(LATENCY_RECORD)
        assert text == encode_json(decode_json(MeasurementRecord, text))
        assert text == json.dumps(
            json.loads(text), sort_keys=True, separators=(",", ":")
        )

    def test_tag_enum_and_datetime_shapes(self) -> None:
        payload = encode_payload(FAILURE)
        assert payload[TYPE_TAG] == "ProfilingFailure"
        assert payload["category"] == "device_busy"  # StrEnum encodes as value
        assert encode_payload(NOW) == NOW.isoformat()

    def test_nested_containers(self) -> None:
        payload = encode_payload((("a", 1), ("b", None)))
        assert payload == [["a", 1], ["b", None]]
        assert encode_payload({"k": (1, 2)}) == {"k": [1, 2]}

    def test_non_finite_floats_rejected(self) -> None:
        with pytest.raises(PayloadCodecError, match="not JSON-serializable"):
            encode_json(float("nan"))

    def test_unencodable_values_rejected(self) -> None:
        with pytest.raises(PayloadCodecError, match="cannot encode"):
            encode_payload(object())
        with pytest.raises(PayloadCodecError, match="mapping keys"):
            encode_payload({1: "x"})

    def test_dataclass_types_are_not_instances(self) -> None:
        with pytest.raises(PayloadCodecError, match="cannot encode"):
            encode_payload(ProfilingFailure)


class TestStrictDecoding:
    def test_malformed_json(self) -> None:
        with pytest.raises(PayloadCodecError, match="malformed JSON"):
            decode_json(ProfilingFailure, "{not json")

    def test_wrong_shape_for_dataclass(self) -> None:
        with pytest.raises(PayloadCodecError, match="expected"):
            decode_json(ProfilingFailure, "[1, 2]")

    def test_missing_or_wrong_tag(self) -> None:
        with pytest.raises(PayloadCodecError, match="does not match"):
            decode_json(ProfilingFailure, json.dumps({"category": "device_busy"}))
        with pytest.raises(PayloadCodecError, match="does not match"):
            decode_json(
                ProfilingFailure,
                json.dumps({TYPE_TAG: "MeasurementRecord", "category": "device_busy"}),
            )

    def test_unknown_and_missing_fields(self) -> None:
        payload = json.loads(encode_json(FAILURE))
        payload["extra"] = 1
        with pytest.raises(PayloadCodecError, match="unknown field"):
            decode_json(ProfilingFailure, json.dumps(payload))
        del payload["message"]
        del payload["extra"]
        with pytest.raises(PayloadCodecError, match="missing field"):
            decode_json(ProfilingFailure, json.dumps(payload))

    def test_wrong_target_class_rejected(self) -> None:
        text = encode_json(
            ProfilingCase.for_spec(
                "w-1",
                NetworkCaseSpec(
                    probe_kind=ProbeKind.RTT,
                    source_worker_id="w-1",
                    destination_worker_id="w-2",
                ),
            )
        )
        with pytest.raises(PayloadCodecError, match="does not match"):
            decode_json(CaseOutcome, text)

    def test_numeric_rules(self) -> None:
        """int fields reject bool and float; float fields widen int."""
        base = json.loads(encode_json(RTT_RECORD))
        base["sample_count"] = True
        with pytest.raises(PayloadCodecError, match="expected"):
            decode_json(MeasurementRecord, json.dumps(base))
        base["sample_count"] = 5.0
        with pytest.raises(PayloadCodecError, match="expected"):
            decode_json(MeasurementRecord, json.dumps(base))
        base["sample_count"] = "5"
        with pytest.raises(PayloadCodecError, match="expected"):
            decode_json(MeasurementRecord, json.dumps(base))
        bandwidth = json.loads(encode_json(BANDWIDTH_RECORD))
        bandwidth["metrics"]["bandwidth"]["bits_per_second"] = 9
        decoded = decode_json(MeasurementRecord, json.dumps(bandwidth))
        assert decoded.metrics.bandwidth is not None
        assert decoded.metrics.bandwidth.bits_per_second == 9.0
        assert type(decoded.metrics.bandwidth.bits_per_second) is float

    def test_enum_membership_enforced(self) -> None:
        payload = json.loads(encode_json(LATENCY_RECORD))
        payload["metrics"]["latency"]["unit"] = "fortnights"
        with pytest.raises(PayloadCodecError, match="TimeUnit"):
            decode_json(MeasurementRecord, json.dumps(payload))

    def test_fixed_tuple_length_enforced(self) -> None:
        payload = json.loads(encode_json(FAILURE))
        payload["details"] = [["device_id", "gpu-0", "extra"]]
        with pytest.raises(PayloadCodecError, match="tuple items"):
            decode_json(ProfilingFailure, json.dumps(payload))
        payload["details"] = "device_id"
        with pytest.raises(PayloadCodecError, match="expected"):
            decode_json(ProfilingFailure, json.dumps(payload))

    def test_variant_tag_commits(self) -> None:
        """A known variant tag commits: its own errors surface, no fall-through."""
        payload = json.loads(
            encode_json(
                ProfilingCase.for_spec(
                    "w-1",
                    ModelCaseSpec(
                        granularity=ProfilingGranularity.OPERATOR,
                        device_ids=("gpu-0",),
                        dtype="fp32",
                        operator_signature=OPERATOR_VARIANTS[0],
                    ),
                )
            )
        )
        del payload["spec"]["dtype"]
        with pytest.raises(PayloadCodecError, match="missing field 'dtype'"):
            decode_json(ProfilingCase, json.dumps(payload))

    def test_unknown_variant_tag_lists_expected(self) -> None:
        case_payload = {
            TYPE_TAG: "ProfilingCase",
            "case_id": "c-1",
            "worker_id": "w-1",
            "spec": {TYPE_TAG: "QuantumCaseSpec"},
        }
        with pytest.raises(PayloadCodecError, match="ModelCaseSpec"):
            decode_json(ProfilingCase, json.dumps(case_payload))

    def test_domain_validation_reruns_at_boundary(self) -> None:
        """Invalid domain values are rejected, never repaired (§41)."""
        payload = json.loads(encode_json(FAILURE))
        payload["message"] = ""
        with pytest.raises(PayloadCodecError, match="invalid ProfilingFailure"):
            decode_json(ProfilingFailure, json.dumps(payload))

    def test_naive_datetime_rejected_by_domain_rule(self) -> None:
        payload = json.loads(encode_json(LATENCY_RECORD))
        payload["started_at"] = datetime(2026, 9, 7, 12, 0).isoformat()
        with pytest.raises(PayloadCodecError, match="invalid MeasurementRecord"):
            decode_json(MeasurementRecord, json.dumps(payload))

    def test_non_iso_datetime_rejected(self) -> None:
        payload = json.loads(encode_json(LATENCY_RECORD))
        payload["started_at"] = "yesterday noon"
        with pytest.raises(PayloadCodecError, match="ISO-8601"):
            decode_json(MeasurementRecord, json.dumps(payload))

    def test_decode_payload_top_level_type_check(self) -> None:
        """Scalar payloads decode for scalar targets and fail otherwise."""
        assert decode_payload(str, "fp-1") == "fp-1"
        with pytest.raises(PayloadCodecError, match="expected"):
            decode_payload(int, "fp-1")
