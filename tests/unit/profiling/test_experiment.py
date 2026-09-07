"""Experiments, cases, and typed failures (spec §8, §24, §42)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from edgeshard.profiling.domain.experiment import (
    CaseOutcome,
    CaseState,
    ExperimentState,
    ModelCaseSpec,
    NetworkCaseSpec,
    ProfilingCase,
    ProfilingErrorCategory,
    ProfilingExperiment,
    ProfilingFailure,
    profiling_case_id,
    profiling_experiment_id,
)
from edgeshard.profiling.domain.hashing import normalized_items
from edgeshard.profiling.domain.measurement import (
    LatencyMetrics,
    MeasurementMetrics,
    MeasurementRecord,
    TimeUnit,
    summarize_samples,
)
from edgeshard.profiling.domain.model import ModelReference
from edgeshard.profiling.domain.network import (
    NetworkDirection,
    NetworkMeasurementRegime,
    NetworkPathClass,
    NetworkTransport,
    ProbeKind,
)
from edgeshard.profiling.domain.signature import (
    GemmSignature,
    InferencePhase,
    ModuleKind,
    ModuleSignature,
    OperatorKind,
    OperatorSignature,
    ProfilingGranularity,
    TransformerLayerSignature,
)

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)

LAYER = TransformerLayerSignature(
    architecture_family="qwen2",
    layer_type="decoder",
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=16,
    dtype="bf16",
    quantization=None,
)
MODULE = ModuleSignature(
    kind=ModuleKind.ATTENTION,
    architecture_family="qwen2",
    structural_parameters=normalized_items({"hidden_size": 64}, "p"),
    dtype="bf16",
    quantization=None,
)
OPERATOR = OperatorSignature(
    kind=OperatorKind.GEMM,
    parameters=GemmSignature(m=512, n=64, k=64, dtype="bf16"),
    backend_family="torch",
)


def _model_case(**overrides: object) -> ModelCaseSpec:
    base: dict[str, object] = {
        "granularity": ProfilingGranularity.TRANSFORMER_LAYER,
        "model": ModelReference(model_id="tiny/qwen2", revision="r1"),
        "layer_signature": LAYER,
        "device_ids": ("GPU-uuid-1",),
        "dtype": "bf16",
        "sequence_length": 512,
    }
    base.update(overrides)
    return ModelCaseSpec(**base)  # type: ignore[arg-type]


def _network_case(**overrides: object) -> NetworkCaseSpec:
    base: dict[str, object] = {
        "probe_kind": ProbeKind.RTT,
        "source_worker_id": "w-a",
        "destination_worker_id": "w-b",
    }
    base.update(overrides)
    return NetworkCaseSpec(**base)  # type: ignore[arg-type]


def test_error_category_vocabulary_matches_spec() -> None:
    """§42: the fixed 14-category typed-failure vocabulary."""
    assert {category.value for category in ProfilingErrorCategory} == {
        "unsupported_model",
        "unsupported_granularity",
        "unsupported_phase",
        "unsupported_operator",
        "device_busy",
        "insufficient_memory",
        "export_failed",
        "profiler_failed",
        "benchmark_failed",
        "network_unreachable",
        "iperf_unavailable",
        "timeout",
        "cancelled",
        "internal_error",
    }


def test_failure_requires_message_and_normalized_details() -> None:
    failure = ProfilingFailure(
        category=ProfilingErrorCategory.DEVICE_BUSY,
        message="gpu busy",
        details=normalized_items({"utilization": 97.5}, "details"),
    )
    assert failure.category is ProfilingErrorCategory.DEVICE_BUSY
    with pytest.raises(ValueError, match="message"):
        ProfilingFailure(category=ProfilingErrorCategory.TIMEOUT, message="")


def test_lifecycle_state_vocabularies() -> None:
    assert {state.value for state in ExperimentState} == {
        "pending",
        "running",
        "completed",
        "partially_completed",
        "cancelled",
        "failed",
    }
    assert {state.value for state in CaseState} == {
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
    }


def test_granularity_requires_matching_signature() -> None:
    with pytest.raises(ValueError, match="matching"):
        _model_case(layer_signature=None)
    with pytest.raises(ValueError, match="absent"):
        _model_case(module_signature=MODULE)
    module_case = _model_case(
        granularity=ProfilingGranularity.MODULE,
        layer_signature=None,
        module_signature=MODULE,
    )
    assert module_case.granularity is ProfilingGranularity.MODULE


def test_operator_cases_are_model_free() -> None:
    """§25: microbenchmarks run without loading a checkpoint."""
    case = _model_case(
        granularity=ProfilingGranularity.OPERATOR,
        layer_signature=None,
        operator_signature=OPERATOR,
        model=None,
    )
    assert case.model is None
    with pytest.raises(ValueError, match="model reference"):
        _model_case(
            granularity=ProfilingGranularity.MODULE,
            layer_signature=None,
            module_signature=MODULE,
            model=None,
        )


def test_decode_requires_context_length_and_prefill_forbids_it() -> None:
    """§24: decode is never silently approximated by prefill."""
    with pytest.raises(ValueError, match="context_length"):
        _model_case(phase=InferencePhase.DECODE)
    with pytest.raises(ValueError, match="context_length"):
        _model_case(phase=InferencePhase.PREFILL, context_length=128)
    decode = _model_case(phase=InferencePhase.DECODE, sequence_length=1, context_length=512)
    assert decode.context_length == 512
    with pytest.raises(ValueError, match="context_length"):
        _model_case(phase=InferencePhase.DECODE, context_length=0)


def test_model_case_device_and_shape_validation() -> None:
    with pytest.raises(ValueError, match="device_ids"):
        _model_case(device_ids=())
    with pytest.raises(ValueError, match="duplicate device_id"):
        _model_case(device_ids=("GPU-uuid-1", "GPU-uuid-1"))
    with pytest.raises(ValueError, match="sequence_length"):
        _model_case(sequence_length=0)
    with pytest.raises(ValueError, match="batch_size"):
        _model_case(batch_size=0)


def test_case_id_is_canonical_and_deduplicates() -> None:
    """§7: identical requests hash identically — re-dispatch is idempotent."""
    spec = _model_case()
    assert profiling_case_id("w-1", spec) == profiling_case_id("w-1", _model_case())
    assert profiling_case_id("w-1", spec) != profiling_case_id("w-2", spec)
    with pytest.raises(ValueError, match="worker_id"):
        profiling_case_id("", spec)


def test_case_for_spec_computes_canonical_id() -> None:
    case = ProfilingCase.for_spec("w-1", _model_case())
    assert case.case_id == profiling_case_id("w-1", case.spec)


def test_rtt_case_rejects_bandwidth_fields() -> None:
    with pytest.raises(ValueError, match="transport"):
        _network_case(transport=NetworkTransport.TCP)
    with pytest.raises(ValueError, match="duration_s"):
        _network_case(duration_s=3.0)
    with pytest.raises(ValueError, match="packet_count"):
        _network_case(packet_count=0)


def test_bandwidth_case_requires_transport_direction_duration() -> None:
    with pytest.raises(ValueError, match="transport"):
        _network_case(probe_kind=ProbeKind.BANDWIDTH)
    with pytest.raises(ValueError, match="direction"):
        _network_case(probe_kind=ProbeKind.BANDWIDTH, transport=NetworkTransport.TCP)
    with pytest.raises(ValueError, match="duration_s"):
        _network_case(
            probe_kind=ProbeKind.BANDWIDTH,
            transport=NetworkTransport.TCP,
            direction=NetworkDirection.FORWARD,
        )
    with pytest.raises(ValueError, match="duration_s"):
        _network_case(
            probe_kind=ProbeKind.BANDWIDTH,
            transport=NetworkTransport.TCP,
            direction=NetworkDirection.FORWARD,
            duration_s=0.0,
        )
    case = _network_case(
        probe_kind=ProbeKind.BANDWIDTH,
        transport=NetworkTransport.TCP,
        direction=NetworkDirection.REVERSE,
        duration_s=3.0,
        payload_bytes=3_670_016,
        path_class=NetworkPathClass.WIRED_LAN,
    )
    assert case.regime is NetworkMeasurementRegime.IDLE_SINGLE_FLOW


def test_network_case_rejects_self_probe() -> None:
    with pytest.raises(ValueError, match="itself"):
        _network_case(destination_worker_id="w-a")


def test_network_case_executes_on_source_worker() -> None:
    case = ProfilingCase.for_spec("w-a", _network_case())
    assert case.worker_id == "w-a"
    with pytest.raises(ValueError, match="source worker"):
        ProfilingCase(
            case_id="c-x", worker_id="w-b", spec=_network_case()
        )


def test_experiment_validation() -> None:
    experiment = ProfilingExperiment(
        experiment_id="exp-1",
        strategy_id="default-v1",
        created_at=NOW,
        requested_by=None,
        case_ids=("c-1", "c-2"),
    )
    assert experiment.case_ids == ("c-1", "c-2")
    with pytest.raises(ValueError, match="duplicate case_id"):
        ProfilingExperiment(
            experiment_id="exp-1",
            strategy_id="default-v1",
            created_at=NOW,
            requested_by=None,
            case_ids=("c-1", "c-1"),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        ProfilingExperiment(
            experiment_id="exp-1",
            strategy_id="default-v1",
            created_at=datetime(2026, 9, 7, 12, 0),
            requested_by=None,
            case_ids=(),
        )
    with pytest.raises(ValueError, match="strategy_id"):
        ProfilingExperiment(
            experiment_id="exp-1",
            strategy_id="",
            created_at=NOW,
            requested_by=None,
            case_ids=(),
        )


# ---------------------------------------------------------------------------
# P2G: layer positions (§22) and case outcomes (§42)
# ---------------------------------------------------------------------------


def _measurement(case_id: str = "c-1") -> MeasurementRecord:
    return MeasurementRecord(
        measurement_id=f"m-{case_id}",
        case_id=case_id,
        environment_fingerprint="fp-1",
        started_at=NOW,
        finished_at=NOW,
        sample_count=1,
        samples=(1.0,),
        metrics=MeasurementMetrics(
            latency=LatencyMetrics(
                summary=summarize_samples((1.0,)), unit=TimeUnit.MILLISECONDS
            )
        ),
    )


def test_layer_index_is_transformer_layer_only() -> None:
    """§22: the positional check applies to enumerated layers, nothing else."""
    assert _model_case(layer_index=0).layer_index == 0
    with pytest.raises(ValueError, match="transformer-layer"):
        _model_case(
            granularity=ProfilingGranularity.MODULE,
            layer_signature=None,
            module_signature=MODULE,
            layer_index=1,
        )
    with pytest.raises(ValueError, match="transformer-layer"):
        _model_case(
            granularity=ProfilingGranularity.OPERATOR,
            layer_signature=None,
            operator_signature=OPERATOR,
            layer_index=1,
        )


def test_layer_index_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="layer_index"):
        _model_case(layer_index=-1)


def test_layer_index_keeps_positional_cases_distinct() -> None:
    """§7 + §22: same-shape layers at different positions are different cases.

    Without ``layer_index`` the early/middle/late sparse probes (§47 step 6)
    would deduplicate into one case id and the positional check could not be
    expressed; with it, positions stay distinct while the un-indexed spec
    keeps its reuse-identity meaning (``None`` = first matching layer).
    """
    early = ProfilingCase.for_spec("w-1", _model_case(layer_index=0))
    middle = ProfilingCase.for_spec("w-1", _model_case(layer_index=2))
    late = ProfilingCase.for_spec("w-1", _model_case(layer_index=3))
    unindexed = ProfilingCase.for_spec("w-1", _model_case())
    ids = {early.case_id, middle.case_id, late.case_id, unindexed.case_id}
    assert len(ids) == 4


def test_case_outcome_carries_exactly_one_member() -> None:
    """§42: a success travels with its record, a failure with its category."""
    record = _measurement()
    failure = ProfilingFailure(
        category=ProfilingErrorCategory.DEVICE_BUSY, message="device busy"
    )
    success = CaseOutcome.from_record(record)
    assert success.succeeded and success.record is record and success.failure is None
    failed = CaseOutcome.from_failure(failure)
    assert not failed.succeeded and failed.failure is failure and failed.record is None
    with pytest.raises(ValueError, match="exactly one"):
        CaseOutcome()
    with pytest.raises(ValueError, match="exactly one"):
        CaseOutcome(record=record, failure=failure)


def test_experiment_id_is_canonical_and_order_insensitive() -> None:
    """§7: the id hashes the strategy plus the *set* of cases."""
    first = profiling_experiment_id("default", ["c-1", "c-2"])
    assert len(first) == 64
    assert first == profiling_experiment_id("default", ["c-2", "c-1"])
    assert first != profiling_experiment_id("other-strategy", ["c-1", "c-2"])
    assert first != profiling_experiment_id("default", ["c-1", "c-3"])


def test_experiment_id_requires_strategy_id() -> None:
    with pytest.raises(ValueError, match="strategy_id"):
        profiling_experiment_id("", ["c-1"])


def test_experiment_for_cases_dedupes_and_precomputes_id() -> None:
    created = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
    experiment = ProfilingExperiment.for_cases(
        strategy_id="default",
        case_ids=["c-2", "c-1", "c-2"],
        created_at=created,
        requested_by="operator",
    )
    assert experiment.case_ids == ("c-2", "c-1")
    assert experiment.experiment_id == profiling_experiment_id(
        "default", ["c-1", "c-2"]
    )
    assert experiment.requested_by == "operator"
    assert experiment.created_at == created
