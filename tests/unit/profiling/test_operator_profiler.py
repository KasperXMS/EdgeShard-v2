"""P2E OperatorProfiler tests (spec §25-§27, P2E DoD).

Operator profiling runs *without model execution*: every case here is a
synthetic workload measured through the shared §12 harness with CPU
wall-clock timing. Pinned: the case-spec derivation from a signature
(model-free, phase/context from the signature's own facts), the record
shape and metadata, and the typed failures — unregistered kinds, custom
operators, backend-family mismatches (§27), and contradictory requests.
"""

from __future__ import annotations

from typing import Any

import pytest

from edgeshard.profiling.benchmark.harness import InstrumentationBundle
from edgeshard.profiling.benchmark.sampling import DurationSamplingPolicy
from edgeshard.profiling.domain.experiment import (
    ModelCaseSpec,
    NetworkCaseSpec,
    ProfilingCase,
    ProfilingErrorCategory,
)
from edgeshard.profiling.domain.measurement import MeasurementRecord, TimeUnit
from edgeshard.profiling.domain.model import ModelReference
from edgeshard.profiling.domain.network import ProbeKind
from edgeshard.profiling.domain.signature import (
    AttentionSignature,
    CustomOperatorParameters,
    EmbeddingSignature,
    GemmSignature,
    GenericOperatorParameters,
    InferencePhase,
    ModuleKind,
    ModuleSignature,
    NormSignature,
    NormVariant,
    OperatorKind,
    OperatorSignature,
    ProfilingGranularity,
    TransformerLayerSignature,
    operator_signature_id,
)
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.instrumentation.timing import WallClockTimer
from edgeshard.profiling.operator.profiler import OperatorProfiler, operator_case_spec


def _policy() -> DurationSamplingPolicy:
    return DurationSamplingPolicy(
        min_warmups=1, min_runs=3, max_runs=3, target_duration_ms=1_000_000.0
    )


def _gemm(**overrides: Any) -> OperatorSignature:
    params: dict[str, Any] = {"m": 8, "n": 16, "k": 4, "dtype": "fp32", "transpose_b": True}
    params.update(overrides)
    return OperatorSignature(
        kind=OperatorKind.GEMM, parameters=GemmSignature(**params), backend_family="torch"
    )


def _attention(**overrides: Any) -> OperatorSignature:
    params: dict[str, Any] = {
        "batch_size": 1,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 16,
        "q_len": 8,
        "kv_len": 8,
        "dtype": "fp32",
        "phase": InferencePhase.PREFILL,
    }
    params.update(overrides)
    return OperatorSignature(
        kind=OperatorKind.ATTENTION,
        parameters=AttentionSignature(**params),
        backend_family="torch",
    )


def _norm(**overrides: Any) -> OperatorSignature:
    params: dict[str, Any] = {
        "batch_size": 1,
        "sequence_length": 8,
        "hidden_size": 64,
        "dtype": "fp32",
        "variant": NormVariant.RMS,
    }
    params.update(overrides)
    return OperatorSignature(
        kind=OperatorKind.NORM, parameters=NormSignature(**params), backend_family="torch"
    )


def _case(signature: OperatorSignature, *, worker_id: str = "worker-test") -> ProfilingCase:
    return ProfilingCase.for_spec(
        worker_id, operator_case_spec(signature, device_id="cpu-0")
    )


def _profile(signature: OperatorSignature, **kwargs: Any) -> MeasurementRecord:
    return OperatorProfiler(sampling_policy=_policy()).profile(
        _case(signature),
        instrumentation=InstrumentationBundle(timer=WallClockTimer()),
        environment_fingerprint="env-test",
        **kwargs,
    )


class TestOperatorCaseSpec:
    def test_gemm_spec_is_model_free(self) -> None:
        spec = operator_case_spec(_gemm(m=8), device_id="cpu-0")
        assert spec.granularity is ProfilingGranularity.OPERATOR
        assert spec.model is None
        assert spec.device_ids == ("cpu-0",)
        assert spec.dtype == "fp32"
        assert spec.backend == "torch"
        assert spec.phase is InferencePhase.PREFILL
        assert spec.context_length is None
        assert spec.batch_size == 1
        assert spec.sequence_length == 8  # flattened token count (m)

    def test_attention_prefill_spec(self) -> None:
        spec = operator_case_spec(_attention(batch_size=2, q_len=8), device_id="cpu-0")
        assert spec.phase is InferencePhase.PREFILL
        assert spec.context_length is None
        assert spec.batch_size == 2
        assert spec.sequence_length == 8

    def test_attention_decode_spec_carries_context(self) -> None:
        """Decode facts come from the signature; context is never guessed (§24)."""
        spec = operator_case_spec(
            _attention(q_len=1, kv_len=9, phase=InferencePhase.DECODE), device_id="cpu-0"
        )
        assert spec.phase is InferencePhase.DECODE
        assert spec.context_length == 8
        assert spec.sequence_length == 1

    def test_norm_spec_dimensions(self) -> None:
        spec = operator_case_spec(_norm(batch_size=2, sequence_length=16), device_id="cpu-0")
        assert spec.batch_size == 2
        assert spec.sequence_length == 16

    def test_generic_signature_falls_back_to_defaults(self) -> None:
        signature = OperatorSignature(
            kind=OperatorKind.ELEMENTWISE,
            parameters=GenericOperatorParameters(
                operation="silu", input_shapes=((1, 8, 64),), dtype="fp32"
            ),
            backend_family="torch",
        )
        spec = operator_case_spec(signature, device_id="cpu-0")
        assert spec.batch_size == 1
        assert spec.sequence_length == 512

    def test_custom_operator_fails_typed(self) -> None:
        signature = OperatorSignature(
            kind=OperatorKind.CUSTOM,
            parameters=CustomOperatorParameters(
                raw_name="aten.mystery.default",
                input_shapes=((2, 2),),
                input_dtypes=("fp32",),
            ),
            backend_family="torch",
        )
        with pytest.raises(ProfilingError) as excinfo:
            operator_case_spec(signature, device_id="cpu-0")
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_OPERATOR
        assert "aten.mystery.default" in str(excinfo.value)

    def test_case_id_deterministic(self) -> None:
        first = _case(_gemm())
        second = _case(_gemm())
        assert first.case_id == second.case_id
        assert _case(_gemm(m=16)).case_id != first.case_id


class TestOperatorProfiler:
    def test_gemm_end_to_end_record(self) -> None:
        signature = _gemm()
        case = _case(signature)
        record = _profile(signature)
        assert record.case_id == case.case_id
        assert record.environment_fingerprint == "env-test"
        assert record.measurement_id
        assert record.sample_count == 3
        assert record.samples is not None and len(record.samples) == 3
        latency = record.metrics.latency
        assert latency is not None
        assert latency.unit is TimeUnit.MILLISECONDS
        assert latency.summary.mean > 0.0
        # CPU-only host without memory/telemetry instruments: absent, never
        # guessed (§52.2).
        assert record.metrics.allocator_memory is None
        assert record.metrics.physical_memory is None
        assert record.metrics.telemetry is None
        assert record.finished_at >= record.started_at
        metadata = record.metadata_mapping
        assert metadata["granularity"] == "operator"
        assert metadata["operator_kind"] == "gemm"
        assert metadata["backend_family"] == "torch"
        assert metadata["operator_signature_id"] == operator_signature_id(signature)
        assert metadata["dtype"] == "fp32"
        assert metadata["phase"] == "prefill"
        assert metadata["timing_unit"] == "ms"
        assert metadata["warmup_runs"] == 1
        assert metadata["measured_runs"] == 3

    def test_attention_decode_metadata(self) -> None:
        record = _profile(_attention(q_len=1, kv_len=9, phase=InferencePhase.DECODE))
        assert record.metadata_mapping["phase"] == "decode"
        assert record.metadata_mapping["operator_kind"] == "attention"

    def test_norm_end_to_end(self) -> None:
        record = _profile(_norm())
        assert record.sample_count == 3
        assert record.metadata_mapping["operator_kind"] == "norm"

    def test_repeated_runs_distinct_ids_same_case(self) -> None:
        signature = _gemm()
        first, second = _profile(signature), _profile(signature)
        assert first.measurement_id != second.measurement_id
        assert first.case_id == second.case_id

    def test_contradictory_case_dtype_rejected(self) -> None:
        spec = ModelCaseSpec(
            granularity=ProfilingGranularity.OPERATOR,
            device_ids=("cpu-0",),
            dtype="bf16",
            operator_signature=_gemm(dtype="fp32"),
        )
        with pytest.raises(ValueError, match="contradicts"):
            OperatorProfiler().profile(
                ProfilingCase.for_spec("worker-test", spec),
                instrumentation=InstrumentationBundle(timer=WallClockTimer()),
                environment_fingerprint="env-test",
            )

    def test_unknown_signature_dtype_fails_typed(self) -> None:
        with pytest.raises(ProfilingError) as excinfo:
            _profile(_gemm(dtype="unknown"))
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_OPERATOR

    def test_unregistered_kind_fails_typed(self) -> None:
        """Kinds awaiting incremental workloads (§25) fail typed, not skipped."""
        signature = OperatorSignature(
            kind=OperatorKind.EMBEDDING,
            parameters=EmbeddingSignature(
                batch_size=1, sequence_length=8, vocab_size=32, embedding_dim=64, dtype="fp32"
            ),
            backend_family="torch",
        )
        with pytest.raises(ProfilingError) as excinfo:
            _profile(signature)
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_OPERATOR

    def test_backend_family_mismatch_fails_typed(self) -> None:
        """A torch workload must not stand in for a triton recording (§27)."""
        signature = OperatorSignature(
            kind=OperatorKind.GEMM,
            parameters=GemmSignature(m=8, n=16, k=4, dtype="fp32"),
            backend_family="triton",
        )
        with pytest.raises(ProfilingError) as excinfo:
            _profile(signature)
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_OPERATOR
        details = dict(excinfo.value.to_failure().details)
        assert details["signature_backend_family"] == "triton"
        assert details["workload_backend_family"] == "torch"

    def test_layer_granularity_rejected(self) -> None:
        spec = ModelCaseSpec(
            granularity=ProfilingGranularity.TRANSFORMER_LAYER,
            device_ids=("cpu-0",),
            dtype="fp32",
            model=ModelReference(model_id="tiny/qwen2"),
            layer_signature=TransformerLayerSignature(
                architecture_family="qwen2",
                layer_type="decoder",
                hidden_size=64,
                intermediate_size=128,
                num_attention_heads=4,
                num_kv_heads=2,
                head_dim=16,
                dtype="fp32",
                quantization=None,
            ),
        )
        with pytest.raises(ValueError, match="granularity"):
            OperatorProfiler().profile(
                ProfilingCase.for_spec("worker-test", spec),
                instrumentation=InstrumentationBundle(timer=WallClockTimer()),
                environment_fingerprint="env-test",
            )

    def test_network_case_rejected(self) -> None:
        spec = NetworkCaseSpec(
            probe_kind=ProbeKind.RTT,
            source_worker_id="worker-test",
            destination_worker_id="worker-other",
        )
        with pytest.raises(ValueError, match="model case spec"):
            OperatorProfiler().profile(
                ProfilingCase.for_spec("worker-test", spec),
                instrumentation=InstrumentationBundle(timer=WallClockTimer()),
                environment_fingerprint="env-test",
            )

    def test_environment_check_failure_propagates_typed(self) -> None:
        def reject() -> None:
            raise ProfilingError(
                ProfilingErrorCategory.INSUFFICIENT_MEMORY, "not enough free memory"
            )

        with pytest.raises(ProfilingError) as excinfo:
            OperatorProfiler(sampling_policy=_policy()).profile(
                _case(_gemm()),
                instrumentation=InstrumentationBundle(
                    timer=WallClockTimer(), environment_check=reject
                ),
                environment_fingerprint="env-test",
            )
        assert excinfo.value.category is ProfilingErrorCategory.INSUFFICIENT_MEMORY


class TestModuleSignatureNotConsumed:
    def test_operator_profiler_ignores_module_cases(self) -> None:
        """Module identities belong to P2D; the operator path never reads them."""
        spec = ModelCaseSpec(
            granularity=ProfilingGranularity.MODULE,
            device_ids=("cpu-0",),
            dtype="fp32",
            model=ModelReference(model_id="tiny/qwen2"),
            module_signature=ModuleSignature(
                kind=ModuleKind.MLP,
                architecture_family="qwen2",
                structural_parameters=(),
                dtype="fp32",
                quantization=None,
            ),
        )
        with pytest.raises(ValueError, match="does not match"):
            OperatorProfiler().profile(
                ProfilingCase.for_spec("worker-test", spec),
                instrumentation=InstrumentationBundle(timer=WallClockTimer()),
                environment_fingerprint="env-test",
            )
