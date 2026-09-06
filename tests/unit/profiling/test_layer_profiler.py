"""P2D TransformerLayerProfiler tests (spec §21, §22, §50 P2D DoD).

The profiler is exercised end to end on a real tiny checkpoint with CPU
wall-clock timing (the CUDA-event path is P2B-tested and hardware-gated):
record shape, metadata, run counts, and typed failures. The §22 position
sanity check is pure computation over observed means and is pinned
exactly: coefficient of variation, maximum relative deviation,
threshold behavior, and representative position selection.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import torch.nn as nn

from edgeshard.model.layout import ModelLayout
from edgeshard.profiling.benchmark.harness import InstrumentationBundle
from edgeshard.profiling.benchmark.sampling import DurationSamplingPolicy
from edgeshard.profiling.domain.experiment import (
    ModelCaseSpec,
    NetworkCaseSpec,
    ProfilingCase,
    ProfilingErrorCategory,
)
from edgeshard.profiling.domain.measurement import (
    LatencyMetrics,
    MeasurementMetrics,
    MeasurementRecord,
    SampleSummary,
    TelemetryContextMetrics,
    TimeUnit,
)
from edgeshard.profiling.domain.model import ModelCharacterization, ModelReference
from edgeshard.profiling.domain.network import ProbeKind
from edgeshard.profiling.domain.signature import (
    InferencePhase,
    LayerPosition,
    ModuleSignature,
    ProfilingGranularity,
    TransformerLayerSignature,
)
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.instrumentation.timing import WallClockTimer
from edgeshard.profiling.model.adapters.base import LayerReference
from edgeshard.profiling.model.adapters.qwen import QwenProfilingAdapter
from edgeshard.profiling.model.layer_profiler import (
    DEFAULT_VARIATION_THRESHOLD,
    LayerPositionObservation,
    TransformerLayerProfiler,
    layer_position_check,
    representative_layer_positions,
    transformer_layer_signature,
)

MODEL = ModelReference(model_id="tiny/qwen2")


def _policy() -> DurationSamplingPolicy:
    return DurationSamplingPolicy(
        min_warmups=1, min_runs=3, max_runs=3, target_duration_ms=1_000_000.0
    )


def _layer_case(
    signature: TransformerLayerSignature,
    *,
    dtype: str = "fp32",
    phase: InferencePhase = InferencePhase.PREFILL,
    context_length: int | None = None,
) -> ProfilingCase:
    spec = ModelCaseSpec(
        granularity=ProfilingGranularity.TRANSFORMER_LAYER,
        device_ids=("cpu-0",),
        dtype=dtype,
        model=MODEL,
        layer_signature=signature,
        phase=phase,
        batch_size=1,
        sequence_length=8,
        context_length=context_length,
    )
    return ProfilingCase.for_spec("worker-test", spec)


@pytest.fixture
def signature(qwen2_characterization: ModelCharacterization) -> TransformerLayerSignature:
    return transformer_layer_signature(qwen2_characterization)


class TestTransformerLayerSignature:
    def test_derived_from_characterization(
        self, qwen2_characterization: ModelCharacterization
    ) -> None:
        signature = transformer_layer_signature(qwen2_characterization)
        assert signature == TransformerLayerSignature(
            architecture_family="qwen2",
            layer_type="decoder",
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_kv_heads=2,
            head_dim=16,
            dtype="fp32",
            quantization=None,
        )

    def test_special_role_passthrough(
        self, qwen2_characterization: ModelCharacterization
    ) -> None:
        signature = transformer_layer_signature(
            qwen2_characterization, layer_type="mamba_block", special_role="first"
        )
        assert signature.layer_type == "mamba_block"
        assert signature.special_role == "first"


class TestRepresentativeLayerPositions:
    @pytest.mark.parametrize(
        ("num_layers", "expected"),
        [
            (32, ((LayerPosition.EARLY, 0), (LayerPosition.MIDDLE, 16), (LayerPosition.LATE, 31))),
            (4, ((LayerPosition.EARLY, 0), (LayerPosition.MIDDLE, 2), (LayerPosition.LATE, 3))),
            (3, ((LayerPosition.EARLY, 0), (LayerPosition.MIDDLE, 1), (LayerPosition.LATE, 2))),
            (2, ((LayerPosition.EARLY, 0), (LayerPosition.LATE, 1))),
            (1, ((LayerPosition.EARLY, 0),)),
        ],
    )
    def test_positions(
        self, num_layers: int, expected: tuple[tuple[LayerPosition, int], ...]
    ) -> None:
        assert representative_layer_positions(num_layers) == expected

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            representative_layer_positions(0)


def _observation(position: LayerPosition, index: int, mean: float) -> LayerPositionObservation:
    return LayerPositionObservation(position=position, layer_index=index, mean_latency_ms=mean)


class TestLayerPositionCheck:
    def test_homogeneous_stack(self) -> None:
        check = layer_position_check(
            (
                _observation(LayerPosition.EARLY, 0, 10.0),
                _observation(LayerPosition.MIDDLE, 2, 10.1),
                _observation(LayerPosition.LATE, 3, 9.9),
            )
        )
        assert check.coefficient_of_variation == pytest.approx(0.01, abs=1e-9)
        assert check.max_relative_deviation == pytest.approx(0.01, abs=1e-9)
        assert check.variation_threshold == DEFAULT_VARIATION_THRESHOLD
        assert check.homogeneous is True

    def test_heterogeneous_stack_preserves_classes(self) -> None:
        observations = (
            _observation(LayerPosition.EARLY, 0, 10.0),
            _observation(LayerPosition.MIDDLE, 16, 10.0),
            _observation(LayerPosition.LATE, 31, 20.0),
        )
        check = layer_position_check(observations)
        assert check.coefficient_of_variation == pytest.approx(0.4330127, abs=1e-6)
        assert check.max_relative_deviation == pytest.approx(0.5, abs=1e-9)
        assert check.homogeneous is False
        # A generous threshold flips the verdict: it is a policy input.
        assert layer_position_check(observations, variation_threshold=0.5).homogeneous is True

    def test_single_observation(self) -> None:
        check = layer_position_check((_observation(LayerPosition.EARLY, 0, 5.0),))
        assert check.coefficient_of_variation == 0.0
        assert check.max_relative_deviation == 0.0
        assert check.homogeneous is True

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            layer_position_check(())
        with pytest.raises(ValueError, match="positive"):
            layer_position_check(
                (_observation(LayerPosition.EARLY, 0, 5.0),), variation_threshold=0.0
            )
        with pytest.raises(ValueError, match="positive"):
            _observation(LayerPosition.EARLY, 0, 0.0)
        with pytest.raises(ValueError, match="negative"):
            _observation(LayerPosition.EARLY, -1, 5.0)


class TestLayerPositionObservationFromRecord:
    @staticmethod
    def _record(summary_mean: float | None) -> MeasurementRecord:
        now = datetime.now(UTC)
        latency = (
            LatencyMetrics(
                summary=SampleSummary(
                    mean=summary_mean,
                    median=summary_mean,
                    stddev=0.0,
                    minimum=summary_mean,
                    maximum=summary_mean,
                    p95=None,
                ),
                unit=TimeUnit.MILLISECONDS,
            )
            if summary_mean is not None
            else None
        )
        metrics = MeasurementMetrics(
            latency=latency,
            telemetry=None if latency is not None else TelemetryContextMetrics(),
        )
        return MeasurementRecord(
            measurement_id="m1",
            case_id="c1",
            environment_fingerprint="e1",
            started_at=now,
            finished_at=now,
            sample_count=1,
            samples=(summary_mean,) if summary_mean is not None else None,
            metrics=metrics,
        )

    def test_from_latency_record(self) -> None:
        observation = LayerPositionObservation.from_record(
            LayerPosition.MIDDLE, 2, self._record(7.5)
        )
        assert observation == _observation(LayerPosition.MIDDLE, 2, 7.5)

    def test_record_without_latency_rejected(self) -> None:
        with pytest.raises(ValueError, match="no latency"):
            LayerPositionObservation.from_record(LayerPosition.MIDDLE, 2, self._record(None))


class TestTransformerLayerProfiler:
    def test_end_to_end_record(
        self,
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
        signature: TransformerLayerSignature,
    ) -> None:
        adapter = QwenProfilingAdapter()
        layers = adapter.enumerate_transformer_layers(tiny_qwen2_checkpoint, tiny_qwen2_layout)
        case = _layer_case(signature)
        record = TransformerLayerProfiler(sampling_policy=_policy()).profile(
            case,
            tiny_qwen2_checkpoint,
            layers[1],
            tiny_qwen2_layout,
            adapter,
            instrumentation=InstrumentationBundle(timer=WallClockTimer()),
            environment_fingerprint="env-test",
            seed=42,
        )
        assert record.case_id == case.case_id
        assert record.environment_fingerprint == "env-test"
        assert record.measurement_id
        assert record.sample_count == 3
        assert record.samples is not None and len(record.samples) == 3
        latency = record.metrics.latency
        assert latency is not None
        assert latency.unit is TimeUnit.MILLISECONDS
        assert latency.summary.mean > 0.0
        # CPU-only host: no allocator/physical/telemetry instruments were
        # attached, so those observations are absent — never guessed.
        assert record.metrics.allocator_memory is None
        assert record.metrics.physical_memory is None
        assert record.metrics.telemetry is None
        assert record.finished_at >= record.started_at
        metadata = record.metadata_mapping
        assert metadata["granularity"] == "transformer_layer"
        assert metadata["phase"] == "prefill"
        assert metadata["layer_index"] == 1
        assert metadata["module_path"] == "model.layers.1"
        assert metadata["model_id"] == "tiny/qwen2"
        assert metadata["model_revision"] is None
        assert metadata["batch_size"] == 1
        assert metadata["sequence_length"] == 8
        assert metadata["warmup_runs"] == 1
        assert metadata["measured_runs"] == 3
        assert metadata["timing_unit"] == "ms"

    def test_repeated_runs_produce_distinct_record_ids(
        self,
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
        signature: TransformerLayerSignature,
    ) -> None:
        adapter = QwenProfilingAdapter()
        layer = adapter.enumerate_transformer_layers(tiny_qwen2_checkpoint, tiny_qwen2_layout)[0]
        profiler = TransformerLayerProfiler(sampling_policy=_policy())
        case = _layer_case(signature)

        def run() -> MeasurementRecord:
            return profiler.profile(
                case,
                tiny_qwen2_checkpoint,
                layer,
                tiny_qwen2_layout,
                adapter,
                instrumentation=InstrumentationBundle(timer=WallClockTimer()),
                environment_fingerprint="env-test",
            )

        first, second = run(), run()
        assert first.measurement_id != second.measurement_id
        assert first.case_id == second.case_id  # same request, same canonical id

    def test_wrong_granularity_rejected(
        self,
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
    ) -> None:
        adapter = QwenProfilingAdapter()
        layer = adapter.enumerate_transformer_layers(tiny_qwen2_checkpoint, tiny_qwen2_layout)[0]
        module_signature = ModuleSignature(
            kind=adapter.module_kinds["mlp"],
            architecture_family="qwen2",
            structural_parameters=(),
            dtype="fp32",
            quantization=None,
        )
        spec = ModelCaseSpec(
            granularity=ProfilingGranularity.MODULE,
            device_ids=("cpu-0",),
            dtype="fp32",
            model=MODEL,
            module_signature=module_signature,
        )
        with pytest.raises(ValueError, match="granularity"):
            TransformerLayerProfiler().profile(
                ProfilingCase.for_spec("worker-test", spec),
                tiny_qwen2_checkpoint,
                layer,
                tiny_qwen2_layout,
                adapter,
                instrumentation=InstrumentationBundle(timer=WallClockTimer()),
                environment_fingerprint="env-test",
            )

    def test_network_case_rejected(
        self,
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
    ) -> None:
        adapter = QwenProfilingAdapter()
        layer = adapter.enumerate_transformer_layers(tiny_qwen2_checkpoint, tiny_qwen2_layout)[0]
        spec = NetworkCaseSpec(
            probe_kind=ProbeKind.RTT,
            source_worker_id="worker-test",
            destination_worker_id="worker-other",
        )
        with pytest.raises(ValueError, match="model case spec"):
            TransformerLayerProfiler().profile(
                ProfilingCase.for_spec("worker-test", spec),
                tiny_qwen2_checkpoint,
                layer,
                tiny_qwen2_layout,
                adapter,
                instrumentation=InstrumentationBundle(timer=WallClockTimer()),
                environment_fingerprint="env-test",
            )

    def test_decode_fails_typed(
        self,
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
        signature: TransformerLayerSignature,
    ) -> None:
        adapter = QwenProfilingAdapter()
        layer = adapter.enumerate_transformer_layers(tiny_qwen2_checkpoint, tiny_qwen2_layout)[0]
        case = _layer_case(signature, phase=InferencePhase.DECODE, context_length=8)
        with pytest.raises(ProfilingError) as excinfo:
            TransformerLayerProfiler().profile(
                case,
                tiny_qwen2_checkpoint,
                layer,
                tiny_qwen2_layout,
                adapter,
                instrumentation=InstrumentationBundle(timer=WallClockTimer()),
                environment_fingerprint="env-test",
            )
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_PHASE

    def test_probe_failure_propagates_typed(
        self,
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
        signature: TransformerLayerSignature,
    ) -> None:
        """Environment-check failures stay typed through the profiler (§12)."""
        adapter = QwenProfilingAdapter()
        layer = adapter.enumerate_transformer_layers(tiny_qwen2_checkpoint, tiny_qwen2_layout)[0]

        def reject() -> None:
            raise ProfilingError(
                ProfilingErrorCategory.INSUFFICIENT_MEMORY, "not enough free memory"
            )

        with pytest.raises(ProfilingError) as excinfo:
            TransformerLayerProfiler(sampling_policy=_policy()).profile(
                _layer_case(signature),
                tiny_qwen2_checkpoint,
                layer,
                tiny_qwen2_layout,
                adapter,
                instrumentation=InstrumentationBundle(
                    timer=WallClockTimer(), environment_check=reject
                ),
                environment_fingerprint="env-test",
            )
        assert excinfo.value.category is ProfilingErrorCategory.INSUFFICIENT_MEMORY


class TestLayerReferenceUnchanged:
    def test_enumeration_still_matches_profiler_input(
        self, tiny_qwen2_checkpoint: nn.Module, tiny_qwen2_layout: ModelLayout
    ) -> None:
        layers: tuple[LayerReference, ...] = QwenProfilingAdapter().enumerate_transformer_layers(
            tiny_qwen2_checkpoint, tiny_qwen2_layout
        )
        assert len(layers) == 4
        assert layers[3].layer is tiny_qwen2_checkpoint.model.layers[3]
