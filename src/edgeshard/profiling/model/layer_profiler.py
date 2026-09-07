"""Direct TransformerLayer profiling (spec §21) and the layer-position
sanity check (§22).

``TransformerLayerProfiler`` runs the §21 behavior on a real checkpoint:
adapter-built inputs (§16.1) → ``BenchmarkHarness`` lifecycle (§12) →
one empirical ``MeasurementRecord`` (§8.3). It holds no timing or memory
logic of its own — the harness and instrumentation decide fidelity
(CUDA events on GPU, wall clock only for CPU-side runs, §11.1).

The position sanity check answers a *measurement* question, not a
modeling one: are early/middle/late layers empirically equivalent on
this device? Below the configured variation threshold a single middle
representative suffices for future default calibration; above it, the
positional classes (EARLY/MIDDLE/LATE/SPECIAL) must be preserved. Layer
position equivalence is never assumed globally (§22, §52.1).
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass

import torch.nn as nn

from edgeshard.model.layout import ModelLayout
from edgeshard.profiling.benchmark.harness import BenchmarkHarness, InstrumentationBundle
from edgeshard.profiling.benchmark.sampling import DurationSamplingPolicy, SamplingPolicy
from edgeshard.profiling.domain.experiment import ProfilingCase
from edgeshard.profiling.domain.measurement import MeasurementRecord, TimeUnit
from edgeshard.profiling.domain.model import ModelCharacterization
from edgeshard.profiling.domain.signature import (
    LayerPosition,
    ProfilingGranularity,
    TransformerLayerSignature,
)
from edgeshard.profiling.model.adapters.base import LayerReference, ModelProfilingAdapter
from edgeshard.profiling.model.execution import (
    ModuleCallWorkload,
    measurement_record_from_result,
    require_model_spec,
)
from edgeshard.profiling.model.planning import (
    DEFAULT_PREFILL_SEQUENCE_LENGTHS as DEFAULT_PREFILL_SEQUENCE_LENGTHS,
)
from edgeshard.profiling.model.planning import (
    representative_layer_positions as representative_layer_positions,
)

logger = logging.getLogger("profiling.model.layer_profiler")

STANDARD_DECODER_LAYER_TYPE = "decoder"
"""Layer type label for homogeneous standard-decoder stacks (§6.1)."""

DEFAULT_VARIATION_THRESHOLD = 0.05
"""Default coefficient-of-variation threshold of the §22 sanity check."""


def transformer_layer_signature(
    characterization: ModelCharacterization,
    *,
    layer_type: str = STANDARD_DECODER_LAYER_TYPE,
    special_role: str | None = None,
) -> TransformerLayerSignature:
    """Reusable layer signature of a characterized standard decoder (§6.1).

    Derived from the static characterization only — no measurement is
    involved, so the signature is stable across devices and runs.
    """
    return TransformerLayerSignature(
        architecture_family=characterization.architecture_family,
        layer_type=layer_type,
        hidden_size=characterization.hidden_size,
        intermediate_size=characterization.intermediate_size,
        num_attention_heads=characterization.num_attention_heads,
        num_kv_heads=characterization.num_kv_heads,
        head_dim=characterization.head_dim,
        dtype=characterization.dtype,
        quantization=characterization.quantization,
        special_role=special_role,
    )


class TransformerLayerProfiler:
    """Benchmarks one real transformer layer of a loaded checkpoint (§21)."""

    def __init__(
        self,
        *,
        harness: BenchmarkHarness | None = None,
        sampling_policy: SamplingPolicy | None = None,
    ) -> None:
        self._harness = harness if harness is not None else BenchmarkHarness()
        self._sampling_policy = sampling_policy

    def profile(
        self,
        case: ProfilingCase,
        model: nn.Module,
        layer: LayerReference,
        layout: ModelLayout,
        adapter: ModelProfilingAdapter,
        *,
        instrumentation: InstrumentationBundle,
        environment_fingerprint: str,
        seed: int | None = None,
    ) -> MeasurementRecord:
        """One empirical layer measurement for ``case``.

        Failures stay typed end to end: unsupported decode
        (``UNSUPPORTED_PHASE``, §24), structural or dtype mismatches
        (``UNSUPPORTED_MODEL``), and any benchmark failure
        (``BENCHMARK_FAILED``, §42) — never a zero-latency record.
        """
        spec = require_model_spec(case, ProfilingGranularity.TRANSFORMER_LAYER)
        inputs = adapter.build_layer_inputs(spec, model, layer, layout, seed=seed)
        result = self._harness.run(
            ModuleCallWorkload(layer.layer, inputs),
            sampling_policy=self._sampling_policy or DurationSamplingPolicy(),
            instrumentation=instrumentation,
        )
        assert spec.model is not None  # domain enforces this for layer cases
        metadata = {
            "granularity": ProfilingGranularity.TRANSFORMER_LAYER.value,
            "phase": spec.phase.value,
            "batch_size": spec.batch_size,
            "sequence_length": spec.sequence_length,
            "layer_index": layer.index,
            "module_path": layer.module_path,
            "model_id": spec.model.model_id,
            "model_revision": spec.model.revision,
            "timing_unit": TimeUnit.MILLISECONDS.value,
            "warmup_runs": result.warmup_runs,
            "measured_runs": result.measured_runs,
        }
        record = measurement_record_from_result(
            result,
            case_id=case.case_id,
            environment_fingerprint=environment_fingerprint,
            metadata=metadata,
        )
        logger.info(
            "layer %s profiled: %d measured runs (%d warmups)",
            layer.module_path,
            record.sample_count,
            result.warmup_runs,
        )
        return record


@dataclass(frozen=True)
class LayerPositionObservation:
    """One measured representative layer position (§22)."""

    position: LayerPosition
    layer_index: int
    mean_latency_ms: float

    def __post_init__(self) -> None:
        if self.layer_index < 0:
            raise ValueError(f"layer_index must not be negative, got {self.layer_index}")
        if not math.isfinite(self.mean_latency_ms) or self.mean_latency_ms <= 0.0:
            raise ValueError(
                f"mean_latency_ms must be positive and finite, got {self.mean_latency_ms}"
            )

    @classmethod
    def from_record(
        cls, position: LayerPosition, layer_index: int, record: MeasurementRecord
    ) -> LayerPositionObservation:
        """Observation from a profiler record's latency mean (facts only)."""
        latency = record.metrics.latency
        if latency is None:
            raise ValueError(f"record {record.measurement_id} carries no latency observation")
        return cls(
            position=position,
            layer_index=layer_index,
            mean_latency_ms=latency.summary.mean,
        )


@dataclass(frozen=True)
class LayerPositionCheck:
    """Outcome of the early/middle/late sanity check (§22).

    ``homogeneous`` means the coefficient of variation stayed within
    ``variation_threshold``: future default calibration may then use a
    single middle representative. Otherwise the positional classes must
    be preserved — equivalence is a property of this measurement, never
    a global assumption.
    """

    observations: tuple[LayerPositionObservation, ...]
    coefficient_of_variation: float
    max_relative_deviation: float
    variation_threshold: float
    homogeneous: bool


def layer_position_check(
    observations: tuple[LayerPositionObservation, ...] | list[LayerPositionObservation],
    *,
    variation_threshold: float = DEFAULT_VARIATION_THRESHOLD,
) -> LayerPositionCheck:
    """Simple variation diagnostics over representative positions (§22)."""
    if not observations:
        raise ValueError("layer position check requires at least one observation")
    if variation_threshold <= 0.0:
        raise ValueError(f"variation_threshold must be positive, got {variation_threshold}")
    means = [observation.mean_latency_ms for observation in observations]
    center = statistics.fmean(means)
    stddev = statistics.stdev(means) if len(means) >= 2 else 0.0
    coefficient_of_variation = stddev / center
    max_relative_deviation = max(abs(mean - center) for mean in means) / center
    return LayerPositionCheck(
        observations=tuple(observations),
        coefficient_of_variation=coefficient_of_variation,
        max_relative_deviation=max_relative_deviation,
        variation_threshold=variation_threshold,
        homogeneous=coefficient_of_variation <= variation_threshold,
    )
