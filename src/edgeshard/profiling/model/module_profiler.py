"""Direct Module profiling (spec §23).

``ModuleProfiler`` benchmarks one normalized profile module (Attention,
MLP; optionally Norm/Projection/Embedding/LM Head) of a real checkpoint
through direct invocation and the shared ``BenchmarkHarness`` lifecycle
— the module granularity is an independent empirical fidelity and never
implied to sum with layers or operators (§5.1).

Nested forward-hook timing is explicitly not the production path;
``torch.profiler`` remains available for *diagnostics* of a module's
internal composition (§23) via the P2C operator extractors.
"""

from __future__ import annotations

import logging

import torch.nn as nn

from edgeshard.model.layout import ModelLayout
from edgeshard.profiling.benchmark.harness import BenchmarkHarness, InstrumentationBundle
from edgeshard.profiling.benchmark.sampling import DurationSamplingPolicy, SamplingPolicy
from edgeshard.profiling.domain.experiment import ProfilingCase
from edgeshard.profiling.domain.measurement import MeasurementRecord, TimeUnit
from edgeshard.profiling.domain.signature import ProfilingGranularity
from edgeshard.profiling.model.adapters.base import ModelProfilingAdapter, ProfileModule
from edgeshard.profiling.model.execution import (
    ModuleCallWorkload,
    measurement_record_from_result,
    require_model_spec,
)

logger = logging.getLogger("profiling.model.module_profiler")


class ModuleProfiler:
    """Benchmarks one real module of a loaded checkpoint (§23)."""

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
        module: ProfileModule,
        layout: ModelLayout,
        adapter: ModelProfilingAdapter,
        *,
        instrumentation: InstrumentationBundle,
        environment_fingerprint: str,
        seed: int | None = None,
    ) -> MeasurementRecord:
        """One empirical module measurement for ``case``.

        The case must be a ``MODULE``-granularity model case; input
        building and every benchmark step keep their typed failures
        (§24, §42) — a failed module never yields a zero-latency record.
        """
        spec = require_model_spec(case, ProfilingGranularity.MODULE)
        inputs = adapter.build_module_inputs(spec, model, module, layout, seed=seed)
        result = self._harness.run(
            ModuleCallWorkload(module.module, inputs),
            sampling_policy=self._sampling_policy or DurationSamplingPolicy(),
            instrumentation=instrumentation,
        )
        assert spec.model is not None  # domain enforces this for module cases
        metadata = {
            "granularity": ProfilingGranularity.MODULE.value,
            "phase": spec.phase.value,
            "batch_size": spec.batch_size,
            "sequence_length": spec.sequence_length,
            "module_kind": module.kind.value,
            "module_name": module.name,
            "module_path": module.module_path,
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
            "module %s profiled: %d measured runs (%d warmups)",
            module.module_path,
            record.sample_count,
            result.warmup_runs,
        )
        return record
