"""Direct module invocation for model-side benchmarks (spec §21, §23).

Both P2D profilers (layer and module) measure *real execution* of the
loaded checkpoint through one workload shape: direct invocation of the
target ``nn.Module`` with adapter-built inputs. Nested forward-hook
timing is explicitly not the production implementation (§23).

:func:`measurement_record_from_result` turns a harness
:class:`BenchmarkResult` into the persisted-domain
:class:`MeasurementRecord` — one mechanical mapping, so every model-side
record carries latency (always), allocator/physical memory and telemetry
(when the instruments observed them, else ``None`` — never guessed,
§52.2), and the run counts as metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import torch
import torch.nn as nn

from edgeshard.profiling.benchmark.harness import BenchmarkResult
from edgeshard.profiling.domain.experiment import (
    ModelCaseSpec,
    ProfilingCase,
    ProfilingErrorCategory,
)
from edgeshard.profiling.domain.hashing import JsonScalar
from edgeshard.profiling.domain.measurement import (
    LatencyMetrics,
    MeasurementMetrics,
    MeasurementRecord,
    TimeUnit,
)
from edgeshard.profiling.domain.signature import ProfilingGranularity
from edgeshard.profiling.errors import ProfilingError


class ModuleCallWorkload:
    """Benchmark workload invoking one module directly (§12 lifecycle).

    ``prepare`` pins the module into eval mode so measurement never
    includes training-time nondeterminism (dropout); ``run_once`` is one
    forward under ``no_grad`` — warmup and measured runs are identical
    work. There is no transient state between runs (no KV cache in v1
    prefill benchmarks), so ``reset``/``cleanup`` have nothing to clear.
    """

    def __init__(self, module: nn.Module, inputs: Mapping[str, Any]) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError(f"module must be an nn.Module, got {type(module).__name__}")
        self._module = module
        self._inputs = dict(inputs)

    def prepare(self) -> None:
        self._module.eval()

    def run_once(self) -> None:
        with torch.no_grad():
            self._module(**self._inputs)

    def reset(self) -> None:
        """No transient state: prefill calls do not mutate the module."""

    def cleanup(self) -> None:
        """Inputs are released with the workload; nothing else to free."""


def require_model_spec(case: ProfilingCase, granularity: ProfilingGranularity) -> ModelCaseSpec:
    """The case's model spec, validated for the profiler's granularity."""
    spec = case.spec
    if not isinstance(spec, ModelCaseSpec):
        raise ValueError(
            f"{granularity.value} profiling requires a model case spec, "
            f"got {type(spec).__name__}"
        )
    if spec.granularity is not granularity:
        raise ValueError(
            f"case granularity {spec.granularity.value!r} does not match this "
            f"profiler ({granularity.value!r})"
        )
    return spec


def measurement_record_from_result(
    result: BenchmarkResult,
    *,
    case_id: str,
    environment_fingerprint: str,
    metadata: Mapping[str, JsonScalar] | None = None,
) -> MeasurementRecord:
    """Persist-ready record of one completed benchmark (§8.3).

    ``measurement_id`` is fresh per observation (records are
    append-oriented, §44); reuse identity lives in the case spec, the
    signatures it carries, and the environment fingerprint — never in
    this id.
    """
    if len(result.samples_ms) != result.measured_runs:
        raise ProfilingError(
            ProfilingErrorCategory.INTERNAL_ERROR,
            f"benchmark result carries {len(result.samples_ms)} samples but "
            f"reports {result.measured_runs} measured runs",
        )
    return MeasurementRecord(
        measurement_id=uuid4().hex,
        case_id=case_id,
        environment_fingerprint=environment_fingerprint,
        started_at=result.started_at,
        finished_at=result.finished_at,
        sample_count=result.measured_runs,
        samples=result.samples_ms,
        metrics=MeasurementMetrics(
            latency=LatencyMetrics(summary=result.summary, unit=TimeUnit.MILLISECONDS),
            allocator_memory=result.allocator_memory,
            physical_memory=result.physical_memory,
            telemetry=result.telemetry,
        ),
        metadata=MeasurementRecord.normalize_metadata(dict(metadata) if metadata else None),
    )
