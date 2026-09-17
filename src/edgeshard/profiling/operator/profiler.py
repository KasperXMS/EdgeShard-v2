"""Operator microprofiling, incremental planning, and class verification
(spec §25-§29).

``OperatorProfiler`` measures one synthetic workload per case through the
same §12 harness lifecycle as every other profiler — operator profiling
runs without any model execution (P2E DoD). Failures stay typed: an
unregistered kind, a backend-family mismatch (§27), or a signature whose
facts do not support synthetic semantics all surface as
``UNSUPPORTED_OPERATOR``, never as a skipped or fabricated measurement.

``plan_incremental_profiling`` + ``IncrementalOperatorProfiler`` implement
the §28 algorithm: deduplicated signatures are split against the ids that
already have compatible measurements (the reuse query side is a protocol
— the P2G ``ProfileStore`` answers it after filtering by performance
class/environment), and only the missing signatures are benchmarked.
Adding a new model therefore adds only its new shapes. The pure planning
helpers (``operator_case_spec``, ``plan_incremental_profiling``,
``parameters_dtype``) live in the torch-free
:mod:`edgeshard.profiling.operator.planning` module — shared with the
Master-side strategy (§47) — and are re-exported here for the Worker API.

``verification_suite`` + ``verify_performance_class`` implement §29: a
very small suite (one representative GEMM, one attention case, one
memory-sensitive norm) whose candidate means are compared against class
reference means within a configurable tolerance. No statistical
certification system — a verdict of ``compatible`` licenses profile
reuse; ``incompatible`` means the device must be separated into its own
performance class.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from edgeshard.profiling.benchmark.harness import BenchmarkHarness, InstrumentationBundle
from edgeshard.profiling.benchmark.sampling import DurationSamplingPolicy, SamplingPolicy
from edgeshard.profiling.domain.experiment import (
    ProfilingCase,
    ProfilingErrorCategory,
)
from edgeshard.profiling.domain.measurement import MeasurementRecord, TimeUnit
from edgeshard.profiling.domain.signature import (
    OperatorSignature,
    ProfilingGranularity,
    operator_signature_id,
)
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.model.execution import (
    measurement_record_from_result,
    require_model_spec,
)
from edgeshard.profiling.operator.planning import (
    IncrementalPlan,
    operator_case_spec,
    parameters_dtype,
    plan_incremental_profiling,
)
from edgeshard.profiling.operator.registry import (
    OperatorWorkloadRegistry,
    default_operator_workload_registry,
)
from edgeshard.profiling.operator.verification import (
    DEFAULT_VERIFICATION_TOLERANCE,
    PerformanceClassVerification,
    VerificationComparison,
    select_verification_signatures,
    verification_suite,
    verify_performance_class,
)
from edgeshard.profiling.operator.workloads import OperatorWorkload

logger = logging.getLogger("profiling.operator.profiler")


class _BoundOperatorWorkload:
    """Binds a §25.1 workload to one signature/device for the harness.

    The harness owns the lifecycle (prepare → warmup → reset → measure →
    cleanup); the workload protocol takes signature and device at
    ``prepare``. Synthetic workloads carry no transient state between
    runs, so ``reset`` is a no-op — exactly like the model-side prefill
    invocation (no KV cache is mutated).
    """

    def __init__(
        self, workload: OperatorWorkload, signature: OperatorSignature, device: torch.device
    ) -> None:
        self._workload = workload
        self._signature = signature
        self._device = device

    def prepare(self) -> None:
        self._workload.prepare(self._signature, self._device)

    def run_once(self) -> None:
        self._workload.run_once()

    def reset(self) -> None:
        """No transient state: each run recomputes from fixed tensors."""

    def cleanup(self) -> None:
        self._workload.cleanup()


class OperatorProfiler:
    """Benchmarks one operator signature as a synthetic workload (§25)."""

    def __init__(
        self,
        *,
        registry: OperatorWorkloadRegistry | None = None,
        harness: BenchmarkHarness | None = None,
        sampling_policy: SamplingPolicy | None = None,
    ) -> None:
        self._registry = (
            registry if registry is not None else default_operator_workload_registry()
        )
        self._harness = harness if harness is not None else BenchmarkHarness()
        self._sampling_policy = sampling_policy

    def profile(
        self,
        case: ProfilingCase,
        *,
        instrumentation: InstrumentationBundle,
        environment_fingerprint: str,
        device: torch.device | str = "cpu",
    ) -> MeasurementRecord:
        """One empirical operator measurement for ``case``.

        The record's reuse identity is the signature (via the canonical
        ``case_id`` and the ``operator_signature_id`` metadata); the
        measurement never touches a model checkpoint.
        """
        spec = require_model_spec(case, ProfilingGranularity.OPERATOR)
        assert spec.operator_signature is not None  # domain enforces for OPERATOR
        signature = spec.operator_signature
        if spec.dtype != parameters_dtype(signature.parameters):
            raise ValueError(
                f"case dtype {spec.dtype!r} contradicts the operator signature "
                f"dtype {parameters_dtype(signature.parameters)!r}"
            )
        workload = self._registry.resolve(signature.kind)()
        if workload.backend_family != signature.backend_family:
            raise ProfilingError(
                ProfilingErrorCategory.UNSUPPORTED_OPERATOR,
                f"signature was recorded on backend family "
                f"{signature.backend_family!r} but the registered workload "
                f"measures {workload.backend_family!r}; measurements are only "
                "reusable within the same logical backend primitive (§27)",
                {
                    "signature_backend_family": signature.backend_family,
                    "workload_backend_family": workload.backend_family,
                },
            )
        result = self._harness.run(
            _BoundOperatorWorkload(workload, signature, torch.device(device)),
            sampling_policy=self._sampling_policy or DurationSamplingPolicy(),
            instrumentation=instrumentation,
        )
        signature_id = operator_signature_id(signature)
        metadata = {
            "granularity": ProfilingGranularity.OPERATOR.value,
            "operator_kind": signature.kind.value,
            "backend_family": signature.backend_family,
            "operator_signature_id": signature_id,
            "dtype": spec.dtype,
            "phase": spec.phase.value,
            "timing_unit": TimeUnit.MILLISECONDS.value,
            "warmup_runs": result.warmup_runs,
            "requested_min_warmup_runs": result.requested_min_warmup_runs,
            "actual_warmup_runs": result.warmup_runs,
            "warmup_converged": result.warmup_converged,
            "measured_runs": result.measured_runs,
        }
        record = measurement_record_from_result(
            result,
            case_id=case.case_id,
            environment_fingerprint=environment_fingerprint,
            metadata=metadata,
        )
        logger.info(
            "operator %s (%s) profiled: %d measured runs",
            signature.kind.value,
            signature_id[:12],
            record.sample_count,
        )
        return record


# ---------------------------------------------------------------------------
# §28 — incremental operator profiling
# ---------------------------------------------------------------------------


@runtime_checkable
class MeasuredSignatureIndex(Protocol):
    """Reuse-query side of the profile store (§28).

    Answers: which operator signature ids already have measurements
    compatible with the target environment/performance class? The
    filtering itself (by ``DevicePerformanceClass`` and environment
    fingerprint) is a store concern — P2G implements this protocol; the
    planning logic here depends only on the id set.
    """

    def measured_signature_ids(self) -> AbstractSet[str]: ...


@dataclass(frozen=True)
class IncrementalProfilingResult:
    """Outcome of one incremental run: the plan plus the new records.

    Records are returned for the caller (P2G runner) to *append* to the
    store (§44) — reused signatures produce no new records.
    """

    plan: IncrementalPlan
    records: tuple[MeasurementRecord, ...]


class IncrementalOperatorProfiler:
    """Benchmarks only missing signatures (§28)."""

    def __init__(
        self,
        *,
        index: MeasuredSignatureIndex,
        profiler: OperatorProfiler | None = None,
    ) -> None:
        self._index = index
        self._profiler = profiler if profiler is not None else OperatorProfiler()

    def run(
        self,
        signatures: Iterable[OperatorSignature],
        *,
        worker_id: str,
        device_id: str,
        environment_fingerprint: str,
        instrumentation: InstrumentationBundle,
        device: torch.device | str = "cpu",
    ) -> IncrementalProfilingResult:
        """Plan against the index, benchmark exactly the missing set."""
        plan = plan_incremental_profiling(
            signatures, measured_ids=self._index.measured_signature_ids()
        )
        records = tuple(
            self._profiler.profile(
                ProfilingCase.for_spec(
                    worker_id, operator_case_spec(signature, device_id=device_id)
                ),
                instrumentation=instrumentation,
                environment_fingerprint=environment_fingerprint,
                device=device,
            )
            for signature in plan.missing
        )
        logger.info(
            "incremental operator profiling: %d reused, %d benchmarked",
            len(plan.reused),
            len(records),
        )
        return IncrementalProfilingResult(plan=plan, records=records)


__all__ = [
    "DEFAULT_VERIFICATION_TOLERANCE",
    "IncrementalOperatorProfiler",
    "IncrementalPlan",
    "IncrementalProfilingResult",
    "MeasuredSignatureIndex",
    "OperatorProfiler",
    "PerformanceClassVerification",
    "VerificationComparison",
    "operator_case_spec",
    "plan_incremental_profiling",
    "select_verification_signatures",
    "verification_suite",
    "verify_performance_class",
]
