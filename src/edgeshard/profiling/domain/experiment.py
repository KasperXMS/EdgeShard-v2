"""Profiling experiments, cases, and typed failures (spec §8, §42).

A ``ProfilingCase`` is a *request* for exactly one benchmark configuration,
never a result (§8.2). ``ProfilingExperiment`` is the immutable definition
of a logical profiling job; runtime lifecycle state lives separately on
Master-side records (``ExperimentState``/``CaseState`` mirror the Phase 1
pattern where Master-assigned bookkeeping never mixes into reported facts).

``case_id`` is a canonical hash of the assigned worker plus the case spec
(§7): identical requests deduplicate to identical ids across experiments
and restarts, which keeps re-dispatch idempotent.

Failures are typed (§42) — a failed case produces a ``ProfilingFailure``
with one of the fixed error categories, never a zero-latency, empty-sample,
or ``None`` measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from edgeshard.profiling.domain.hashing import (
    JsonScalar,
    canonical_sha256,
    check_normalized_items,
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
    InferencePhase,
    ModuleSignature,
    OperatorSignature,
    ProfilingGranularity,
    TransformerLayerSignature,
)


class ProfilingErrorCategory(StrEnum):
    """Fixed typed-failure vocabulary (spec §42)."""

    UNSUPPORTED_MODEL = "unsupported_model"
    UNSUPPORTED_GRANULARITY = "unsupported_granularity"
    UNSUPPORTED_PHASE = "unsupported_phase"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    DEVICE_BUSY = "device_busy"
    INSUFFICIENT_MEMORY = "insufficient_memory"
    EXPORT_FAILED = "export_failed"
    PROFILER_FAILED = "profiler_failed"
    BENCHMARK_FAILED = "benchmark_failed"
    NETWORK_UNREACHABLE = "network_unreachable"
    IPERF_UNAVAILABLE = "iperf_unavailable"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class ProfilingFailure:
    """Typed failure of one profiling case (spec §42)."""

    category: ProfilingErrorCategory
    message: str
    details: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        if not self.message:
            raise ValueError("failure message must not be empty")
        check_normalized_items(self.details, "details")


class ExperimentState(StrEnum):
    """Lifecycle state of a whole experiment (Master-side bookkeeping)."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class CaseState(StrEnum):
    """Lifecycle state of one profiling case (Master-side bookkeeping)."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _require_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field} must be timezone-aware")


@dataclass(frozen=True)
class ModelCaseSpec:
    """One model-side benchmark configuration request (spec §8.2).

    Exactly one signature must be present, matching ``granularity``.
    ``TRANSFORMER_LAYER``/``MODULE`` cases benchmark a real checkpoint and
    require ``model``; ``OPERATOR`` microbenchmarks are model-free — the
    model reference stays ``None`` unless the operator workload came from a
    characterized model (§25-26).

    ``sequence_length`` is the number of positions the benchmark forwards
    (prefill: the prompt length; decode: one step's ``q_len``).
    ``context_length`` is the already-cached past length and is mandatory
    for ``DECODE`` and forbidden for ``PREFILL`` (§24) — decode is never
    silently approximated by prefill.
    """

    granularity: ProfilingGranularity
    device_ids: tuple[str, ...]
    dtype: str
    backend: str = "torch"
    model: ModelReference | None = None
    layer_signature: TransformerLayerSignature | None = None
    module_signature: ModuleSignature | None = None
    operator_signature: OperatorSignature | None = None
    phase: InferencePhase = InferencePhase.PREFILL
    batch_size: int = 1
    sequence_length: int = 512
    context_length: int | None = None

    def __post_init__(self) -> None:
        if not self.device_ids:
            raise ValueError("device_ids must not be empty")
        seen: set[str] = set()
        for device_id in self.device_ids:
            if not device_id:
                raise ValueError("device_ids must not contain empty entries")
            if device_id in seen:
                raise ValueError(f"duplicate device_id {device_id!r}")
            seen.add(device_id)
        if not self.dtype:
            raise ValueError("dtype must not be empty")
        if not self.backend:
            raise ValueError("backend must not be empty")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.sequence_length <= 0:
            raise ValueError(
                f"sequence_length must be positive, got {self.sequence_length}"
            )
        self._check_phase()
        self._check_granularity()

    def _check_phase(self) -> None:
        if self.phase is InferencePhase.DECODE:
            if self.context_length is None:
                raise ValueError("decode cases require context_length (§24)")
            if self.context_length < 1:
                raise ValueError(
                    f"decode context_length must be positive, got {self.context_length}"
                )
        elif self.context_length is not None:
            raise ValueError("prefill cases must not carry context_length (§24)")

    def _check_granularity(self) -> None:
        present = {
            ProfilingGranularity.TRANSFORMER_LAYER: self.layer_signature,
            ProfilingGranularity.MODULE: self.module_signature,
            ProfilingGranularity.OPERATOR: self.operator_signature,
        }
        if present[self.granularity] is None:
            raise ValueError(
                f"granularity {self.granularity.value!r} requires its matching "
                f"signature to be present"
            )
        mismatched = [
            granularity.value
            for granularity, signature in present.items()
            if granularity is not self.granularity and signature is not None
        ]
        if mismatched:
            raise ValueError(
                f"signatures of other granularities must be absent, got "
                f"{', '.join(sorted(mismatched))}"
            )
        if self.granularity is not ProfilingGranularity.OPERATOR and self.model is None:
            raise ValueError(
                f"granularity {self.granularity.value!r} benchmarks a real "
                f"checkpoint and requires a model reference (§1.3)"
            )


@dataclass(frozen=True)
class NetworkCaseSpec:
    """One network benchmark configuration request (spec §8.2, §33-36).

    RTT cases are ICMP ping probes: transport/direction/duration/payload do
    not apply and must stay ``None``; ``packet_count`` bounds the probe
    (default policy is 5-10 packets per pair, §33 — a policy, not a domain
    constant). Bandwidth cases are iperf3-style single-flow baselines and
    require transport, direction, and duration (§34).

    ``payload_bytes`` should be workload-relevant (derived from model
    characterization, §36), never an exhaustive sweep.
    """

    probe_kind: ProbeKind
    source_worker_id: str
    destination_worker_id: str
    source_interface_id: str | None = None
    destination_interface_id: str | None = None
    path_class: NetworkPathClass | None = None
    transport: NetworkTransport | None = None
    direction: NetworkDirection | None = None
    duration_s: float | None = None
    packet_count: int | None = None
    payload_bytes: int | None = None
    regime: NetworkMeasurementRegime = NetworkMeasurementRegime.IDLE_SINGLE_FLOW

    def __post_init__(self) -> None:
        if not self.source_worker_id:
            raise ValueError("source_worker_id must not be empty")
        if not self.destination_worker_id:
            raise ValueError("destination_worker_id must not be empty")
        if self.source_interface_id is not None and not self.source_interface_id:
            raise ValueError("source_interface_id must not be empty when present")
        if (
            self.destination_interface_id is not None
            and not self.destination_interface_id
        ):
            raise ValueError("destination_interface_id must not be empty when present")
        if (
            self.source_worker_id == self.destination_worker_id
            and self.source_interface_id == self.destination_interface_id
        ):
            raise ValueError("a case must not probe an endpoint against itself")
        if self.packet_count is not None and self.packet_count <= 0:
            raise ValueError(f"packet_count must be positive, got {self.packet_count}")
        if self.payload_bytes is not None and self.payload_bytes <= 0:
            raise ValueError(f"payload_bytes must be positive, got {self.payload_bytes}")
        if self.probe_kind is ProbeKind.RTT:
            for field_name in ("transport", "direction", "duration_s", "payload_bytes"):
                if getattr(self, field_name) is not None:
                    raise ValueError(
                        f"rtt cases must not set {field_name} (§33: ICMP probes)"
                    )
        elif self.probe_kind is ProbeKind.BANDWIDTH:
            if self.transport is None:
                raise ValueError("bandwidth cases require transport (§34)")
            if self.direction is None:
                raise ValueError("bandwidth cases require direction (§34)")
            if self.duration_s is None:
                raise ValueError("bandwidth cases require duration_s (§34)")
            if self.duration_s <= 0:
                raise ValueError(f"duration_s must be positive, got {self.duration_s}")
            if self.packet_count is not None:
                raise ValueError("bandwidth cases must not set packet_count")


CaseSpec = ModelCaseSpec | NetworkCaseSpec


def profiling_case_id(worker_id: str, spec: CaseSpec) -> str:
    """Canonical SHA-256 identity of a case request (spec §7).

    Hashes the assigned executor plus the full spec, so identical requests
    deduplicate across experiments and Master restarts.
    """
    if not worker_id:
        raise ValueError("worker_id must not be empty")
    return canonical_sha256(("profiling_case", worker_id, spec))


@dataclass(frozen=True)
class ProfilingCase:
    """One dispatched benchmark request (spec §8.2).

    ``worker_id`` is the assigned executor: the target worker for model
    cases, the source worker for network cases (probes run where the flow
    originates). Construct via :meth:`for_spec` so ``case_id`` is always the
    canonical hash.
    """

    case_id: str
    worker_id: str
    spec: CaseSpec

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if not self.worker_id:
            raise ValueError("worker_id must not be empty")
        if (
            isinstance(self.spec, NetworkCaseSpec)
            and self.worker_id != self.spec.source_worker_id
        ):
            raise ValueError(
                f"network cases execute on the source worker "
                f"({self.spec.source_worker_id!r}), got {self.worker_id!r}"
            )

    @classmethod
    def for_spec(cls, worker_id: str, spec: CaseSpec) -> ProfilingCase:
        """Build a case with its canonical id precomputed (§7)."""
        return cls(case_id=profiling_case_id(worker_id, spec), worker_id=worker_id, spec=spec)


@dataclass(frozen=True)
class ProfilingExperiment:
    """Immutable definition of a logical profiling job (spec §8.1).

    The definition never mutates: lifecycle state is tracked separately by
    the Master controller against ``experiment_id`` (see
    ``ExperimentState``), matching the Phase 1 split between reported facts
    and Master-side bookkeeping.
    """

    experiment_id: str
    strategy_id: str
    created_at: datetime
    requested_by: str | None
    case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if not self.strategy_id:
            raise ValueError("strategy_id must not be empty")
        _require_aware(self.created_at, "created_at")
        if self.requested_by is not None and not self.requested_by:
            raise ValueError("requested_by must not be empty when present")
        seen: set[str] = set()
        for case_id in self.case_ids:
            if not case_id:
                raise ValueError("case_ids must not contain empty entries")
            if case_id in seen:
                raise ValueError(f"duplicate case_id {case_id!r}")
            seen.add(case_id)
