"""Torch-free DevicePerformanceClass verification policy (spec §29)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from edgeshard.profiling.domain.environment import EnvironmentFingerprint
from edgeshard.profiling.domain.measurement import MeasurementRecord
from edgeshard.profiling.domain.signature import (
    AttentionSignature,
    GemmSignature,
    InferencePhase,
    NormSignature,
    NormVariant,
    OperatorKind,
    OperatorSignature,
    operator_signature_id,
)

DEFAULT_VERIFICATION_TOLERANCE = 0.15
"""Default relative-mean tolerance for class-compatibility verdicts."""


def verification_suite(
    *,
    dtype: str = "bf16",
    backend_family: str = "torch",
    gemm_dim: int = 1024,
    attention_batch: int = 1,
    attention_heads: int = 16,
    attention_kv_heads: int = 4,
    attention_head_dim: int = 64,
    attention_length: int = 256,
    norm_batch: int = 1,
    norm_tokens: int = 512,
    norm_hidden: int = 4096,
) -> tuple[OperatorSignature, ...]:
    """Build the explicit synthetic §29 suite used by direct callers."""
    return (
        OperatorSignature(
            kind=OperatorKind.GEMM,
            parameters=GemmSignature(
                m=gemm_dim, n=gemm_dim, k=gemm_dim, dtype=dtype, transpose_b=True
            ),
            backend_family=backend_family,
        ),
        OperatorSignature(
            kind=OperatorKind.ATTENTION,
            parameters=AttentionSignature(
                batch_size=attention_batch,
                num_heads=attention_heads,
                num_kv_heads=attention_kv_heads,
                head_dim=attention_head_dim,
                q_len=attention_length,
                kv_len=attention_length,
                dtype=dtype,
                phase=InferencePhase.PREFILL,
            ),
            backend_family=backend_family,
        ),
        OperatorSignature(
            kind=OperatorKind.NORM,
            parameters=NormSignature(
                batch_size=norm_batch,
                sequence_length=norm_tokens,
                hidden_size=norm_hidden,
                dtype=dtype,
                variant=NormVariant.RMS,
            ),
            backend_family=backend_family,
        ),
    )


def select_verification_signatures(
    signatures: Iterable[OperatorSignature],
) -> tuple[OperatorSignature, ...]:
    """Select the small deterministic §29 sentinel suite from a model plan.

    A mixed plan contributes one representative GEMM, one attention
    workload when present, and one memory-sensitive norm when present. A
    GEMM-only plan contributes small/medium/large shapes instead of silently
    expanding to the full operator plan.
    """
    unique = {
        operator_signature_id(signature): signature for signature in signatures
    }
    ordered = tuple(unique[signature_id] for signature_id in sorted(unique))
    gemms = _ordered_by_work(ordered, OperatorKind.GEMM)
    attentions = _ordered_by_work(ordered, OperatorKind.ATTENTION)
    norms = _ordered_by_work(ordered, OperatorKind.NORM)

    if gemms and len(gemms) == len(ordered):
        positions = (0, len(gemms) // 2, len(gemms) - 1)
        return tuple(dict.fromkeys(gemms[position] for position in positions))

    selected: list[OperatorSignature] = []
    if gemms:
        selected.append(gemms[len(gemms) // 2])
    if attentions:
        selected.append(attentions[len(attentions) // 2])
    if norms:
        selected.append(norms[-1])
    return tuple(selected)


def _ordered_by_work(
    signatures: Iterable[OperatorSignature], kind: OperatorKind
) -> tuple[OperatorSignature, ...]:
    matching = (signature for signature in signatures if signature.kind is kind)
    return tuple(
        sorted(
            matching,
            key=lambda signature: (
                _work_size(signature),
                operator_signature_id(signature),
            ),
        )
    )


def _work_size(signature: OperatorSignature) -> int:
    parameters = signature.parameters
    if isinstance(parameters, GemmSignature):
        return parameters.m * parameters.n * parameters.k
    if isinstance(parameters, AttentionSignature):
        return (
            parameters.batch_size
            * parameters.num_heads
            * parameters.q_len
            * parameters.kv_len
            * parameters.head_dim
        )
    if isinstance(parameters, NormSignature):
        return (
            parameters.batch_size
            * parameters.sequence_length
            * parameters.hidden_size
        )
    return 0


@dataclass(frozen=True)
class VerificationComparison:
    """One suite signature compared between reference and candidate."""

    operator_signature_id: str
    reference_mean_ms: float
    candidate_mean_ms: float
    relative_deviation: float
    within_tolerance: bool


@dataclass(frozen=True)
class PerformanceClassVerification:
    """§29 verdict for a device claiming an existing performance class."""

    tolerance: float
    comparisons: tuple[VerificationComparison, ...]
    max_relative_deviation: float
    compatible: bool


@dataclass(frozen=True)
class PerformanceClassVerificationPlan:
    """A candidate suite paired with one verified compatible reference."""

    candidate_environment: EnvironmentFingerprint
    signatures: tuple[OperatorSignature, ...]
    reference_measurements: tuple[MeasurementRecord, ...]
    reference_worker_id: str
    reference_device_id: str
    tolerance: float

    def __post_init__(self) -> None:
        if not self.signatures:
            raise ValueError("verification plan must not be empty")
        if len(self.signatures) != len(self.reference_measurements):
            raise ValueError(
                "verification signatures and reference measurements must align"
            )
        if not self.reference_worker_id or not self.reference_device_id:
            raise ValueError("verification reference provenance must be complete")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError(
                f"verification tolerance must be finite and positive, got "
                f"{self.tolerance}"
            )

    @property
    def references(self) -> dict[str, MeasurementRecord]:
        return {
            operator_signature_id(signature): record
            for signature, record in zip(
                self.signatures, self.reference_measurements, strict=True
            )
        }


def verify_performance_class(
    references: Mapping[str, MeasurementRecord],
    candidates: Mapping[str, MeasurementRecord],
    *,
    tolerance: float = DEFAULT_VERIFICATION_TOLERANCE,
) -> PerformanceClassVerification:
    """Compare candidate-device means against verified class references."""
    if not references:
        raise ValueError("verification suite must not be empty")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"tolerance must be finite and positive, got {tolerance}")
    missing_in_candidate = sorted(set(references) - set(candidates))
    missing_in_reference = sorted(set(candidates) - set(references))
    if missing_in_candidate or missing_in_reference:
        raise ValueError(
            "reference and candidate must cover the same suite signatures; "
            f"missing in candidate: {missing_in_candidate}, "
            f"missing in reference: {missing_in_reference}"
        )
    comparisons = []
    for signature_id in sorted(references):
        reference_mean = _latency_mean_ms(references[signature_id], "reference")
        candidate_mean = _latency_mean_ms(candidates[signature_id], "candidate")
        deviation = abs(candidate_mean - reference_mean) / reference_mean
        comparisons.append(
            VerificationComparison(
                operator_signature_id=signature_id,
                reference_mean_ms=reference_mean,
                candidate_mean_ms=candidate_mean,
                relative_deviation=deviation,
                within_tolerance=deviation <= tolerance,
            )
        )
    return PerformanceClassVerification(
        tolerance=tolerance,
        comparisons=tuple(comparisons),
        max_relative_deviation=max(
            comparison.relative_deviation for comparison in comparisons
        ),
        compatible=all(comparison.within_tolerance for comparison in comparisons),
    )


def _latency_mean_ms(record: MeasurementRecord, role: str) -> float:
    latency = record.metrics.latency
    if latency is None or latency.summary.mean is None:
        raise ValueError(
            f"{role} record {record.measurement_id} carries no latency observation"
        )
    mean = latency.summary.mean
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError(f"{role} record {record.measurement_id} has a non-positive mean")
    return mean


__all__ = [
    "DEFAULT_VERIFICATION_TOLERANCE",
    "PerformanceClassVerification",
    "PerformanceClassVerificationPlan",
    "VerificationComparison",
    "select_verification_signatures",
    "verification_suite",
    "verify_performance_class",
]
