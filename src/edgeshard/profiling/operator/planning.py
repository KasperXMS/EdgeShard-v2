"""Pure operator planning helpers (spec §25, §28).

The torch-free half of the operator layer, shared by both sides of the
control plane:

* :func:`operator_case_spec` builds the §8.2 case request that benchmarks
  exactly one signature — the Master-side strategy plans with it (§47
  step 4) and the Worker-side profiler validates against it;
* :func:`parameters_dtype` reads the declared dtype label of typed
  parameters (never guessed, §52.2);
* :func:`plan_incremental_profiling` implements the §28 split of
  deduplicated signatures into reused vs missing against the id set the
  profile store already holds measurements for.

Deliberately free of torch, harnesses, and stores so a control-plane-only
deployment (no ``inference`` extra) can plan operator work; execution
lives in :mod:`edgeshard.profiling.operator.profiler`, which re-exports
these names.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from edgeshard.profiling.domain.experiment import (
    ModelCaseSpec,
    ProfilingErrorCategory,
)
from edgeshard.profiling.domain.signature import (
    AttentionSignature,
    CustomOperatorParameters,
    GemmSignature,
    InferencePhase,
    KvCopySignature,
    OperatorParameters,
    OperatorSignature,
    ProfilingGranularity,
    operator_signature_id,
)
from edgeshard.profiling.errors import ProfilingError


def parameters_dtype(parameters: OperatorParameters) -> str:
    """Declared dtype label of typed operator parameters.

    ``CUSTOM`` parameters carry raw per-input dtypes instead: custom
    operators are preserved for coverage (§19), not benchmarked in v1.
    """
    dtype = getattr(parameters, "dtype", None)
    if not isinstance(dtype, str):
        raise ProfilingError(
            ProfilingErrorCategory.UNSUPPORTED_OPERATOR,
            f"{type(parameters).__name__} carries no single workload dtype",
        )
    return dtype


def operator_case_spec(signature: OperatorSignature, *, device_id: str) -> ModelCaseSpec:
    """Case-spec request that benchmarks exactly this signature (§8.2).

    Operator cases are model-free: the signature *is* the workload
    identity, so ``model`` stays ``None``. Phase and context come from the
    signature's own facts (decode attention knows its ``kv_len``);
    batch/sequence fields mirror the signature dimensions where they are
    defined. Custom operators are preserved for coverage (§19), never
    benchmarked — requesting one fails typed.
    """
    parameters = signature.parameters
    if isinstance(parameters, CustomOperatorParameters):
        raise ProfilingError(
            ProfilingErrorCategory.UNSUPPORTED_OPERATOR,
            f"custom operator {parameters.raw_name!r} has no synthetic workload; "
            "unknown operations are preserved for coverage, not benchmarked (§19)",
            {"kind": signature.kind.value},
        )
    dtype = parameters_dtype(parameters)
    phase = InferencePhase.PREFILL
    context_length: int | None = None
    if isinstance(parameters, AttentionSignature):
        phase = parameters.phase
        if phase is InferencePhase.DECODE:
            context_length = parameters.kv_len - parameters.q_len
    return ModelCaseSpec(
        granularity=ProfilingGranularity.OPERATOR,
        device_ids=(device_id,),
        dtype=dtype,
        backend=signature.backend_family,
        operator_signature=signature,
        phase=phase,
        batch_size=_case_batch_size(parameters),
        sequence_length=_case_sequence_length(parameters),
        context_length=context_length,
    )


def _case_batch_size(parameters: OperatorParameters) -> int:
    batch_size = getattr(parameters, "batch_size", None)
    return batch_size if isinstance(batch_size, int) else 1


def _case_sequence_length(parameters: OperatorParameters) -> int:
    if isinstance(parameters, AttentionSignature):
        return parameters.q_len
    if isinstance(parameters, GemmSignature):
        return parameters.m  # flattened token count of the production call
    if isinstance(parameters, KvCopySignature):
        return parameters.context_length
    sequence_length = getattr(parameters, "sequence_length", None)
    return sequence_length if isinstance(sequence_length, int) else 512


@dataclass(frozen=True)
class IncrementalPlan:
    """Split of requested signatures into reused vs missing (§28).

    ``reused`` signatures already have compatible measurements and MUST
    NOT be re-benchmarked; ``missing`` signatures are the only ones a new
    model contributes to the workload list.
    """

    reused: tuple[OperatorSignature, ...]
    missing: tuple[OperatorSignature, ...]


def plan_incremental_profiling(
    signatures: Iterable[OperatorSignature],
    *,
    measured_ids: AbstractSet[str],
) -> IncrementalPlan:
    """Compute the missing-signature plan (§28).

    Signatures are deduplicated by canonical id preserving first-appearance
    order (the input may already be deduped by §20; dedup here keeps the
    plan correct for any caller). No signature is dropped: every input
    lands in exactly one of ``reused``/``missing``.
    """
    unique: dict[str, OperatorSignature] = {}
    for signature in signatures:
        unique.setdefault(operator_signature_id(signature), signature)
    reused = tuple(
        signature
        for signature_id, signature in unique.items()
        if signature_id in measured_ids
    )
    missing = tuple(
        signature
        for signature_id, signature in unique.items()
        if signature_id not in measured_ids
    )
    return IncrementalPlan(reused=reused, missing=missing)


__all__ = [
    "IncrementalPlan",
    "operator_case_spec",
    "parameters_dtype",
    "plan_incremental_profiling",
]
