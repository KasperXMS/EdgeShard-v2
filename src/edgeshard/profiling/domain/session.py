"""Profiling session domain (Phase 2 spec §38, §41, §47).

A session groups compatible cases so expensive state is built once: a model
session loads a checkpoint once and runs many layer/module cases against it
(§38); an operator session groups model-free microbenchmarks on one device;
a network session groups the probes of one source worker and carries the
cluster address facts the Master resolved (the executing Worker never
guesses destinations, §52.2).

The request/facts types here are pure domain values — the wire
(``protocol.profiling``) and the Worker runner both speak them, and the
Master-side strategy plans cases from :class:`ModelSessionFacts` without
ever touching a live model (§46: planning consumes facts, not runners).

Live per-session state (loaded checkpoints, lease reservations, cleanup) is
Worker-side implementation and deliberately absent from the domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from edgeshard.profiling.domain.environment import EnvironmentFingerprint
from edgeshard.profiling.domain.hashing import canonical_sha256
from edgeshard.profiling.domain.model import ModelCharacterization, ModelReference
from edgeshard.profiling.domain.signature import (
    ModuleKind,
    ModuleSignature,
    OperatorSignature,
    TransformerLayerSignature,
)


class ProfilingSessionKind(StrEnum):
    """Which case family a session groups (spec §38)."""

    MODEL = "model"
    OPERATOR = "operator"
    NETWORK = "network"


@dataclass(frozen=True)
class ProfilingSessionRequest:
    """Domain request to prepare one profiling session (spec §38, §41).

    ``MODEL`` sessions benchmark a real checkpoint and require the model
    reference plus the declared measurement ``dtype`` (§17: dtype is a
    declared context label, never an observation). ``OPERATOR`` sessions are
    model-free — each case carries its own signature and dtype (§25).
    ``NETWORK`` sessions need no device at all; ``device_ids`` stays
    mandatory only for the device-bound kinds.
    """

    kind: ProfilingSessionKind
    device_ids: tuple[str, ...] = ()
    backend: str = "torch"
    model: ModelReference | None = None
    dtype: str | None = None
    quantization: str | None = None
    target_layer_index: int | None = None

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must not be empty")
        for device_id in self.device_ids:
            if not device_id:
                raise ValueError("device_ids must not contain empty entries")
        if self.kind is ProfilingSessionKind.MODEL:
            if len(self.device_ids) != 1:
                raise ValueError(
                    "model sessions require exactly one device_id in Phase 2 v1"
                )
            if self.model is None:
                raise ValueError("model sessions require a model reference (§38)")
            if self.dtype is None:
                raise ValueError("model sessions require the declared measurement dtype")
            if self.target_layer_index is not None and self.target_layer_index < 0:
                raise ValueError("target_layer_index must not be negative")
        else:
            if self.model is not None:
                raise ValueError(
                    f"{self.kind.value} sessions must not carry a model reference"
                )
            if self.kind is ProfilingSessionKind.OPERATOR and not self.device_ids:
                raise ValueError("operator sessions require at least one device_id")
            if self.target_layer_index is not None:
                raise ValueError(
                    "target_layer_index is only valid for model sessions"
                )
        if self.dtype is not None and not self.dtype:
            raise ValueError("dtype must not be empty when present")
        if self.quantization is not None and not self.quantization:
            raise ValueError("quantization must not be empty when present")


def profiling_session_id(
    experiment_id: str, worker_id: str, session_request: ProfilingSessionRequest
) -> str:
    """Canonical SHA-256 identity of one dispatched profiling session (§7).

    Scoped to the experiment so a session id is never reused across jobs (a
    closed id must never come back — the Worker refuses re-preparing it,
    §38), and derived from the *full* request so two sessions with different
    content (kind, devices, model, dtype) can never collide onto one id. A
    Master restart re-derives the identical id from the same experiment and
    request, and the Worker replays the prepare idempotently (§50) instead of
    loading the model a second time.
    """
    if not experiment_id:
        raise ValueError("experiment_id must not be empty")
    if not worker_id:
        raise ValueError("worker_id must not be empty")
    return canonical_sha256(
        ("profiling_session", experiment_id, worker_id, session_request)
    )


@dataclass(frozen=True)
class LayerEntry:
    """One enumerated Transformer layer as a plannable fact (spec §21-22).

    ``index`` is the enumerated position the positional check reasons about
    (early/middle/late, §22); ``signature`` is the layer's reuse identity.
    The live ``nn.Module`` stays Worker-side and never crosses this boundary.
    """

    index: int
    module_path: str
    signature: TransformerLayerSignature

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must not be negative, got {self.index}")
        if not self.module_path:
            raise ValueError("module_path must not be empty")


@dataclass(frozen=True)
class ModuleEntry:
    """One enumerated profile module as a plannable fact (spec §23)."""

    name: str
    module_path: str
    kind: ModuleKind
    layer_index: int
    signature: ModuleSignature

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.module_path:
            raise ValueError("module_path must not be empty")
        if self.layer_index < 0:
            raise ValueError(f"layer_index must not be negative, got {self.layer_index}")


@dataclass(frozen=True)
class ModelSessionFacts:
    """Static facts one prepared model session reports back (spec §47).

    Everything the Master-side strategy needs to plan the §47 model workflow
    — characterization (step 1), deduplicated operator signatures (steps
    2-3), and the layer/module enumerations (steps 5-6) — without ever
    running a benchmark on the Master (§40). The layer enumeration must
    account for exactly ``characterization.num_layers`` layers, so facts and
    characterization can never drift apart silently (§47).
    """

    characterization: ModelCharacterization
    layer_entries: tuple[LayerEntry, ...]
    module_entries: tuple[ModuleEntry, ...]
    operator_signatures: tuple[OperatorSignature, ...]
    environment: EnvironmentFingerprint | None = None

    def __post_init__(self) -> None:
        if len(self.layer_entries) != self.characterization.num_layers:
            raise ValueError(
                f"layer enumeration covers {len(self.layer_entries)} layers but "
                f"the characterization declares {self.characterization.num_layers}"
            )
        indices = [entry.index for entry in self.layer_entries]
        if len(set(indices)) != len(indices):
            raise ValueError("layer_entries indices must be unique")
