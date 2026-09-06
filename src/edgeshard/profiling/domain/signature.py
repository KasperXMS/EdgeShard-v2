"""Reusable profiling workload signatures (Phase 2 spec §5-6).

The three profiling granularities — ``TRANSFORMER_LAYER``, ``MODULE``,
``OPERATOR`` — are independent empirical fidelities. They MUST NOT imply
``TransformerLayer = sum(Module)`` or ``Module = sum(Operator)``;
composition belongs to Phase 3 (§5.1).

Signatures are *reusable* workload identities: they never carry physical
identifiers (``worker_id``/``device_id``) or timestamps (§6.1, §7). Two
measurements of the same signature in compatible environments describe the
same workload, which is what makes cross-model and cross-device reuse
(§1.4, §20, §28) possible.

Mapping-shaped parameters (module structural parameters, custom-operator
metadata) are stored as canonical sorted item tuples — see
``profiling.domain.hashing.normalized_items`` — so signatures stay hashable
(they are deduplicated in sets) and mapping order never leaks into
identity.

ATen operator names are extractor output, not long-term domain identities
(§5.4): unknown operators are preserved as ``CUSTOM`` with raw metadata and
never dropped silently (§19).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from edgeshard.profiling.domain.hashing import (
    JsonScalar,
    StructuralValue,
    canonical_sha256,
    check_normalized_items,
)


class ProfilingGranularity(StrEnum):
    """Independent empirical fidelity levels (spec §5.1)."""

    TRANSFORMER_LAYER = "transformer_layer"
    MODULE = "module"
    OPERATOR = "operator"


class InferencePhase(StrEnum):
    """Prefill/decode phase of a profiled workload (spec §5.2).

    Values deliberately match ``edgeshard.inference.state.InferencePhase``.
    The profiling domain must stay torch-free (P2A DoD), so the enum is
    redeclared instead of imported; implementation layers bridge the two by
    value. ``DECODE`` exists in the schema but may remain unsupported by
    specific adapters until the runtime exposes stable KV-cache semantics —
    unsupported decode must fail explicitly, never fall back to prefill
    (§24).
    """

    PREFILL = "prefill"
    DECODE = "decode"


class ModuleKind(StrEnum):
    """Normalized profiling-domain module vocabulary (spec §5.3).

    An abstraction over model structure; it must not mirror every
    ``torch.nn.Module`` class (§52.4).
    """

    ATTENTION = "attention"
    MLP = "mlp"
    NORM = "norm"
    PROJECTION = "projection"
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"
    VISION_ENCODER = "vision_encoder"
    PROJECTOR = "projector"
    OTHER = "other"


class OperatorKind(StrEnum):
    """Stable EdgeShard operator vocabulary (spec §5.4).

    Operator identity is not kernel identity (§52.5): CUDA/Triton kernel
    names never appear here. Anything the normalizer does not recognize is
    preserved as ``CUSTOM`` with raw metadata.
    """

    GEMM = "gemm"
    ATTENTION = "attention"
    NORM = "norm"
    ROTARY = "rotary"
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    EMBEDDING = "embedding"
    KV_COPY = "kv_copy"
    CUSTOM = "custom"


class LayerPosition(StrEnum):
    """Positional class of a layer inside a homogeneous stack (spec §22).

    Layer-position equivalence must not be assumed globally; the
    early/middle/late sanity check decides whether one representative layer
    suffices or positional classes must be preserved.
    """

    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"
    SPECIAL = "special"


class NormVariant(StrEnum):
    """Normalization flavor for norm workloads."""

    RMS = "rms"
    LAYER = "layer"
    OTHER = "other"


def _require_non_empty(value: str, field: str) -> None:
    if not value:
        raise ValueError(f"{field} must not be empty")


def _require_positive(value: int, field: str) -> None:
    if value <= 0:
        raise ValueError(f"{field} must be positive, got {value}")


def _require_shape(shape: tuple[int, ...], field: str) -> None:
    for dimension in shape:
        if dimension <= 0:
            raise ValueError(f"{field} dimensions must be positive, got {shape}")


@dataclass(frozen=True)
class TransformerLayerSignature:
    """Reusable identity of one Transformer layer shape (spec §6.1).

    Carries no physical ``worker_id``/``device_id`` — provenance lives on
    the measurement, reuse identity lives here.
    """

    architecture_family: str
    layer_type: str
    hidden_size: int
    intermediate_size: int | None
    num_attention_heads: int
    num_kv_heads: int | None
    head_dim: int | None
    dtype: str
    quantization: str | None
    special_role: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.architecture_family, "architecture_family")
        _require_non_empty(self.layer_type, "layer_type")
        _require_non_empty(self.dtype, "dtype")
        _require_positive(self.hidden_size, "hidden_size")
        _require_positive(self.num_attention_heads, "num_attention_heads")
        if self.intermediate_size is not None:
            _require_positive(self.intermediate_size, "intermediate_size")
        if self.num_kv_heads is not None:
            _require_positive(self.num_kv_heads, "num_kv_heads")
            if self.num_attention_heads % self.num_kv_heads != 0:
                raise ValueError(
                    f"num_attention_heads ({self.num_attention_heads}) must be "
                    f"divisible by num_kv_heads ({self.num_kv_heads})"
                )
        if self.head_dim is not None:
            _require_positive(self.head_dim, "head_dim")
        if self.quantization is not None:
            _require_non_empty(self.quantization, "quantization")
        if self.special_role is not None:
            _require_non_empty(self.special_role, "special_role")


def transformer_layer_signature_id(signature: TransformerLayerSignature) -> str:
    """Canonical SHA-256 identity of a Transformer layer signature (§7)."""
    return canonical_sha256(("transformer_layer_signature", signature))


@dataclass(frozen=True)
class ModuleSignature:
    """Reusable identity of one normalized module shape (spec §6.2).

    ``structural_parameters`` holds normalized (sorted, unique-key) items,
    e.g. ``(("head_dim", 128), ("hidden_size", 3584), ("num_heads", 28),
    ("num_kv_heads", 4))`` for an attention module. Build it from a mapping
    with ``normalized_items``; read it back via :attr:`parameters`.
    """

    kind: ModuleKind
    architecture_family: str
    structural_parameters: tuple[tuple[str, StructuralValue], ...]
    dtype: str
    quantization: str | None

    def __post_init__(self) -> None:
        _require_non_empty(self.architecture_family, "architecture_family")
        _require_non_empty(self.dtype, "dtype")
        if self.quantization is not None:
            _require_non_empty(self.quantization, "quantization")
        check_normalized_items(self.structural_parameters, "structural_parameters")

    @property
    def parameters(self) -> Mapping[str, StructuralValue]:
        """Structural parameters as a read-friendly mapping view."""
        return dict(self.structural_parameters)


def module_signature_id(signature: ModuleSignature) -> str:
    """Canonical SHA-256 identity of a module signature (§7)."""
    return canonical_sha256(("module_signature", signature))


@dataclass(frozen=True)
class GemmSignature:
    """Typed parameters of a GEMM workload (spec §6.3)."""

    m: int
    n: int
    k: int
    dtype: str
    transpose_a: bool = False
    transpose_b: bool = False

    def __post_init__(self) -> None:
        _require_positive(self.m, "m")
        _require_positive(self.n, "n")
        _require_positive(self.k, "k")
        _require_non_empty(self.dtype, "dtype")


@dataclass(frozen=True)
class AttentionSignature:
    """Typed parameters of an attention workload (spec §6.3, §27)."""

    batch_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    q_len: int
    kv_len: int
    dtype: str
    phase: InferencePhase

    def __post_init__(self) -> None:
        _require_positive(self.batch_size, "batch_size")
        _require_positive(self.num_heads, "num_heads")
        _require_positive(self.num_kv_heads, "num_kv_heads")
        _require_positive(self.head_dim, "head_dim")
        _require_positive(self.q_len, "q_len")
        _require_positive(self.kv_len, "kv_len")
        _require_non_empty(self.dtype, "dtype")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )


@dataclass(frozen=True)
class NormSignature:
    """Typed parameters of a normalization workload (spec §25)."""

    batch_size: int
    sequence_length: int
    hidden_size: int
    dtype: str
    variant: NormVariant

    def __post_init__(self) -> None:
        _require_positive(self.batch_size, "batch_size")
        _require_positive(self.sequence_length, "sequence_length")
        _require_positive(self.hidden_size, "hidden_size")
        _require_non_empty(self.dtype, "dtype")


@dataclass(frozen=True)
class EmbeddingSignature:
    """Typed parameters of an embedding lookup workload."""

    batch_size: int
    sequence_length: int
    vocab_size: int
    embedding_dim: int
    dtype: str

    def __post_init__(self) -> None:
        _require_positive(self.batch_size, "batch_size")
        _require_positive(self.sequence_length, "sequence_length")
        _require_positive(self.vocab_size, "vocab_size")
        _require_positive(self.embedding_dim, "embedding_dim")
        _require_non_empty(self.dtype, "dtype")


@dataclass(frozen=True)
class RotarySignature:
    """Typed parameters of a rotary-embedding workload."""

    batch_size: int
    sequence_length: int
    num_heads: int
    head_dim: int
    dtype: str

    def __post_init__(self) -> None:
        _require_positive(self.batch_size, "batch_size")
        _require_positive(self.sequence_length, "sequence_length")
        _require_positive(self.num_heads, "num_heads")
        _require_positive(self.head_dim, "head_dim")
        _require_non_empty(self.dtype, "dtype")


@dataclass(frozen=True)
class KvCopySignature:
    """Typed parameters of a KV-cache copy workload."""

    batch_size: int
    num_kv_heads: int
    head_dim: int
    context_length: int
    dtype: str

    def __post_init__(self) -> None:
        _require_positive(self.batch_size, "batch_size")
        _require_positive(self.num_kv_heads, "num_kv_heads")
        _require_positive(self.head_dim, "head_dim")
        _require_positive(self.context_length, "context_length")
        _require_non_empty(self.dtype, "dtype")


@dataclass(frozen=True)
class GenericOperatorParameters:
    """Shape-typed parameters for elementwise/reduction workloads.

    These kinds are not part of the initial microbenchmark set (spec §25)
    but are extracted and deduplicated from model graphs; a shape-level
    identity is enough to benchmark them later.
    """

    operation: str
    input_shapes: tuple[tuple[int, ...], ...]
    dtype: str

    def __post_init__(self) -> None:
        _require_non_empty(self.operation, "operation")
        _require_non_empty(self.dtype, "dtype")
        for shape in self.input_shapes:
            _require_shape(shape, "input_shapes")


@dataclass(frozen=True)
class CustomOperatorParameters:
    """Preserved raw identity of an operator the normalizer did not map.

    Unknown operations must never disappear silently (spec §19): they keep
    their raw extractor name, observed input shapes/dtypes, and any raw
    metadata the extractor produced.
    """

    raw_name: str
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.raw_name, "raw_name")
        for shape in self.input_shapes:
            _require_shape(shape, "input_shapes")
        for dtype in self.input_dtypes:
            _require_non_empty(dtype, "input_dtypes")
        check_normalized_items(self.metadata, "metadata")


type OperatorParameters = (
    GemmSignature
    | AttentionSignature
    | NormSignature
    | EmbeddingSignature
    | RotarySignature
    | KvCopySignature
    | GenericOperatorParameters
    | CustomOperatorParameters
)

_PARAMETERS_BY_KIND: dict[OperatorKind, tuple[type, ...]] = {
    OperatorKind.GEMM: (GemmSignature,),
    OperatorKind.ATTENTION: (AttentionSignature,),
    OperatorKind.NORM: (NormSignature,),
    OperatorKind.ROTARY: (RotarySignature,),
    OperatorKind.ELEMENTWISE: (GenericOperatorParameters,),
    OperatorKind.REDUCTION: (GenericOperatorParameters,),
    OperatorKind.EMBEDDING: (EmbeddingSignature,),
    OperatorKind.KV_COPY: (KvCopySignature,),
    OperatorKind.CUSTOM: (CustomOperatorParameters,),
}


@dataclass(frozen=True)
class OperatorSignature:
    """Reusable identity of one deduplicated operator workload (§6.3).

    ``backend_family`` keeps identities of different execution backends
    apart (e.g. ``torch`` SDPA vs a fused custom attention): operator
    identity is not kernel identity, but measurements are only reusable
    within the same logical backend primitive (§27).
    """

    kind: OperatorKind
    parameters: OperatorParameters
    backend_family: str

    def __post_init__(self) -> None:
        _require_non_empty(self.backend_family, "backend_family")
        allowed = _PARAMETERS_BY_KIND[self.kind]
        if not isinstance(self.parameters, allowed):
            expected = " or ".join(cls.__name__ for cls in allowed)
            raise ValueError(
                f"operator kind {self.kind.value!r} requires {expected} "
                f"parameters, got {type(self.parameters).__name__}"
            )


def operator_signature_id(signature: OperatorSignature) -> str:
    """Canonical SHA-256 identity of an operator signature (§7)."""
    return canonical_sha256(("operator_signature", signature))
