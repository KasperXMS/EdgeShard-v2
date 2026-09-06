"""Operator normalization and signature deduplication (spec §19-§20).

The normalizer maps raw extractor output onto the stable EdgeShard
operator vocabulary:

- the mapping is an explicit, testable table — no model names, no
  heuristics on shapes alone;
- unknown operations are preserved as ``CUSTOM`` with their raw name,
  shapes, and dtypes: no operation may disappear silently (§19);
- recognized operations whose typed parameters cannot be derived from
  the recorded facts (missing shapes, missing phase context, rank
  mismatches) also degrade to ``CUSTOM`` — facts are never guessed to
  fit a typed signature (§52.2, and §24 for the attention phase);
- the normalized graph always holds exactly one operator per raw
  occurrence, in the same order.

``unique_operator_signatures`` then deduplicates across graphs (§20):
only unique signatures reach microprofiling, which is what keeps
model/device profiling from becoming a Cartesian product.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from edgeshard.profiling.domain.signature import (
    AttentionSignature,
    CustomOperatorParameters,
    EmbeddingSignature,
    GemmSignature,
    GenericOperatorParameters,
    InferencePhase,
    NormSignature,
    NormVariant,
    OperatorKind,
    OperatorParameters,
    OperatorSignature,
)
from edgeshard.profiling.dtypes import UNKNOWN_DTYPE
from edgeshard.profiling.operator.extractor import RawOperatorGraph, RawOperatorOccurrence

GEMM_OPERATIONS = frozenset({"mm", "addmm", "linear", "matmul"})
ATTENTION_OPERATIONS = frozenset(
    {
        "scaled_dot_product_attention",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_cudnn_attention",
        "_scaled_dot_product_efficient_attention",
    }
)
NORM_OPERATIONS: dict[str, NormVariant] = {
    "rms_norm": NormVariant.RMS,
    "layer_norm": NormVariant.LAYER,
    "native_layer_norm": NormVariant.LAYER,
}
EMBEDDING_OPERATIONS = frozenset({"embedding"})
ELEMENTWISE_OPERATIONS = frozenset(
    {
        "add",
        "sub",
        "rsub",
        "mul",
        "div",
        "neg",
        "abs",
        "clamp",
        "exp",
        "log",
        "pow",
        "rsqrt",
        "sqrt",
        "reciprocal",
        "gelu",
        "relu",
        "sigmoid",
        "silu",
        "tanh",
    }
)
REDUCTION_OPERATIONS = frozenset({"mean", "sum", "amax", "amin", "prod"})


def canonical_operation_name(raw_operation: str) -> str:
    """Strip extractor naming to the bare aten operation name.

    ``aten.mm.default`` (export) and ``aten::mm`` (profiler) both reduce
    to ``mm``; overload suffixes (``pow.Tensor_Scalar``) are dropped.
    """
    name = raw_operation
    for prefix in ("aten::", "aten."):
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    return name.split(".", 1)[0]


@dataclass(frozen=True)
class NormalizedOperator:
    """One raw occurrence mapped into the stable vocabulary (§19)."""

    order: int
    raw_operation: str
    kind: OperatorKind
    signature: OperatorSignature


@dataclass(frozen=True)
class NormalizedOperatorGraph:
    """Normalized view of a raw graph; one operator per raw occurrence."""

    extractor: str
    operators: tuple[NormalizedOperator, ...]


class OperatorNormalizer:
    """Explicit raw-name → OperatorKind/parameters mapping (spec §19)."""

    def __init__(self, *, backend_family: str = "torch") -> None:
        if not backend_family:
            raise ValueError("backend_family must not be empty")
        self._backend_family = backend_family

    def normalize(
        self, raw_graph: RawOperatorGraph, *, phase: InferencePhase | None = None
    ) -> NormalizedOperatorGraph:
        """Normalize every occurrence; none may disappear (§19).

        ``phase`` is the workload context for attention signatures; when
        absent, attention occurrences degrade to ``CUSTOM`` rather than
        guessing prefill vs decode (§24).
        """
        operators = tuple(
            self._normalize_one(occurrence, phase=phase)
            for occurrence in raw_graph.operations
        )
        return NormalizedOperatorGraph(extractor=raw_graph.extractor, operators=operators)

    def _normalize_one(
        self, occurrence: RawOperatorOccurrence, *, phase: InferencePhase | None
    ) -> NormalizedOperator:
        kind, parameters = self._classify(occurrence, phase=phase)
        signature = OperatorSignature(
            kind=kind, parameters=parameters, backend_family=self._backend_family
        )
        return NormalizedOperator(
            order=occurrence.order,
            raw_operation=occurrence.operation,
            kind=kind,
            signature=signature,
        )

    def _classify(
        self, occurrence: RawOperatorOccurrence, *, phase: InferencePhase | None
    ) -> tuple[OperatorKind, OperatorParameters]:
        name = canonical_operation_name(occurrence.operation)
        typed: OperatorParameters | None = None
        if name in GEMM_OPERATIONS:
            typed = _gemm_parameters(occurrence, name)
            if typed is not None:
                return OperatorKind.GEMM, typed
        elif name in ATTENTION_OPERATIONS:
            typed = _attention_parameters(occurrence, phase)
            if typed is not None:
                return OperatorKind.ATTENTION, typed
        elif name in NORM_OPERATIONS:
            typed = _norm_parameters(occurrence, NORM_OPERATIONS[name])
            if typed is not None:
                return OperatorKind.NORM, typed
        elif name in EMBEDDING_OPERATIONS:
            typed = _embedding_parameters(occurrence)
            if typed is not None:
                return OperatorKind.EMBEDDING, typed
        elif name in ELEMENTWISE_OPERATIONS:
            return OperatorKind.ELEMENTWISE, _generic_parameters(name, occurrence)
        elif name in REDUCTION_OPERATIONS:
            return OperatorKind.REDUCTION, _generic_parameters(name, occurrence)
        # Unknown operations — and known operations whose typed parameters
        # could not be derived from recorded facts — are preserved as
        # CUSTOM with their raw identity (§19, §52.2).
        return OperatorKind.CUSTOM, CustomOperatorParameters(
            raw_name=occurrence.operation,
            input_shapes=occurrence.input_shapes,
            input_dtypes=occurrence.input_dtypes,
        )


def _first_dtype(occurrence: RawOperatorOccurrence) -> str:
    return occurrence.input_dtypes[0] if occurrence.input_dtypes else UNKNOWN_DTYPE


def _generic_parameters(
    name: str, occurrence: RawOperatorOccurrence
) -> GenericOperatorParameters:
    return GenericOperatorParameters(
        operation=name,
        input_shapes=occurrence.input_shapes,
        dtype=_first_dtype(occurrence),
    )


def _gemm_parameters(
    occurrence: RawOperatorOccurrence, name: str
) -> GemmSignature | None:
    shapes = occurrence.input_shapes
    dtype = _first_dtype(occurrence)
    try:
        if name == "linear":
            # linear(input(*, k), weight(n, k), bias(n)?)
            if len(shapes) < 2 or len(shapes[0]) < 1 or len(shapes[1]) != 2:
                return None
            input_shape, weight_shape = shapes[0], shapes[1]
            k = input_shape[-1]
            n, weight_k = weight_shape
            if weight_k != k:
                return None
            m = 1
            for dimension in input_shape[:-1]:
                m *= dimension
            return GemmSignature(m=m, n=n, k=k, dtype=dtype, transpose_b=True)
        if name in {"mm", "matmul"}:
            if len(shapes) != 2 or len(shapes[0]) != 2 or len(shapes[1]) != 2:
                return None  # batched matmul is not a plain GEMM identity
            (m, k), (k2, n) = shapes
            if k != k2:
                return None
            return GemmSignature(m=m, n=n, k=k, dtype=dtype)
        if name == "addmm":
            # addmm(bias, mat1(m, k), mat2(k, n))
            if len(shapes) != 3 or len(shapes[1]) != 2 or len(shapes[2]) != 2:
                return None
            (m, k), (k2, n) = shapes[1], shapes[2]
            if k != k2:
                return None
            return GemmSignature(m=m, n=n, k=k, dtype=dtype)
    except ValueError:
        return None
    return None


def _attention_parameters(
    occurrence: RawOperatorOccurrence, phase: InferencePhase | None
) -> AttentionSignature | None:
    if phase is None:
        return None  # never guess prefill vs decode (§24)
    shapes = occurrence.input_shapes
    if len(shapes) < 3 or any(len(shape) != 4 for shape in shapes[:3]):
        return None
    batch, heads, q_len, head_dim = shapes[0]
    batch_k, kv_heads, kv_len, key_dim = shapes[1]
    batch_v, _, kv_len_v, value_dim = shapes[2]
    if (
        batch != batch_k
        or batch != batch_v
        or kv_len != kv_len_v
        or head_dim != key_dim
        or head_dim != value_dim
    ):
        return None
    try:
        return AttentionSignature(
            batch_size=batch,
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            q_len=q_len,
            kv_len=kv_len,
            dtype=_first_dtype(occurrence),
            phase=phase,
        )
    except ValueError:
        return None  # e.g. GQA divisibility violated by expanded shapes


def _norm_parameters(
    occurrence: RawOperatorOccurrence, variant: NormVariant
) -> NormSignature | None:
    shapes = occurrence.input_shapes
    if not shapes or len(shapes[0]) != 3:
        return None  # (batch, sequence, hidden) required; ranks are facts
    batch, sequence_length, hidden_size = shapes[0]
    try:
        return NormSignature(
            batch_size=batch,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            dtype=_first_dtype(occurrence),
            variant=variant,
        )
    except ValueError:
        return None


def _embedding_parameters(occurrence: RawOperatorOccurrence) -> EmbeddingSignature | None:
    shapes = occurrence.input_shapes
    # embedding(weight(vocab, dim), indices(batch, sequence))
    if len(shapes) < 2 or len(shapes[0]) != 2 or len(shapes[1]) != 2:
        return None
    (vocab_size, embedding_dim), (batch, sequence_length) = shapes[0], shapes[1]
    try:
        return EmbeddingSignature(
            batch_size=batch,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            dtype=_first_dtype(occurrence),
        )
    except ValueError:
        return None


def unique_operator_signatures(
    graphs: Iterable[NormalizedOperatorGraph],
) -> tuple[OperatorSignature, ...]:
    """Deduplicate signatures across graphs, deterministically (§20).

    First-appearance order is preserved so repeated runs over the same
    inputs produce the same workload list; ``OperatorSignature`` is a
    frozen structural value, so equality is identity of workload, not of
    provenance.
    """
    unique: dict[OperatorSignature, None] = {}
    for graph in graphs:
        for operator in graph.operators:
            unique.setdefault(operator.signature, None)
    return tuple(unique)
