"""P2C operator-normalizer tests (spec §19-§20, §24, §50 P2C DoD).

Pinned behaviors: explicit mapping tables produce typed signatures;
failed typed derivations and unknown operations degrade to ``CUSTOM``
with raw facts preserved (never dropped, never guessed); attention is
never assigned a phase it was not given; normalized graphs hold exactly
one operator per raw occurrence in order; deduplication is
first-appearance deterministic and cross-extractor stable.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

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
    operator_signature_id,
)
from edgeshard.profiling.operator.extractor import (
    RawOperatorGraph,
    RawOperatorOccurrence,
    TorchExportExtractor,
    TorchProfilerExtractor,
)
from edgeshard.profiling.operator.normalizer import (
    OperatorNormalizer,
    canonical_operation_name,
    unique_operator_signatures,
)


def _graph(
    *operations: tuple[str, tuple[tuple[int, ...], ...], tuple[str, ...]],
) -> RawOperatorGraph:
    return RawOperatorGraph(
        extractor="unit",
        operations=tuple(
            RawOperatorOccurrence(
                order=order, operation=name, input_shapes=shapes, input_dtypes=dtypes
            )
            for order, (name, shapes, dtypes) in enumerate(operations)
        ),
    )


class TestCanonicalOperationName:
    @pytest.mark.parametrize(
        ("raw", "canonical"),
        [
            ("aten.mm.default", "mm"),
            ("aten::mm", "mm"),
            ("aten.linear.default", "linear"),
            ("pow.Tensor_Scalar", "pow"),
            (
                "aten::_scaled_dot_product_flash_attention.default",
                "_scaled_dot_product_flash_attention",
            ),
            ("myorg_fused_kernel", "myorg_fused_kernel"),
        ],
    )
    def test_mapping(self, raw: str, canonical: str) -> None:
        assert canonical_operation_name(raw) == canonical


class TestTypedClassification:
    def test_mm_becomes_gemm(self) -> None:
        graph = _graph(("aten.mm.default", ((64, 32), (32, 16)), ("bf16", "bf16")))
        operator = OperatorNormalizer().normalize(graph).operators[0]
        assert operator.kind is OperatorKind.GEMM
        assert operator.signature.parameters == GemmSignature(m=64, n=16, k=32, dtype="bf16")

    def test_addmm_skips_bias(self) -> None:
        graph = _graph(("aten::addmm", ((16,), (64, 32), (32, 16)), ("fp32", "fp32", "fp32")))
        operator = OperatorNormalizer().normalize(graph).operators[0]
        assert operator.kind is OperatorKind.GEMM
        assert operator.signature.parameters == GemmSignature(m=64, n=16, k=32, dtype="fp32")

    def test_linear_flattens_batch_and_transposes_weight(self) -> None:
        graph = _graph(
            ("aten.linear.default", ((1, 8, 64), (32, 64), (32,)), ("bf16", "bf16", "bf16"))
        )
        operator = OperatorNormalizer().normalize(graph).operators[0]
        assert operator.kind is OperatorKind.GEMM
        assert operator.signature.parameters == GemmSignature(
            m=8, n=32, k=64, dtype="bf16", transpose_b=True
        )

    def test_k_mismatch_degrades_to_custom(self) -> None:
        graph = _graph(("aten.mm.default", ((64, 32), (48, 16)), ("fp32", "fp32")))
        operator = OperatorNormalizer().normalize(graph).operators[0]
        assert operator.kind is OperatorKind.CUSTOM
        assert operator.raw_operation == "aten.mm.default"

    def test_batched_matmul_is_not_plain_gemm(self) -> None:
        graph = _graph(("aten.matmul.default", ((2, 8, 4), (2, 4, 8)), ("fp32", "fp32")))
        operator = OperatorNormalizer().normalize(graph).operators[0]
        assert operator.kind is OperatorKind.CUSTOM

    def test_unknown_dtype_label_preserved(self) -> None:
        graph = _graph(("aten::mm", ((4, 4), (4, 4)), ("unknown", "unknown")))
        operator = OperatorNormalizer().normalize(graph).operators[0]
        assert operator.kind is OperatorKind.GEMM
        assert operator.signature.parameters == GemmSignature(m=4, n=4, k=4, dtype="unknown")


class TestAttentionClassification:
    SDPA_SHAPES = ((1, 4, 8, 16), (1, 2, 8, 16), (1, 2, 8, 16))
    SDPA_DTYPES = ("fp32", "fp32", "fp32")

    def test_prefill_phase_produces_typed_signature(self) -> None:
        graph = _graph(
            ("aten.scaled_dot_product_attention.default", self.SDPA_SHAPES, self.SDPA_DTYPES)
        )
        operator = OperatorNormalizer().normalize(graph, phase=InferencePhase.PREFILL).operators[0]
        assert operator.kind is OperatorKind.ATTENTION
        assert operator.signature.parameters == AttentionSignature(
            batch_size=1,
            num_heads=4,
            num_kv_heads=2,
            head_dim=16,
            q_len=8,
            kv_len=8,
            dtype="fp32",
            phase=InferencePhase.PREFILL,
        )

    def test_decode_phase_recorded(self) -> None:
        graph = _graph(
            ("aten.scaled_dot_product_attention.default", self.SDPA_SHAPES, self.SDPA_DTYPES)
        )
        operator = OperatorNormalizer().normalize(graph, phase=InferencePhase.DECODE).operators[0]
        assert operator.signature.parameters.phase is InferencePhase.DECODE

    def test_phase_is_never_guessed(self) -> None:
        graph = _graph(
            ("aten.scaled_dot_product_attention.default", self.SDPA_SHAPES, self.SDPA_DTYPES)
        )
        operator = OperatorNormalizer().normalize(graph).operators[0]
        assert operator.kind is OperatorKind.CUSTOM
        assert operator.raw_operation == "aten.scaled_dot_product_attention.default"
        assert isinstance(operator.signature.parameters, CustomOperatorParameters)

    def test_inconsistent_head_dim_degrades_to_custom(self) -> None:
        graph = _graph(
            (
                "aten.scaled_dot_product_attention.default",
                ((1, 4, 8, 16), (1, 2, 8, 32), (1, 2, 8, 16)),
                self.SDPA_DTYPES,
            )
        )
        operator = OperatorNormalizer().normalize(graph, phase=InferencePhase.PREFILL).operators[0]
        assert operator.kind is OperatorKind.CUSTOM

    def test_gqa_divisibility_violation_degrades_to_custom(self) -> None:
        graph = _graph(
            (
                "aten.scaled_dot_product_attention.default",
                ((1, 4, 8, 16), (1, 3, 8, 16), (1, 3, 8, 16)),
                self.SDPA_DTYPES,
            )
        )
        operator = OperatorNormalizer().normalize(graph, phase=InferencePhase.PREFILL).operators[0]
        assert operator.kind is OperatorKind.CUSTOM


class TestNormAndEmbeddingClassification:
    def test_rms_norm(self) -> None:
        graph = _graph(("aten::rms_norm", ((2, 8, 64), (64,)), ("fp32", "fp32")))
        operator = OperatorNormalizer().normalize(graph).operators[0]
        assert operator.kind is OperatorKind.NORM
        assert operator.signature.parameters == NormSignature(
            batch_size=2, sequence_length=8, hidden_size=64, dtype="fp32", variant=NormVariant.RMS
        )

    def test_native_layer_norm(self) -> None:
        graph = _graph(("aten::native_layer_norm", ((2, 8, 64),), ("fp32",)))
        operator = OperatorNormalizer().normalize(graph).operators[0]
        assert operator.kind is OperatorKind.NORM
        assert operator.signature.parameters.variant is NormVariant.LAYER

    def test_rank_two_norm_input_degrades_to_custom(self) -> None:
        graph = _graph(("aten::rms_norm", ((8, 64), (64,)), ("fp32", "fp32")))
        operator = OperatorNormalizer().normalize(graph).operators[0]
        assert operator.kind is OperatorKind.CUSTOM

    def test_embedding(self) -> None:
        graph = _graph(("aten.embedding.default", ((128, 64), (2, 8)), ("fp32", "int64")))
        operator = OperatorNormalizer().normalize(graph).operators[0]
        assert operator.kind is OperatorKind.EMBEDDING
        assert operator.signature.parameters == EmbeddingSignature(
            batch_size=2, sequence_length=8, vocab_size=128, embedding_dim=64, dtype="fp32"
        )


class TestGenericAndCustom:
    def test_elementwise(self) -> None:
        graph = _graph(("aten.silu.default", ((2, 8, 128),), ("bf16",)))
        operator = OperatorNormalizer().normalize(graph).operators[0]
        assert operator.kind is OperatorKind.ELEMENTWISE
        assert operator.signature.parameters == GenericOperatorParameters(
            operation="silu", input_shapes=((2, 8, 128),), dtype="bf16"
        )

    def test_reduction_drops_overload(self) -> None:
        graph = _graph(("aten.mean.dim", ((2, 8, 64),), ("fp32",)))
        operator = OperatorNormalizer().normalize(graph).operators[0]
        assert operator.kind is OperatorKind.REDUCTION
        assert operator.signature.parameters.operation == "mean"

    def test_unknown_operation_preserved_verbatim(self) -> None:
        graph = _graph(("myorg.fused_mamba.default", ((2, 8), (8, 2)), ("bf16", "bf16")))
        operator = OperatorNormalizer().normalize(graph).operators[0]
        assert operator.kind is OperatorKind.CUSTOM
        assert operator.signature.parameters == CustomOperatorParameters(
            raw_name="myorg.fused_mamba.default",
            input_shapes=((2, 8), (8, 2)),
            input_dtypes=("bf16", "bf16"),
        )

    def test_nothing_dropped_and_order_preserved(self) -> None:
        graph = _graph(
            ("aten.linear.default", ((8, 64), (32, 64)), ("fp32", "fp32")),
            ("myorg.unknown_kernel", ((8, 32),), ("fp32",)),
            ("aten.silu.default", ((8, 32),), ("fp32",)),
            ("aten::mm", ((8, 32), (32, 64)), ("fp32", "fp32")),
            ("aten.native_layer_norm", ((1, 8, 64),), ("fp32",)),
        )
        normalized = OperatorNormalizer().normalize(graph)
        assert len(normalized.operators) == len(graph.operations)
        assert [o.order for o in normalized.operators] == [0, 1, 2, 3, 4]
        assert [o.raw_operation for o in normalized.operators] == [
            raw.operation for raw in graph.operations
        ]
        assert [o.kind for o in normalized.operators] == [
            OperatorKind.GEMM,
            OperatorKind.CUSTOM,
            OperatorKind.ELEMENTWISE,
            OperatorKind.GEMM,
            OperatorKind.NORM,
        ]
        assert normalized.extractor == "unit"

    def test_backend_family_recorded(self) -> None:
        graph = _graph(("aten::mm", ((4, 4), (4, 4)), ("fp32", "fp32")))
        default = OperatorNormalizer().normalize(graph)
        assert default.operators[0].signature.backend_family == "torch"
        custom = OperatorNormalizer(backend_family="triton").normalize(graph)
        assert custom.operators[0].signature.backend_family == "triton"

    def test_empty_backend_family_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            OperatorNormalizer(backend_family="")


class TestCrossExtractorStability:
    def test_export_and_profiler_names_normalize_identically(self) -> None:
        shapes = ((64, 32), (32, 16))
        dtypes = ("fp32", "fp32")
        export = OperatorNormalizer().normalize(_graph(("aten.mm.default", shapes, dtypes)))
        profiler = OperatorNormalizer().normalize(_graph(("aten::mm", shapes, dtypes)))
        assert export.operators[0].signature == profiler.operators[0].signature
        assert operator_signature_id(export.operators[0].signature) == operator_signature_id(
            profiler.operators[0].signature
        )

    def test_real_extractors_agree_on_linear_signature(self) -> None:
        linear = nn.Linear(8, 4)
        args = (torch.randn(2, 8),)
        export_graph = TorchExportExtractor().extract(linear, args, {})
        profiler_graph = TorchProfilerExtractor().extract(linear, args, {})
        normalizer = OperatorNormalizer()
        export_gemms = {
            o.signature
            for o in normalizer.normalize(export_graph).operators
            if o.kind is OperatorKind.GEMM
        }
        profiler_gemms = {
            o.signature
            for o in normalizer.normalize(profiler_graph).operators
            if o.kind is OperatorKind.GEMM
        }
        expected = {
            normalizer.normalize(export_graph).operators[0].signature,
        }
        assert export_gemms == expected
        assert expected <= profiler_gemms


class TestUniqueOperatorSignatures:
    def test_deduplicates_first_appearance_order(self) -> None:
        graph_a = _graph(
            ("aten::mm", ((4, 4), (4, 4)), ("fp32", "fp32")),
            ("aten.silu.default", ((4, 4),), ("fp32",)),
        )
        graph_b = _graph(
            ("aten.mm.default", ((4, 4), (4, 4)), ("fp32", "fp32")),  # same as graph_a's mm
            ("aten::mm", ((8, 8), (8, 8)), ("fp32", "fp32")),  # new shape
        )
        normalizer = OperatorNormalizer()
        unique = unique_operator_signatures(
            [normalizer.normalize(graph_a), normalizer.normalize(graph_b)]
        )
        assert len(unique) == 3
        gemms = [s.parameters for s in unique if s.kind is OperatorKind.GEMM]
        assert gemms[0] == GemmSignature(m=4, n=4, k=4, dtype="fp32")
        assert gemms[-1] == GemmSignature(m=8, n=8, k=8, dtype="fp32")
        assert unique[1].kind is OperatorKind.ELEMENTWISE

    def test_deterministic(self) -> None:
        normalizer = OperatorNormalizer()
        graphs = [
            normalizer.normalize(_graph(("aten::mm", ((4, 4), (4, 4)), ("fp32", "fp32")))),
            normalizer.normalize(_graph(("aten.silu.default", ((4, 4),), ("fp32",)))),
        ]
        assert unique_operator_signatures(graphs) == unique_operator_signatures(graphs)

    def test_empty_input(self) -> None:
        assert unique_operator_signatures([]) == ()

    def test_identical_layers_share_one_workload_surface(
        self, qwen2_layer_exports: tuple[RawOperatorGraph, ...]
    ) -> None:
        normalizer = OperatorNormalizer()
        first, second = (
            normalizer.normalize(graph, phase=InferencePhase.PREFILL)
            for graph in qwen2_layer_exports
        )
        assert unique_operator_signatures([first]) == unique_operator_signatures([first, second])


class TestLayerGraphEndToEnd:
    """Normalize a real tiny-Qwen2 layer export (§50 P2C DoD)."""

    def test_kinds_coverage(self, qwen2_layer_exports: tuple[RawOperatorGraph, ...]) -> None:
        graph = qwen2_layer_exports[0]
        normalized = OperatorNormalizer().normalize(graph, phase=InferencePhase.PREFILL)
        assert len(normalized.operators) == len(graph.operations)
        kinds = {o.kind for o in normalized.operators}
        assert OperatorKind.GEMM in kinds
        assert OperatorKind.ATTENTION in kinds
        assert OperatorKind.ELEMENTWISE in kinds
        assert OperatorKind.CUSTOM in kinds  # transposes, slices, cat: preserved, not dropped

        attention = [o for o in normalized.operators if o.kind is OperatorKind.ATTENTION]
        assert len(attention) == 1
        assert attention[0].signature.parameters == AttentionSignature(
            batch_size=1,
            num_heads=4,
            num_kv_heads=2,
            head_dim=16,
            q_len=8,
            kv_len=8,
            dtype="fp32",
            phase=InferencePhase.PREFILL,
        )

        gemms = [o for o in normalized.operators if o.kind is OperatorKind.GEMM]
        assert len(gemms) == 7
        assert all(
            isinstance(o.signature.parameters, GemmSignature) and o.signature.parameters.transpose_b
            for o in gemms
        )

    def test_without_phase_attention_degrades_but_nothing_is_lost(
        self, qwen2_layer_exports: tuple[RawOperatorGraph, ...]
    ) -> None:
        graph = qwen2_layer_exports[0]
        normalizer = OperatorNormalizer()
        with_phase = normalizer.normalize(graph, phase=InferencePhase.PREFILL)
        without_phase = normalizer.normalize(graph)
        assert len(without_phase.operators) == len(with_phase.operators)
        assert all(o.kind is not OperatorKind.ATTENTION for o in without_phase.operators)
        # Only the attention occurrence changes kind; the rest is identical.
        differing = [
            (a, b)
            for a, b in zip(with_phase.operators, without_phase.operators, strict=True)
            if a.kind != b.kind
        ]
        assert len(differing) == 1
        assert differing[0][0].kind is OperatorKind.ATTENTION
        assert differing[0][1].kind is OperatorKind.CUSTOM
