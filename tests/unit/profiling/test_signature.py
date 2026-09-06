"""Workload signature validation and identity (spec §5-7, §19, §20)."""

from __future__ import annotations

import pytest

from edgeshard.profiling.domain.hashing import normalized_items
from edgeshard.profiling.domain.signature import (
    AttentionSignature,
    CustomOperatorParameters,
    GemmSignature,
    GenericOperatorParameters,
    InferencePhase,
    LayerPosition,
    ModuleKind,
    ModuleSignature,
    NormSignature,
    NormVariant,
    OperatorKind,
    OperatorSignature,
    ProfilingGranularity,
    TransformerLayerSignature,
    module_signature_id,
    operator_signature_id,
    transformer_layer_signature_id,
)

GOLDEN_LAYER_ID = "7bb7ca85edfa86c52606a098aaa2019409929abfbadab183c887a48e0e934d3c"
GOLDEN_GEMM_ID = "2b1d83c06844e9d671cbaf3b15d1a34c9de1f6ab6f5d3e605a7b38f664b99317"


def _layer(**overrides: object) -> TransformerLayerSignature:
    base: dict[str, object] = {
        "architecture_family": "qwen2",
        "layer_type": "decoder",
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "dtype": "bf16",
        "quantization": None,
    }
    base.update(overrides)
    return TransformerLayerSignature(**base)  # type: ignore[arg-type]


def test_enum_values_match_spec_vocabulary() -> None:
    assert [g.value for g in ProfilingGranularity] == [
        "transformer_layer",
        "module",
        "operator",
    ]
    assert [k.value for k in OperatorKind] == [
        "gemm",
        "attention",
        "norm",
        "rotary",
        "elementwise",
        "reduction",
        "embedding",
        "kv_copy",
        "custom",
    ]
    assert {k.value for k in ModuleKind} == {
        "attention",
        "mlp",
        "norm",
        "projection",
        "embedding",
        "lm_head",
        "vision_encoder",
        "projector",
        "other",
    }
    assert {p.value for p in LayerPosition} == {"early", "middle", "late", "special"}


def test_inference_phase_values_are_wire_compatible_with_runtime() -> None:
    """The redeclared enum must match the frozen runtime phase values."""
    from edgeshard.inference.state import InferencePhase as RuntimePhase

    assert {phase.value for phase in InferencePhase} == {
        phase.value for phase in RuntimePhase
    }


def test_layer_signature_validation() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        _layer(hidden_size=0)
    with pytest.raises(ValueError, match="divisible"):
        _layer(num_attention_heads=28, num_kv_heads=3)
    with pytest.raises(ValueError, match="architecture_family"):
        _layer(architecture_family="")
    with pytest.raises(ValueError, match="head_dim"):
        _layer(head_dim=-1)


def test_layer_signature_id_is_deterministic_and_pinned() -> None:
    assert transformer_layer_signature_id(_layer()) == GOLDEN_LAYER_ID
    assert transformer_layer_signature_id(_layer()) == transformer_layer_signature_id(
        _layer()
    )
    assert transformer_layer_signature_id(_layer(dtype="fp16")) != GOLDEN_LAYER_ID


def test_layer_signature_has_no_physical_identity_fields() -> None:
    """§6.1: reusable identities never carry worker/device provenance."""
    field_names = set(_layer().__dataclass_fields__)
    assert not field_names & {"worker_id", "device_id", "device_ids"}


def test_module_signature_mapping_order_is_irrelevant() -> None:
    params_a = normalized_items({"hidden_size": 3584, "num_heads": 28}, "p")
    params_b = normalized_items({"num_heads": 28, "hidden_size": 3584}, "p")
    first = ModuleSignature(
        kind=ModuleKind.ATTENTION,
        architecture_family="qwen2",
        structural_parameters=params_a,
        dtype="bf16",
        quantization=None,
    )
    second = ModuleSignature(
        kind=ModuleKind.ATTENTION,
        architecture_family="qwen2",
        structural_parameters=params_b,
        dtype="bf16",
        quantization=None,
    )
    assert module_signature_id(first) == module_signature_id(second)
    assert first.parameters == {"hidden_size": 3584, "num_heads": 28}


def test_module_signature_is_hashable_for_dedup() -> None:
    """§20: deduplication puts signatures into sets."""
    signature = ModuleSignature(
        kind=ModuleKind.MLP,
        architecture_family="llama",
        structural_parameters=normalized_items({"intermediate_size": 128}, "p"),
        dtype="bf16",
        quantization=None,
    )
    assert len({signature, signature}) == 1


def test_module_signature_rejects_unsorted_parameters() -> None:
    with pytest.raises(ValueError, match="sorted"):
        ModuleSignature(
            kind=ModuleKind.NORM,
            architecture_family="llama",
            structural_parameters=(("b", 1), ("a", 2)),
            dtype="fp32",
            quantization=None,
        )


def test_gemm_and_attention_and_norm_signature_validation() -> None:
    with pytest.raises(ValueError, match="m"):
        GemmSignature(m=0, n=1, k=1, dtype="bf16")
    with pytest.raises(ValueError, match="divisible"):
        AttentionSignature(
            batch_size=1,
            num_heads=7,
            num_kv_heads=4,
            head_dim=128,
            q_len=512,
            kv_len=512,
            dtype="bf16",
            phase=InferencePhase.PREFILL,
        )
    with pytest.raises(ValueError, match="sequence_length"):
        NormSignature(
            batch_size=1,
            sequence_length=0,
            hidden_size=64,
            dtype="bf16",
            variant=NormVariant.RMS,
        )


def test_operator_signature_requires_matching_parameters() -> None:
    with pytest.raises(ValueError, match="GemmSignature"):
        OperatorSignature(
            kind=OperatorKind.GEMM,
            parameters=AttentionSignature(
                batch_size=1,
                num_heads=4,
                num_kv_heads=4,
                head_dim=64,
                q_len=8,
                kv_len=8,
                dtype="bf16",
                phase=InferencePhase.PREFILL,
            ),
            backend_family="torch",
        )


def test_gemm_operator_id_is_pinned() -> None:
    signature = OperatorSignature(
        kind=OperatorKind.GEMM,
        parameters=GemmSignature(m=512, n=3584, k=3584, dtype="bf16"),
        backend_family="torch",
    )
    assert operator_signature_id(signature) == GOLDEN_GEMM_ID


def test_backend_family_participates_in_operator_identity() -> None:
    """§27: measurements are reusable only within the same backend primitive."""
    params = GemmSignature(m=512, n=3584, k=3584, dtype="bf16")
    torch_id = operator_signature_id(
        OperatorSignature(kind=OperatorKind.GEMM, parameters=params, backend_family="torch")
    )
    other_id = operator_signature_id(
        OperatorSignature(kind=OperatorKind.GEMM, parameters=params, backend_family="triton-x")
    )
    assert torch_id != other_id


def test_unknown_operators_are_preserved_as_custom() -> None:
    """§19: unknown ops keep raw name/shapes/metadata and never vanish."""
    custom = OperatorSignature(
        kind=OperatorKind.CUSTOM,
        parameters=CustomOperatorParameters(
            raw_name="aten::_flash_attention_forward",
            input_shapes=((1, 8, 512, 64), (1, 8, 512, 64)),
            input_dtypes=("bf16", "bf16"),
            metadata=normalized_items({"is_causal": True, "scale": 0.125}, "metadata"),
        ),
        backend_family="torch",
    )
    digest = operator_signature_id(custom)
    assert digest == operator_signature_id(
        OperatorSignature(
            kind=OperatorKind.CUSTOM,
            parameters=CustomOperatorParameters(
                raw_name="aten::_flash_attention_forward",
                input_shapes=((1, 8, 512, 64), (1, 8, 512, 64)),
                input_dtypes=("bf16", "bf16"),
                metadata=normalized_items({"scale": 0.125, "is_causal": True}, "metadata"),
            ),
            backend_family="torch",
        )
    )


def test_generic_operator_parameters_validation() -> None:
    with pytest.raises(ValueError, match="operation"):
        GenericOperatorParameters(operation="", input_shapes=((1, 2),), dtype="fp32")
    with pytest.raises(ValueError, match="input_shapes"):
        GenericOperatorParameters(operation="aten.relu", input_shapes=((0, 2),), dtype="fp32")
