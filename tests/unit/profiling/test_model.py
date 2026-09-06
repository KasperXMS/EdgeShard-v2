"""Static model characterization domain (spec §16-17)."""

from __future__ import annotations

import pytest

from edgeshard.profiling.domain.model import (
    ModelCharacterization,
    ModelReference,
    ModelStage,
    StageKind,
    model_signature_id,
)


def _characterization(**overrides: object) -> ModelCharacterization:
    base: dict[str, object] = {
        "model": ModelReference(model_id="Qwen/Qwen2.5-7B", revision="abc123"),
        "architecture_family": "qwen2",
        "num_layers": 28,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "vocab_size": 152064,
        "num_attention_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "dtype": "bf16",
        "quantization": None,
        "tied_word_embeddings": False,
        "stages": (
            ModelStage(kind=StageKind.EMBEDDING),
            ModelStage(kind=StageKind.TRANSFORMER_LAYER_GROUP, layer_count=28),
            ModelStage(kind=StageKind.FINAL_NORM),
            ModelStage(kind=StageKind.LM_HEAD),
        ),
    }
    base.update(overrides)
    return ModelCharacterization(**base)  # type: ignore[arg-type]


def test_model_reference_validation() -> None:
    with pytest.raises(ValueError, match="model_id"):
        ModelReference(model_id="")
    with pytest.raises(ValueError, match="revision"):
        ModelReference(model_id="m", revision="")


def test_stage_layer_count_rules() -> None:
    with pytest.raises(ValueError, match="layer_count"):
        ModelStage(kind=StageKind.TRANSFORMER_LAYER_GROUP)
    with pytest.raises(ValueError, match="positive"):
        ModelStage(kind=StageKind.TRANSFORMER_LAYER_GROUP, layer_count=0)
    with pytest.raises(ValueError, match="transformer layer group"):
        ModelStage(kind=StageKind.EMBEDDING, layer_count=1)


def test_characterization_consistency_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _characterization(num_attention_heads=28, num_kv_heads=3)
    with pytest.raises(ValueError, match="num_layers"):
        _characterization(
            stages=(
                ModelStage(kind=StageKind.TRANSFORMER_LAYER_GROUP, layer_count=27),
            )
        )
    with pytest.raises(ValueError, match="stages"):
        _characterization(stages=())
    with pytest.raises(ValueError, match="hidden_size"):
        _characterization(hidden_size=0)
    with pytest.raises(ValueError, match="quantization"):
        _characterization(quantization="")


def test_signature_id_is_structural_only() -> None:
    """§7/§17: model id and revision are provenance, not structure."""
    baseline = model_signature_id(_characterization())
    renamed = model_signature_id(
        _characterization(model=ModelReference(model_id="mirror/same-model", revision="fff"))
    )
    assert baseline == renamed


def test_signature_id_separates_structure_dtype_quantization() -> None:
    baseline = model_signature_id(_characterization())
    assert baseline != model_signature_id(_characterization(dtype="fp16"))
    assert baseline != model_signature_id(_characterization(quantization="int8"))
    assert baseline != model_signature_id(
        _characterization(
            num_layers=29,
            stages=(
                ModelStage(kind=StageKind.TRANSFORMER_LAYER_GROUP, layer_count=29),
            ),
        )
    )


def test_vlm_stage_graph_is_expressible() -> None:
    """§17: models are not assumed to be homogeneous decoder stacks."""
    vlm = _characterization(
        stages=(
            ModelStage(kind=StageKind.VISION_ENCODER),
            ModelStage(kind=StageKind.PROJECTOR),
            ModelStage(kind=StageKind.TRANSFORMER_LAYER_GROUP, layer_count=28),
            ModelStage(kind=StageKind.LM_HEAD),
        )
    )
    assert model_signature_id(vlm) != model_signature_id(_characterization())
