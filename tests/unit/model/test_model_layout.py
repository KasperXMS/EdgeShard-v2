"""ModelLayout tests: stable static model description (spec 9.3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from edgeshard.model.layout import ModelLayout


def make_layout(**overrides: object) -> ModelLayout:
    values: dict[str, object] = {
        "model_type": "llama",
        "num_blocks": 4,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_kv_heads": 2,
        "embedding_prefix": "model.embed_tokens",
        "block_prefix_template": "model.layers.{index}",
        "final_norm_prefix": "model.norm",
        "lm_head_prefix": "lm_head",
        "tied_word_embeddings": False,
    }
    values.update(overrides)
    return ModelLayout.model_validate(values)


def test_valid_layout() -> None:
    layout = make_layout()
    assert layout.model_type == "llama"
    assert layout.num_blocks == 4


def test_block_prefix_formats_index() -> None:
    layout = make_layout()
    assert layout.block_prefix(0) == "model.layers.0"
    assert layout.block_prefix(3) == "model.layers.3"


def test_block_prefix_rejects_out_of_range() -> None:
    layout = make_layout()
    with pytest.raises(ValueError, match="out of range"):
        layout.block_prefix(4)
    with pytest.raises(ValueError, match="out of range"):
        layout.block_prefix(-1)


@pytest.mark.parametrize(
    "field",
    ["num_blocks", "hidden_size", "intermediate_size", "num_attention_heads", "num_kv_heads"],
)
def test_positive_dimensions_required(field: str) -> None:
    with pytest.raises(ValidationError):
        make_layout(**{field: 0})


@pytest.mark.parametrize(
    "field",
    [
        "model_type",
        "embedding_prefix",
        "block_prefix_template",
        "final_norm_prefix",
        "lm_head_prefix",
    ],
)
def test_prefixes_must_be_non_empty(field: str) -> None:
    with pytest.raises(ValidationError):
        make_layout(**{field: "   "})


def test_block_prefix_template_requires_index_placeholder() -> None:
    with pytest.raises(ValidationError, match="index"):
        make_layout(block_prefix_template="model.layers")


def test_kv_heads_must_divide_attention_heads() -> None:
    with pytest.raises(ValidationError, match="divisible"):
        make_layout(num_attention_heads=4, num_kv_heads=3)


def test_multi_head_attention_is_allowed() -> None:
    """num_kv_heads == num_attention_heads (plain MHA) is valid."""
    layout = make_layout(num_kv_heads=4)
    assert layout.num_kv_heads == 4


def test_frozen_and_strict() -> None:
    layout = make_layout()
    with pytest.raises(ValidationError):
        layout.num_blocks = 8  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ModelLayout.model_validate({**layout.model_dump(), "unknown": 1})
