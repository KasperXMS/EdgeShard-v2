"""Stable static model description produced by adapters (spec 9.3).

``ModelLayout`` is the canonical, version-independent description of a
model's backbone. Later phases (profiling, cost modeling) consume layouts;
they never re-inspect Hugging Face configs.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_BLOCK_INDEX_PLACEHOLDER = "{index}"


class ModelLayout(BaseModel):
    """Static structural description of a supported model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_type: str

    num_blocks: int
    hidden_size: int
    intermediate_size: int

    num_attention_heads: int
    num_kv_heads: int

    embedding_prefix: str
    block_prefix_template: str
    final_norm_prefix: str
    lm_head_prefix: str

    tied_word_embeddings: bool

    @field_validator(
        "model_type",
        "embedding_prefix",
        "block_prefix_template",
        "final_norm_prefix",
        "lm_head_prefix",
    )
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("block_prefix_template")
    @classmethod
    def _template_has_index(cls, value: str) -> str:
        if _BLOCK_INDEX_PLACEHOLDER not in value:
            raise ValueError(f"block_prefix_template must contain '{_BLOCK_INDEX_PLACEHOLDER}'")
        return value

    @field_validator(
        "num_blocks", "hidden_size", "intermediate_size", "num_attention_heads", "num_kv_heads"
    )
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @model_validator(mode="after")
    def _check_grouped_query_attention(self) -> Self:
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")
        return self

    def block_prefix(self, index: int) -> str:
        """Return the weight name prefix for Transformer block ``index``."""
        if not 0 <= index < self.num_blocks:
            raise ValueError(f"block index {index} out of range for {self.num_blocks} blocks")
        return self.block_prefix_template.format(index=index)
