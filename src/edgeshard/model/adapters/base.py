"""Adapter protocol and shared adapter infrastructure (spec 9.2).

All Hugging Face architecture-specific behavior must terminate in this
package (spec 4.1). No module outside ``edgeshard.model.adapters`` may branch
on ``model_type`` or architecture names.
"""

from __future__ import annotations

from typing import ClassVar, Protocol

import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from edgeshard.model.layout import ModelLayout
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import BlockRange, ShardSpec


def load_model_config(source: ModelSource) -> PretrainedConfig:
    """Load only the model configuration — never the weights (spec 9.1)."""
    path = source.ensure_local()
    config: PretrainedConfig = AutoConfig.from_pretrained(path, trust_remote_code=False)
    return config


def keep_layer_range(layers: nn.ModuleList, blocks: BlockRange) -> None:
    """Reduce a ``ModuleList`` in place to the half-open block range.

    Remaining layers are re-indexed from 0; callers must map skeleton index
    ``i`` back to global block ``blocks.start + i``.
    """
    if blocks.start < 0 or len(layers) < blocks.end:
        raise ValueError(f"block range {blocks!r} does not fit {len(layers)} layers")
    del layers[blocks.end :]
    del layers[: blocks.start]


class ModelAdapter(Protocol):
    """Adapter for one Hugging Face model architecture.

    Responsibilities over Phase 0 (spec 9.2): structure discovery and layout,
    shard-compatible skeleton construction, and canonical-state adaptation for
    prefill/decode. Signatures may evolve across 0A-0C, but all
    architecture-specific behavior stays confined to adapter implementations.
    """

    #: ``config.model_type`` handled by this adapter, e.g. ``"llama"``.
    model_type: ClassVar[str]
    #: ``config.architectures`` entries handled by this adapter.
    architectures: ClassVar[tuple[str, ...]]

    def inspect(self, source: ModelSource) -> ModelLayout:
        """Describe the model backbone without loading weights."""
        ...


class StandardDecoderLMAdapter:
    """Shared implementation for HF decoder-only causal LMs with the standard layout.

    Layout::

        <Family>ForCausalLM
        ├── model.embed_tokens
        ├── model.layers.{index}
        ├── model.norm
        └── lm_head

    Concrete subclasses only declare identity (``model_type`` and
    ``architectures``). Models whose layout deviates from the above must
    implement :class:`ModelAdapter` directly instead of subclassing here.
    """

    model_type: ClassVar[str]
    architectures: ClassVar[tuple[str, ...]]

    #: Weight-name prefixes of the standard layout; subclasses may override.
    embedding_prefix: ClassVar[str] = "model.embed_tokens"
    block_prefix_template: ClassVar[str] = "model.layers.{index}"
    final_norm_prefix: ClassVar[str] = "model.norm"
    lm_head_prefix: ClassVar[str] = "lm_head"

    def inspect(self, source: ModelSource) -> ModelLayout:
        """Describe the backbone from ``config.json`` without loading weights."""
        config = load_model_config(source)
        num_attention_heads: int = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", None)
        return ModelLayout(
            model_type=self.model_type,
            num_blocks=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads or num_attention_heads,
            embedding_prefix=self.embedding_prefix,
            block_prefix_template=self.block_prefix_template,
            final_norm_prefix=self.final_norm_prefix,
            lm_head_prefix=self.lm_head_prefix,
            tied_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        )

    def build_skeleton(self, source: ModelSource, shard: ShardSpec) -> nn.Module:
        """Create a meta-device skeleton reduced to the shard's modules (spec 10).

        Layers keep skeleton-local indexing: skeleton index ``i`` corresponds
        to global block ``shard.blocks.start + i``.
        """
        layout = self.inspect(source)
        shard.validate_bounds(layout.num_blocks)
        config = load_model_config(source)
        with init_empty_weights():
            skeleton = AutoModelForCausalLM.from_config(config)

        # HF internals are deliberately treated as opaque here; behavior is
        # covered by adapter tests.
        backbone = skeleton.model
        if not shard.include_input_stage:
            backbone.embed_tokens = nn.Identity()
        keep_layer_range(backbone.layers, shard.blocks)
        if not shard.include_output_stage:
            backbone.norm = nn.Identity()
            skeleton.lm_head = nn.Identity()
        model: nn.Module = skeleton
        return model
