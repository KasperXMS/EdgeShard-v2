"""Adapter protocol and shared adapter infrastructure (spec 9.2).

All Hugging Face architecture-specific behavior must terminate in this
package (spec 4.1). No module outside ``edgeshard.model.adapters`` may branch
on ``model_type`` or architecture names.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.cache_utils import DynamicCache

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


def _rope_base(config: PretrainedConfig) -> float:
    """Read the rotary base frequency, preferring the canonical rope table."""
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        return float(rope_parameters["rope_theta"])
    return float(getattr(config, "rope_theta", 10000.0))


def _rope_head_dim(config: PretrainedConfig) -> int:
    """Rotary dimension per head: ``head_dim`` when set, else derived."""
    head_dim = getattr(config, "head_dim", None)
    if head_dim:
        return int(head_dim)
    return int(config.hidden_size // config.num_attention_heads)


class ModelAdapter(Protocol):
    """Adapter for one Hugging Face model architecture.

    Responsibilities over Phase 0 (spec 9.2): structure discovery and layout,
    shard-compatible skeleton construction, and native-execution hooks used
    by ``ShardModule`` for prefill/decode. Signatures may evolve across
    0A-0C, but all architecture-specific behavior stays confined to adapter
    implementations.
    """

    #: ``config.model_type`` handled by this adapter, e.g. ``"llama"``.
    model_type: ClassVar[str]
    #: ``config.architectures`` entries handled by this adapter.
    architectures: ClassVar[tuple[str, ...]]

    def inspect(self, source: ModelSource) -> ModelLayout:
        """Describe the model backbone without loading weights."""
        ...

    def build_skeleton(self, source: ModelSource, shard: ShardSpec) -> nn.Module:
        """Create a meta-device skeleton reduced to the shard's modules."""
        ...

    def new_cache(self) -> object:
        """Create a fresh native KV cache for one session (spec 13)."""
        ...

    def embed_tokens(self, module: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the input stage on token ids."""
        ...

    def forward_blocks(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache: Any,
    ) -> torch.Tensor:
        """Run the shard's Transformer blocks, updating the session cache."""
        ...

    def finalize(self, module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the output stage (LM head) producing logits.

        The final norm is applied by ``forward_blocks`` as part of the HF
        backbone pass; this hook must not apply it again.
        """
        ...

    def restore_high_precision_buffers(self, module: nn.Module) -> None:
        """Re-materialize buffers HF keeps in float32 after dtype placement.

        Called after the blanket ``module.to(device, dtype)`` so reduced
        precision shards still reproduce the reference model's numerics.
        """
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
        # Attention modules cache by ``self_attn.layer_idx``; after trimming,
        # retained layers must use skeleton-local indices so the session KV
        # cache (indexed 0..k-1) stays coherent.
        for local_index, layer in enumerate(backbone.layers):
            layer.self_attn.layer_idx = local_index
        if not shard.include_output_stage:
            backbone.norm = nn.Identity()
            skeleton.lm_head = nn.Identity()
        model: nn.Module = skeleton
        return model

    def new_cache(self) -> object:
        """Create the native HF dynamic KV cache for one session."""
        return DynamicCache()

    def embed_tokens(self, module: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        backbone: Any = module.model
        embedded: torch.Tensor = backbone.embed_tokens(input_ids)
        return embedded

    def forward_blocks(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache: Any,
    ) -> torch.Tensor:
        """Run the shard's Transformer blocks via the HF backbone forward.

        ``forward_blocks`` delegates to the backbone's model-level forward
        with ``inputs_embeds=hidden_states``: Hugging Face handles rotary
        embeddings, causal masking, attention dispatch, ``cache_position``
        bookkeeping, and the loop over the shard's retained (skeleton-local)
        layers, so shards execute with the same kernels as the untouched
        reference model. The session cache is updated in place.

        Norm ownership: the backbone's final operation is ``model.norm``.
        The output shard's skeleton keeps the real norm, so its backbone
        pass already normalizes; non-final skeletons replace the norm with
        ``Identity``, so their output stays pre-norm. ``finalize`` runs the
        LM head only.
        """
        backbone: Any = module.model
        q_len = int(hidden_states.shape[1])
        past_length: int = int(cache.get_seq_length())
        cache_position = torch.arange(
            past_length, past_length + q_len, device=hidden_states.device
        )
        output = backbone(
            inputs_embeds=hidden_states,
            position_ids=positions,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )
        result: torch.Tensor = output.last_hidden_state
        return result

    def finalize(self, module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the LM head only; the backbone pass already applied the norm."""
        head: Any = module.lm_head
        logits: torch.Tensor = head(hidden_states)
        return logits

    def restore_high_precision_buffers(self, module: nn.Module) -> None:
        """Recompute rotary inverse frequencies in float32 after placement.

        Hugging Face keeps ``rotary_emb.inv_freq`` in float32 even for
        reduced-precision checkpoints and casts only the cos/sin tables to
        the compute dtype inside the rotary forward. A blanket
        ``module.to(dtype)`` would round the frequency table itself,
        perturbing every attention phase and breaking BF16 alignment with the
        reference. Recomputing the table in float32 restores the exact HF
        values. FP32 shards are left untouched.
        """
        backbone: Any = module.model
        rotary = getattr(backbone, "rotary_emb", None)
        if rotary is None:
            return
        inv_freq = getattr(rotary, "inv_freq", None)
        if not isinstance(inv_freq, torch.Tensor) or inv_freq.dtype == torch.float32:
            return
        config: PretrainedConfig = module.config
        head_dim = _rope_head_dim(config)
        base = _rope_base(config)
        arange = torch.arange(0, head_dim, 2, dtype=torch.float32)
        table = 1.0 / (base ** (arange / head_dim))
        table = table.to(inv_freq.device)
        rotary.inv_freq = table
        if hasattr(rotary, "original_inv_freq"):
            rotary.original_inv_freq = table.clone()
