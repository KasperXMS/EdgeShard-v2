"""Llama family adapter (spec 8.1).

Covers decoder-only Llama models with the standard Hugging Face layout;
see :class:`edgeshard.model.adapters.base.StandardDecoderLMAdapter`.
"""

from __future__ import annotations

from typing import ClassVar

from edgeshard.model.adapters.base import StandardDecoderLMAdapter


class LlamaAdapter(StandardDecoderLMAdapter):
    """Adapter for the Llama family (``model_type == "llama"``)."""

    model_type: ClassVar[str] = "llama"
    architectures: ClassVar[tuple[str, ...]] = ("LlamaForCausalLM",)
