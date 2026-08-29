"""Qwen2 / Qwen2.5 family adapter (spec 8.1).

Covers decoder-only Qwen2 models with the standard Hugging Face layout;
see :class:`edgeshard.model.adapters.base.StandardDecoderLMAdapter`.
"""

from __future__ import annotations

from typing import ClassVar

from edgeshard.model.adapters.base import StandardDecoderLMAdapter


class Qwen2Adapter(StandardDecoderLMAdapter):
    """Adapter for the Qwen2 family (``model_type == "qwen2"``)."""

    model_type: ClassVar[str] = "qwen2"
    architectures: ClassVar[tuple[str, ...]] = ("Qwen2ForCausalLM",)
