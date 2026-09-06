"""Llama family profiling adapter (Phase 2 spec §16.1).

Covers decoder-only Llama models with the standard Hugging Face layout;
see :class:`~edgeshard.profiling.model.adapters.base.StandardDecoderProfilingAdapter`.
"""

from __future__ import annotations

from typing import ClassVar

from edgeshard.profiling.model.adapters.base import StandardDecoderProfilingAdapter


class LlamaProfilingAdapter(StandardDecoderProfilingAdapter):
    """Profiling adapter for the Llama family (``model_type == "llama"``)."""

    model_type: ClassVar[str] = "llama"
    architectures: ClassVar[tuple[str, ...]] = ("LlamaForCausalLM",)
    architecture_family: ClassVar[str] = "llama"
