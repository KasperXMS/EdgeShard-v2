"""Qwen family profiling adapter (Phase 2 spec §16.1).

Covers decoder-only Qwen2 models with the standard Hugging Face layout;
see :class:`~edgeshard.profiling.model.adapters.base.StandardDecoderProfilingAdapter`.
"""

from __future__ import annotations

from typing import ClassVar

from edgeshard.profiling.model.adapters.base import StandardDecoderProfilingAdapter


class QwenProfilingAdapter(StandardDecoderProfilingAdapter):
    """Profiling adapter for the Qwen2 family (``model_type == "qwen2"``)."""

    model_type: ClassVar[str] = "qwen2"
    architectures: ClassVar[tuple[str, ...]] = ("Qwen2ForCausalLM",)
    architecture_family: ClassVar[str] = "qwen2"
