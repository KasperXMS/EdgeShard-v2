"""Model-family profiling adapters (Phase 2 spec §16.1).

Mirrors the Phase 0 ``edgeshard.model.adapters`` pattern: all
architecture-family behavior for profiling terminates in this package,
concrete adapters declare identity (``model_type`` / ``architectures``),
and a registry resolves adapters from a :class:`ModelLayout`. Unknown
models fail explicitly with ``UNSUPPORTED_MODEL`` — the registry never
guesses.

v1 supports exactly the families already relevant to the repository
(Qwen2, Llama). VLM stages (vision encoder, projector) enter only after
the core language-model path is hardware-validated (§16.1).
"""
