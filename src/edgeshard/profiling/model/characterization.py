"""Characterization facade: ModelSource → ModelCharacterization (§17).

Resolution chain, each step explicit and fail-loud:

1. Phase 0 ``AdapterRegistry`` resolves the family adapter and produces
   the :class:`ModelLayout` (backbone structure — profiling never
   re-inspects HF configs for it);
2. the profiling registry resolves the :class:`ModelProfilingAdapter`
   from the layout, failing ``UNSUPPORTED_MODEL`` when unknown;
3. the adapter builds the static :class:`ModelCharacterization` —
   no weights are loaded and no benchmark is run (§17).

``dtype`` is a *declared* measurement context label (the dtype the model
will be profiled in), not an observation; quantization defaults to the
config's ``quant_method`` when present.
"""

from __future__ import annotations

from transformers import PretrainedConfig

from edgeshard.model.adapters.base import load_model_config
from edgeshard.model.adapters.registry import (
    AdapterRegistry,
    default_registry,
    resolve_adapter_for_source,
)
from edgeshard.model.source import ModelSource
from edgeshard.profiling.domain.model import ModelCharacterization, ModelReference
from edgeshard.profiling.model.adapters.base import (
    ProfilingAdapterRegistry,
    resolve_profiling_adapter,
)

UNRESOLVED_REVISION = "local"
"""Provenance label when a source carries no immutable revision."""


def model_reference_for(source: ModelSource, config: PretrainedConfig) -> ModelReference:
    """Provenance identity of the characterized snapshot.

    ``model_id`` falls back to the snapshot directory name and
    ``revision`` to :data:`UNRESOLVED_REVISION` when the source does not
    carry them — labels for provenance only; structural identity
    (``model_signature_id``) never includes them (§7).
    """
    hf_id = getattr(config, "name_or_path", None)
    model_id = source.model_id or (str(hf_id) if hf_id else source.path.name)
    return ModelReference(
        model_id=model_id,
        revision=source.revision or UNRESOLVED_REVISION,
    )


def characterize_model_source(
    source: ModelSource,
    *,
    dtype: str,
    quantization: str | None = None,
    model_registry: AdapterRegistry | None = None,
    profiling_registry: ProfilingAdapterRegistry | None = None,
) -> ModelCharacterization:
    """Static characterization of one local model snapshot (spec §17)."""
    config = load_model_config(source)
    phase0_adapter = resolve_adapter_for_source(
        source, model_registry if model_registry is not None else default_registry()
    )
    layout = phase0_adapter.inspect(source)
    profiling_adapter = resolve_profiling_adapter(layout, config, profiling_registry)
    model = model_reference_for(source, config)
    return profiling_adapter.characterize(
        layout, config, model, dtype=dtype, quantization=quantization
    )
