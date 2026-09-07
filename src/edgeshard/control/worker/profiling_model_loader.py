"""MODEL-session checkpoint loading for the Worker runner (spec §38).

``PrepareProfilingSession(kind=MODEL)`` does the expensive static work once:
resolve the requested :class:`ModelReference` to a local ModelStore
snapshot, load the checkpoint onto the session device, enumerate layers and
profile modules through the family adapter, extract the representative
operator signatures, and package everything as
:class:`~edgeshard.profiling.domain.session.ModelSessionFacts` for the
Master's planner (§47) — plus the live handles the layer/module profilers
need per case (§21, §23). Nothing here benchmarks (§40).

The loader is a Protocol so runner tests substitute a stub; the default
:class:`TorchModelSessionLoader` is the only torch/HF-touching piece of the
Worker profiling plane. All failures are typed :class:`ProfilingError`s
(§42) — a model the Worker cannot resolve, load, characterize or export is
a preparation *failure* with its category intact, never a guessed fact or a
silent skip.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from edgeshard.cluster.inventory import ModelAvailability, ModelInventoryEntry
from edgeshard.model.adapters.base import load_model_config
from edgeshard.model.adapters.registry import default_registry, resolve_adapter_for_source
from edgeshard.model.layout import ModelLayout
from edgeshard.model.source import ModelSource
from edgeshard.profiling.domain.experiment import (
    ModelCaseSpec,
    ProfilingErrorCategory,
)
from edgeshard.profiling.domain.session import (
    LayerEntry,
    ModelSessionFacts,
    ModuleEntry,
    ProfilingSessionRequest,
)
from edgeshard.profiling.domain.signature import (
    ModuleSignature,
    OperatorSignature,
    ProfilingGranularity,
    TransformerLayerSignature,
)
from edgeshard.profiling.dtypes import torch_dtype
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.model.adapters.base import (
    LayerReference,
    ModelProfilingAdapter,
    ProfileModule,
    resolve_profiling_adapter,
)
from edgeshard.profiling.model.characterization import model_reference_for
from edgeshard.profiling.model.layer_profiler import transformer_layer_signature
from edgeshard.profiling.operator.extractor import TorchExportExtractor
from edgeshard.profiling.operator.normalizer import (
    OperatorNormalizer,
    unique_operator_signatures,
)
from edgeshard.runtime.model_store import ModelStore

logger = logging.getLogger("worker.profiling.model_loader")

MODEL_EXTRACTION_SEQUENCE_LENGTH = 128
"""Sequence length of the representative layer export at prepare time.

Operator signatures embed their shapes, so the extraction length defines
the operator workload dimensions the Master plans from (§47 step 2). 128 is
the smallest default prefill length (§21): small enough to keep prepare
fast on CPU, canonical enough that repeated preparations agree.
"""

MODEL_EXTRACTION_SEED = 0
"""Fixed seed of the extraction inputs: signatures are shape facts, and a
fixed seed makes repeated prepares produce identical exports (§6)."""


def resolve_model_source(
    model_id: str,
    revision: str | None,
    models: tuple[ModelInventoryEntry, ...],
    store: ModelStore,
) -> ModelSource:
    """Resolve a requested model reference to one READY local snapshot.

    Matching is explicit (§52.2 — never a fuzzy fallback): ``model_id`` must
    match an inventory entry, a *specific* requested revision (anything but
    ``None``/the ``"local"`` provenance label) must match the entry's
    revision, only ``READY`` entries are loadable, and the match must be
    unique — two snapshots claiming the same identity are a store
    inconsistency that fails loudly instead of picking one (§47).
    """
    specific_revision = revision is not None and revision not in ("", "local")
    candidates = [
        entry
        for entry in models
        if entry.model_id == model_id
        and entry.status is ModelAvailability.READY
        and (not specific_revision or entry.revision == revision)
    ]
    if not candidates:
        raise ProfilingError(
            ProfilingErrorCategory.UNSUPPORTED_MODEL,
            f"model {model_id!r} (revision {revision!r}) has no READY snapshot "
            "in this worker's model store",
            {"model_id": model_id, "revision": revision or ""},
        )
    if len(candidates) > 1:
        names = sorted(entry.local_name for entry in candidates)
        raise ProfilingError(
            ProfilingErrorCategory.UNSUPPORTED_MODEL,
            f"model {model_id!r} (revision {revision!r}) resolves to multiple "
            f"local snapshots: {names!r}; the store must hold exactly one",
            {"model_id": model_id, "local_names": ", ".join(names)},
        )
    entry = candidates[0]
    return ModelSource(
        path=store.host_path(entry.local_name),
        model_id=entry.model_id,
        revision=entry.revision,
    )


@dataclass(frozen=True)
class LoadedModelSession:
    """Everything a prepared MODEL session holds between load and cleanup.

    ``facts`` is the static tree returned to the Master (§47 planning
    input); the live handles (``model``, ``layers``, ``modules``) stay
    Worker-side and never cross the wire (§41). ``modules`` is flattened in
    layer-enumeration order so signature lookup is deterministic.
    """

    source: ModelSource
    model: nn.Module
    layout: ModelLayout
    adapter: ModelProfilingAdapter
    layers: tuple[LayerReference, ...]
    modules: tuple[ProfileModule, ...]
    facts: ModelSessionFacts

    def layer_at(self, index: int) -> LayerReference:
        for layer in self.layers:
            if layer.index == index:
                return layer
        raise ProfilingError(
            ProfilingErrorCategory.UNSUPPORTED_GRANULARITY,
            f"layer_index {index} is outside this session's enumeration "
            f"(0..{len(self.layers) - 1})",
            {"layer_index": index, "num_layers": len(self.layers)},
        )

    def module_for(self, signature: ModuleSignature) -> ProfileModule:
        """The first module whose signature matches, in layer order.

        Standard-decoder layers are structurally identical, so a module
        signature identifies one workload class, not one physical instance
        (§23); the first match is the canonical benchmark target — a
        defined v1 policy, not a guess between differing candidates.
        """
        for module in self.modules:
            if module.signature == signature:
                return module
        raise ProfilingError(
            ProfilingErrorCategory.UNSUPPORTED_GRANULARITY,
            "no module of the loaded checkpoint matches the case's module "
            f"signature (kind={signature.kind.value!r})",
            {"module_kind": signature.kind.value},
        )

    def close(self) -> None:
        """Checkpoint cleanup (§38; §37: never a per-case load/destroy).

        The handle is frozen, so the checkpoint itself is released when the
        session record drops this object (``model_handle``/``model_cleanup``
        are cleared on close); this returns the freed device blocks to the
        allocator so the next session or runtime sees the memory back.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ModelSessionLoader(Protocol):
    """Loads one MODEL session's checkpoint and static facts (§38)."""

    def load(
        self, request: ProfilingSessionRequest, source: ModelSource
    ) -> LoadedModelSession:
        """Load ``source`` per the session request; typed failures only (§42)."""
        ...


class TorchModelSessionLoader:
    """Default loader: HF checkpoint → enumerated facts + live handles.

    ``device`` is the torch device the session's single lease resolved to;
    the checkpoint is loaded on CPU (snapshot weights are read once) and
    moved to the target device before any enumeration, so observed weight
    dtypes/devices are the ones every case benchmarks against (§21).
    """

    def __init__(self, *, device: torch.device | None = None) -> None:
        self._device = device if device is not None else torch.device("cpu")

    def load(
        self, request: ProfilingSessionRequest, source: ModelSource
    ) -> LoadedModelSession:
        if request.model is None or request.dtype is None:
            raise ValueError("model session requests carry model and dtype (§38)")
        path = source.ensure_local()
        config = load_model_config(source)
        layout = resolve_adapter_for_source(source, default_registry()).inspect(source)
        adapter = resolve_profiling_adapter(layout, config)
        dtype = torch_dtype(request.dtype)  # label validity first, never guessed
        try:
            model: nn.Module = (
                AutoModelForCausalLM.from_pretrained(path).eval().to(dtype=dtype)
            )
            model = model.to(self._device)
        except ProfilingError:
            raise
        except Exception as exc:
            raise ProfilingError(
                ProfilingErrorCategory.UNSUPPORTED_MODEL,
                f"checkpoint at {path} failed to load as {request.dtype}: {exc}",
                {"path": str(path), "dtype": request.dtype},
            ) from exc

        reference = model_reference_for(source, config)
        characterization = adapter.characterize(
            layout, config, reference, dtype=request.dtype, quantization=request.quantization
        )
        layers = adapter.enumerate_transformer_layers(model, layout)
        layer_signature = transformer_layer_signature(characterization)
        modules: list[ProfileModule] = []
        module_entries: list[ModuleEntry] = []
        for layer in layers:
            for module in adapter.enumerate_profile_modules(
                layer, layout, quantization=characterization.quantization
            ):
                modules.append(module)
                module_entries.append(
                    ModuleEntry(
                        name=module.name,
                        module_path=module.module_path,
                        kind=module.kind,
                        layer_index=layer.index,
                        signature=module.signature,
                    )
                )
        operator_signatures = self._extract_operator_signatures(
            request, adapter, model, layers[0], layout, layer_signature
        )
        facts = ModelSessionFacts(
            characterization=characterization,
            layer_entries=tuple(
                LayerEntry(
                    index=layer.index,
                    module_path=layer.module_path,
                    signature=layer_signature,
                )
                for layer in layers
            ),
            module_entries=tuple(module_entries),
            operator_signatures=operator_signatures,
        )
        logger.info(
            "loaded model session %s: %d layers, %d modules, %d operator signatures",
            reference.model_id,
            len(layers),
            len(modules),
            len(operator_signatures),
        )
        return LoadedModelSession(
            source=source,
            model=model,
            layout=layout,
            adapter=adapter,
            layers=layers,
            modules=tuple(modules),
            facts=facts,
        )

    def _extract_operator_signatures(
        self,
        request: ProfilingSessionRequest,
        adapter: ModelProfilingAdapter,
        model: nn.Module,
        layer: LayerReference,
        layout: ModelLayout,
        layer_signature: TransformerLayerSignature,
    ) -> tuple[OperatorSignature, ...]:
        """Deduplicated operator signatures of the representative layer (§20).

        Standard-decoder layers are structurally identical, so one layer's
        export covers the stack's operator vocabulary; the export uses the
        adapter's own shape-correct inputs with a fixed seed, and extraction
        failures stay typed (``EXPORT_FAILED``) preparation failures (§42).
        """
        extraction_case = ModelCaseSpec(
            granularity=ProfilingGranularity.TRANSFORMER_LAYER,
            device_ids=request.device_ids,
            dtype=str(request.dtype),
            model=request.model,
            layer_signature=layer_signature,
            batch_size=1,
            sequence_length=MODEL_EXTRACTION_SEQUENCE_LENGTH,
        )
        inputs = adapter.build_layer_inputs(
            extraction_case, model, layer, layout, seed=MODEL_EXTRACTION_SEED
        )
        graph = TorchExportExtractor().extract(layer.layer, (), dict(inputs))
        normalized = OperatorNormalizer().normalize(graph)
        return unique_operator_signatures([normalized])


__all__ = [
    "MODEL_EXTRACTION_SEED",
    "MODEL_EXTRACTION_SEQUENCE_LENGTH",
    "LoadedModelSession",
    "ModelSessionLoader",
    "TorchModelSessionLoader",
    "resolve_model_source",
]
