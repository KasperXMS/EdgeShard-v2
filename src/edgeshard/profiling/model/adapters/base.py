"""ModelProfilingAdapter protocol and the standard decoder-LM base.

The adapter protocol follows spec §16.1. Characterization and
enumeration consume the Phase 0 :class:`ModelLayout` — never a fresh
inspection of HF configs for backbone structure. The HF config is read
only for facts the layout does not carry (``vocab_size``, an explicit
``head_dim`` override, the quantization method).

``build_layer_inputs``/``build_module_inputs`` (§16.1) belong to the
protocol but are implemented with the P2D layer/module profilers, which
define the execution conventions they must feed.

Unknown models fail with a typed ``UNSUPPORTED_MODEL`` error (§16.1);
unexpected layer children are preserved as ``ModuleKind.OTHER`` — module
enumeration never drops structure silently (§19 in spirit: what exists
is recorded, even when unrecognized).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import torch.nn as nn
from transformers import PretrainedConfig

from edgeshard.model.layout import ModelLayout
from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.hashing import normalized_items
from edgeshard.profiling.domain.model import (
    ModelCharacterization,
    ModelReference,
    ModelStage,
    StageKind,
)
from edgeshard.profiling.domain.signature import ModuleKind, ModuleSignature
from edgeshard.profiling.dtypes import dtype_label
from edgeshard.profiling.errors import ProfilingError


@dataclass(frozen=True)
class LayerReference:
    """One enumerated Transformer layer of a model instance.

    ``index`` is the position within the enumerated ``ModuleList``: for a
    full model it is the global layer index; for a shard skeleton trimmed
    by Phase 0 ``keep_layer_range`` it is the skeleton-local index
    (global block = ``shard.blocks.start + index``).
    """

    index: int
    module_path: str
    layer: nn.Module


@dataclass(frozen=True)
class ProfileModule:
    """One normalized profile module inside a Transformer layer (§23)."""

    kind: ModuleKind
    name: str
    module_path: str
    module: nn.Module
    signature: ModuleSignature


class ModelProfilingAdapter(Protocol):
    """Family-specific profiling behavior (spec §16.1)."""

    #: ``layout.model_type`` handled by this adapter.
    model_type: ClassVar[str]
    #: ``config.architectures`` entries handled by this adapter.
    architectures: ClassVar[tuple[str, ...]]

    def supports(self, layout: ModelLayout) -> bool:
        """Whether this adapter profiles models with the given layout."""
        ...

    def characterize(
        self,
        layout: ModelLayout,
        config: PretrainedConfig,
        model: ModelReference,
        *,
        dtype: str,
        quantization: str | None = None,
    ) -> ModelCharacterization:
        """Static characterization without any benchmarking (§17)."""
        ...

    def enumerate_transformer_layers(
        self, model: nn.Module, layout: ModelLayout
    ) -> tuple[LayerReference, ...]:
        """The model's Transformer layers in execution order."""
        ...

    def enumerate_profile_modules(
        self, layer: LayerReference, layout: ModelLayout, *, quantization: str | None = None
    ) -> tuple[ProfileModule, ...]:
        """Normalized profile modules of one layer (§23)."""
        ...


def quantization_from_config(config: PretrainedConfig) -> str | None:
    """The config's quantization method label, or ``None`` when unquantized."""
    raw: Any = getattr(config, "quantization_config", None)
    if raw is None:
        return None
    if isinstance(raw, dict):
        method = raw.get("quant_method")
    else:
        method = getattr(raw, "quant_method", None)
    if method is None:
        return None
    method_label = str(method)
    if not method_label:
        raise ProfilingError(
            ProfilingErrorCategory.UNSUPPORTED_MODEL,
            "config carries a quantization_config without a quant_method",
        )
    return method_label


def observed_module_dtype(module: nn.Module) -> str | None:
    """Label of the module's first parameter/buffer dtype, if it has one."""
    for parameter in module.parameters():
        return dtype_label(parameter.dtype)
    for buffer in module.buffers():
        return dtype_label(buffer.dtype)
    return None


class StandardDecoderProfilingAdapter:
    """Profiling adapter for HF decoder-only causal LMs with the standard
    layout (``model.embed_tokens`` / ``model.layers.{i}`` / ``model.norm``
    / ``lm_head``) — the profiling counterpart of Phase 0's
    ``StandardDecoderLMAdapter``.

    Concrete subclasses declare identity (``model_type``,
    ``architectures``, ``architecture_family``) and the layer-child
    vocabulary when it deviates from :attr:`module_kinds`.
    """

    model_type: ClassVar[str]
    architectures: ClassVar[tuple[str, ...]]
    #: Signature-level family label, e.g. ``"qwen2"``.
    architecture_family: ClassVar[str]
    #: Normalization flavor of the family's norm modules.
    norm_variant: ClassVar[str] = "rms"
    #: Layer-child names mapped to normalized module kinds; children not
    #: listed are preserved as ``ModuleKind.OTHER``.
    module_kinds: ClassVar[dict[str, ModuleKind]] = {
        "self_attn": ModuleKind.ATTENTION,
        "mlp": ModuleKind.MLP,
        "input_layernorm": ModuleKind.NORM,
        "post_attention_layernorm": ModuleKind.NORM,
    }

    def supports(self, layout: ModelLayout) -> bool:
        return layout.model_type == self.model_type

    def characterize(
        self,
        layout: ModelLayout,
        config: PretrainedConfig,
        model: ModelReference,
        *,
        dtype: str,
        quantization: str | None = None,
    ) -> ModelCharacterization:
        vocab_size = int(getattr(config, "vocab_size", 0))
        if vocab_size <= 0:
            raise ProfilingError(
                ProfilingErrorCategory.UNSUPPORTED_MODEL,
                f"config for {layout.model_type!r} carries no usable vocab_size",
                {"vocab_size": vocab_size},
            )
        return ModelCharacterization(
            model=model,
            architecture_family=self.architecture_family,
            num_layers=layout.num_blocks,
            hidden_size=layout.hidden_size,
            intermediate_size=layout.intermediate_size,
            vocab_size=vocab_size,
            num_attention_heads=layout.num_attention_heads,
            num_kv_heads=layout.num_kv_heads,
            head_dim=self.head_dim(layout, config),
            dtype=dtype,
            quantization=(
                quantization if quantization is not None else quantization_from_config(config)
            ),
            tied_word_embeddings=layout.tied_word_embeddings,
            stages=self.stage_graph(layout),
        )

    def head_dim(self, layout: ModelLayout, config: PretrainedConfig) -> int | None:
        """Per-head dimension: the config override when present, else derived."""
        override = getattr(config, "head_dim", None)
        if override:
            return int(override)
        if layout.num_attention_heads and layout.hidden_size % layout.num_attention_heads == 0:
            return layout.hidden_size // layout.num_attention_heads
        return None

    def stage_graph(self, layout: ModelLayout) -> tuple[ModelStage, ...]:
        """Homogeneous decoder stack: embed → layers → final norm → head."""
        return (
            ModelStage(kind=StageKind.EMBEDDING),
            ModelStage(kind=StageKind.TRANSFORMER_LAYER_GROUP, layer_count=layout.num_blocks),
            ModelStage(kind=StageKind.FINAL_NORM),
            ModelStage(kind=StageKind.LM_HEAD),
        )

    def enumerate_transformer_layers(
        self, model: nn.Module, layout: ModelLayout
    ) -> tuple[LayerReference, ...]:
        list_path = layout.block_prefix_template.split("{index}")[0].rstrip(".")
        if not list_path:
            raise ProfilingError(
                ProfilingErrorCategory.UNSUPPORTED_MODEL,
                f"layout block prefix template {layout.block_prefix_template!r} "
                "does not identify a layer list",
            )
        try:
            module_list = model.get_submodule(list_path)
        except AttributeError as exc:
            raise ProfilingError(
                ProfilingErrorCategory.UNSUPPORTED_MODEL,
                f"model structure does not match its layout: no module at {list_path!r}",
            ) from exc
        if not isinstance(module_list, nn.ModuleList):
            raise ProfilingError(
                ProfilingErrorCategory.UNSUPPORTED_MODEL,
                f"module at {list_path!r} is {type(module_list).__name__}, "
                "expected a ModuleList of transformer layers",
            )
        layers = tuple(
            LayerReference(
                index=index,
                module_path=f"{list_path}.{index}",
                layer=module_list[index],
            )
            for index in range(len(module_list))
        )
        if not layers:
            raise ProfilingError(
                ProfilingErrorCategory.UNSUPPORTED_MODEL,
                f"model exposes no transformer layers at {list_path!r}",
            )
        return layers

    def enumerate_profile_modules(
        self,
        layer: LayerReference,
        layout: ModelLayout,
        *,
        quantization: str | None = None,
    ) -> tuple[ProfileModule, ...]:
        modules: list[ProfileModule] = []
        for name, module in layer.layer.named_children():
            kind = self.module_kinds.get(name, ModuleKind.OTHER)
            dtype = observed_module_dtype(module)
            if dtype is None:
                raise ProfilingError(
                    ProfilingErrorCategory.UNSUPPORTED_MODEL,
                    f"module {layer.module_path}.{name} exposes no parameter or "
                    "buffer dtype; cannot build a module signature",
                )
            modules.append(
                ProfileModule(
                    kind=kind,
                    name=name,
                    module_path=f"{layer.module_path}.{name}",
                    module=module,
                    signature=ModuleSignature(
                        kind=kind,
                        architecture_family=self.architecture_family,
                        structural_parameters=normalized_items(
                            self.structural_parameters(kind, layout), "structural_parameters"
                        ),
                        dtype=dtype,
                        quantization=quantization,
                    ),
                )
            )
        return tuple(modules)

    def structural_parameters(
        self, kind: ModuleKind, layout: ModelLayout
    ) -> dict[str, int | float | str | bool]:
        """Normalized structural parameters per module kind (§6.2)."""
        if kind is ModuleKind.ATTENTION:
            return {
                "hidden_size": layout.hidden_size,
                "num_heads": layout.num_attention_heads,
                "num_kv_heads": layout.num_kv_heads,
            }
        if kind is ModuleKind.MLP:
            return {
                "hidden_size": layout.hidden_size,
                "intermediate_size": layout.intermediate_size,
            }
        if kind is ModuleKind.NORM:
            return {"hidden_size": layout.hidden_size, "variant": self.norm_variant}
        return {}


class ProfilingAdapterRegistry:
    """Resolves profiling adapters; unknown models fail explicitly (§16.1)."""

    def __init__(self) -> None:
        self._by_model_type: dict[str, ModelProfilingAdapter] = {}
        self._by_architecture: dict[str, ModelProfilingAdapter] = {}

    @property
    def model_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_model_type))

    def register(self, adapter: ModelProfilingAdapter) -> None:
        model_type: str = adapter.model_type
        if model_type in self._by_model_type:
            raise ValueError(f"profiling adapter for model_type {model_type!r} already registered")
        self._by_model_type[model_type] = adapter
        for architecture in adapter.architectures:
            if architecture in self._by_architecture:
                raise ValueError(
                    f"profiling adapter for architecture {architecture!r} already registered"
                )
            self._by_architecture[architecture] = adapter

    def resolve(
        self, model_type: str | None, architectures: tuple[str, ...] = ()
    ) -> ModelProfilingAdapter:
        if model_type is not None:
            adapter = self._by_model_type.get(model_type)
            if adapter is not None:
                return adapter
        for architecture in architectures:
            adapter = self._by_architecture.get(architecture)
            if adapter is not None:
                return adapter
        raise ProfilingError(
            ProfilingErrorCategory.UNSUPPORTED_MODEL,
            f"no profiling adapter for model_type={model_type!r}, "
            f"architectures={list(architectures)!r} "
            f"(registered model types: {list(self.model_types)})",
        )


def default_profiling_registry() -> ProfilingAdapterRegistry:
    """Registry with all built-in Phase 2 v1 profiling adapters."""
    from edgeshard.profiling.model.adapters.llama import LlamaProfilingAdapter
    from edgeshard.profiling.model.adapters.qwen import QwenProfilingAdapter

    registry = ProfilingAdapterRegistry()
    registry.register(QwenProfilingAdapter())
    registry.register(LlamaProfilingAdapter())
    return registry


def resolve_profiling_adapter(
    layout: ModelLayout,
    config: PretrainedConfig,
    registry: ProfilingAdapterRegistry | None = None,
) -> ModelProfilingAdapter:
    """Resolve the profiling adapter for a layout, verifying support.

    The Phase 0 :class:`ModelAdapter` stays responsible for producing the
    layout; this resolution never re-inspects the config for structure.
    """
    active = registry if registry is not None else default_profiling_registry()
    architectures: tuple[str, ...] = tuple(getattr(config, "architectures", None) or ())
    adapter = active.resolve(layout.model_type, architectures)
    if not adapter.supports(layout):
        raise ProfilingError(
            ProfilingErrorCategory.UNSUPPORTED_MODEL,
            f"resolved profiling adapter {type(adapter).__name__} does not support "
            f"layout model_type={layout.model_type!r}",
        )
    return adapter


__all__ = [
    "LayerReference",
    "ModelProfilingAdapter",
    "ProfileModule",
    "ProfilingAdapterRegistry",
    "StandardDecoderProfilingAdapter",
    "default_profiling_registry",
    "observed_module_dtype",
    "quantization_from_config",
    "resolve_profiling_adapter",
]
