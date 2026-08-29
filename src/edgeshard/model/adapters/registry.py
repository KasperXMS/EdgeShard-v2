"""Adapter registry and config-based resolution (spec 9.1).

Resolution uses ``config.model_type`` first and falls back to
``config.architectures``. Unknown models fail explicitly with
:class:`UnsupportedArchitectureError`; the registry never guesses (spec 9.4).
"""

from __future__ import annotations

from collections.abc import Sequence

from edgeshard.model.adapters.base import ModelAdapter, load_model_config
from edgeshard.model.adapters.llama import LlamaAdapter
from edgeshard.model.adapters.qwen2 import Qwen2Adapter
from edgeshard.model.errors import UnsupportedArchitectureError
from edgeshard.model.source import ModelSource


class AdapterRegistry:
    """Maps ``model_type`` strings and architecture names to adapters."""

    def __init__(self) -> None:
        self._by_model_type: dict[str, ModelAdapter] = {}
        self._by_architecture: dict[str, ModelAdapter] = {}

    @property
    def model_types(self) -> tuple[str, ...]:
        """Registered ``model_type`` values, sorted."""
        return tuple(sorted(self._by_model_type))

    def register(self, adapter: ModelAdapter) -> None:
        """Register an adapter under its ``model_type`` and architecture names."""
        model_type = adapter.model_type
        if model_type in self._by_model_type:
            raise ValueError(f"adapter for model_type {model_type!r} is already registered")
        self._by_model_type[model_type] = adapter
        for architecture in adapter.architectures:
            if architecture in self._by_architecture:
                raise ValueError(
                    f"adapter for architecture {architecture!r} is already registered"
                )
            self._by_architecture[architecture] = adapter

    def resolve(
        self,
        model_type: str | None,
        architectures: Sequence[str] = (),
    ) -> ModelAdapter:
        """Resolve an adapter, preferring ``model_type`` over architecture names."""
        if model_type is not None:
            adapter = self._by_model_type.get(model_type)
            if adapter is not None:
                return adapter
        for architecture in architectures:
            adapter = self._by_architecture.get(architecture)
            if adapter is not None:
                return adapter
        raise UnsupportedArchitectureError(
            f"no adapter registered for model_type={model_type!r}, "
            f"architectures={list(architectures)!r} "
            f"(registered model types: {list(self.model_types)})"
        )


def resolve_adapter_for_source(source: ModelSource, registry: AdapterRegistry) -> ModelAdapter:
    """Detect the right adapter for a local snapshot without loading weights."""
    config = load_model_config(source)
    architectures: Sequence[str] = tuple(getattr(config, "architectures", None) or ())
    return registry.resolve(getattr(config, "model_type", None), architectures)


def default_registry() -> AdapterRegistry:
    """Registry with all built-in Phase 0 adapters registered."""
    registry = AdapterRegistry()
    registry.register(LlamaAdapter())
    registry.register(Qwen2Adapter())
    return registry
