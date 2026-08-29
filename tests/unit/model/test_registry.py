"""Adapter registry tests: registration and config-based resolution (spec 9.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgeshard.model.adapters.base import load_model_config
from edgeshard.model.adapters.registry import AdapterRegistry, resolve_adapter_for_source
from edgeshard.model.errors import ModelSourceError, UnsupportedArchitectureError
from edgeshard.model.layout import ModelLayout
from edgeshard.model.source import ModelSource


class StubLlamaAdapter:
    model_type = "llama"
    architectures = ("LlamaForCausalLM",)

    def inspect(self, source: ModelSource) -> ModelLayout:
        raise NotImplementedError


class StubQwenAdapter:
    model_type = "qwen2"
    architectures = ("Qwen2ForCausalLM",)

    def inspect(self, source: ModelSource) -> ModelLayout:
        raise NotImplementedError


def _registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register(StubLlamaAdapter())
    registry.register(StubQwenAdapter())
    return registry


def _write_llama_config(directory: Path) -> None:
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 64,
                "num_hidden_layers": 4,
            }
        )
    )


def test_register_and_resolve_by_model_type() -> None:
    registry = _registry()
    adapter = registry.resolve("qwen2")
    assert isinstance(adapter, StubQwenAdapter)


def test_resolve_falls_back_to_architectures() -> None:
    registry = _registry()
    adapter = registry.resolve(None, architectures=("LlamaForCausalLM",))
    assert isinstance(adapter, StubLlamaAdapter)


def test_model_type_takes_precedence() -> None:
    registry = _registry()
    adapter = registry.resolve("llama", architectures=("Qwen2ForCausalLM",))
    assert isinstance(adapter, StubLlamaAdapter)


def test_resolve_unknown_raises_explicitly() -> None:
    registry = _registry()
    with pytest.raises(UnsupportedArchitectureError, match="mistral"):
        registry.resolve("mistral", architectures=("MistralForCausalLM",))
    with pytest.raises(UnsupportedArchitectureError):
        registry.resolve(None)


def test_registered_model_types_listed() -> None:
    assert _registry().model_types == ("llama", "qwen2")
    assert AdapterRegistry().model_types == ()


def test_duplicate_model_type_rejected() -> None:
    registry = AdapterRegistry()
    registry.register(StubLlamaAdapter())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(StubLlamaAdapter())


def test_duplicate_architecture_rejected() -> None:
    registry = AdapterRegistry()
    registry.register(StubLlamaAdapter())

    class Conflicting:
        model_type = "other"
        architectures = ("LlamaForCausalLM",)

        def inspect(self, source: ModelSource) -> ModelLayout:
            raise NotImplementedError

    with pytest.raises(ValueError, match="already registered"):
        registry.register(Conflicting())


def test_load_model_config_reads_config_only(tmp_path: Path) -> None:
    """A directory with config.json and no weight files is sufficient (spec 9.1)."""
    _write_llama_config(tmp_path)
    config = load_model_config(ModelSource(path=tmp_path))
    assert config.model_type == "llama"
    assert config.architectures == ["LlamaForCausalLM"]


def test_load_model_config_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelSourceError):
        load_model_config(ModelSource(path=tmp_path / "missing"))


def test_resolve_adapter_for_source_end_to_end(tmp_path: Path) -> None:
    _write_llama_config(tmp_path)
    adapter = resolve_adapter_for_source(ModelSource(path=tmp_path), _registry())
    assert isinstance(adapter, StubLlamaAdapter)
