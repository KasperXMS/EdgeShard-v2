"""Default registry tests: automatic model detection (0A gate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgeshard.model.adapters.llama import LlamaAdapter
from edgeshard.model.adapters.qwen2 import Qwen2Adapter
from edgeshard.model.adapters.registry import default_registry, resolve_adapter_for_source
from edgeshard.model.errors import UnsupportedArchitectureError
from edgeshard.model.source import ModelSource


def test_default_registry_model_types() -> None:
    assert default_registry().model_types == ("llama", "qwen2")


def test_tiny_llama_resolves_automatically(tiny_llama_source: ModelSource) -> None:
    adapter = resolve_adapter_for_source(tiny_llama_source, default_registry())
    assert isinstance(adapter, LlamaAdapter)


def test_tiny_qwen2_resolves_automatically(tiny_qwen2_source: ModelSource) -> None:
    adapter = resolve_adapter_for_source(tiny_qwen2_source, default_registry())
    assert isinstance(adapter, Qwen2Adapter)


def test_unsupported_model_fails_explicitly(tmp_path: Path) -> None:
    """Unknown architectures must fail explicitly, never guess (spec 9.4)."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gpt2",
                "architectures": ["GPT2LMHeadModel"],
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 16,
            }
        )
    )
    with pytest.raises(UnsupportedArchitectureError, match="gpt2"):
        resolve_adapter_for_source(ModelSource(path=tmp_path), default_registry())
