"""Shared fixtures: tiny locally generated HF models (spec 25.1).

Tier 1 tests must not depend on the internet; all fixtures are generated
locally with small hidden sizes and saved with safetensors.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from edgeshard.model.source import ModelSource

TINY_LLAMA_CONFIG = LlamaConfig(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
)

TINY_QWEN2_CONFIG = Qwen2Config(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
)


@pytest.fixture(scope="session")
def tiny_llama_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A saved tiny Llama snapshot; treat as read-only (copy to mutate)."""
    directory = tmp_path_factory.mktemp("tiny-llama")
    LlamaForCausalLM(TINY_LLAMA_CONFIG).save_pretrained(directory, safe_serialization=True)
    return directory


@pytest.fixture
def tiny_llama_source(tiny_llama_dir: Path) -> ModelSource:
    return ModelSource(path=tiny_llama_dir, model_id="tiny/llama")


@pytest.fixture(scope="session")
def tiny_llama_tied_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny Llama snapshot with tied word embeddings (spec 11.3)."""
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=True,
    )
    directory = tmp_path_factory.mktemp("tiny-llama-tied")
    LlamaForCausalLM(config).save_pretrained(directory, safe_serialization=True)
    return directory


@pytest.fixture(scope="session")
def sharded_llama_dir(tmp_path_factory: pytest.TempPathFactory, tiny_llama_dir: Path) -> Path:
    """Tiny Llama checkpoint split into two files plus index.json (spec 11.2)."""
    directory = tmp_path_factory.mktemp("tiny-llama-sharded")
    shutil.copy(tiny_llama_dir / "config.json", directory / "config.json")

    single_file = tiny_llama_dir / "model.safetensors"
    with safe_open(single_file, framework="pt") as checkpoint:
        tensor_names = list(checkpoint.keys())
        tensors = {name: checkpoint.get_tensor(name) for name in tensor_names}

    names = sorted(tensors)
    midpoint = len(names) // 2
    file_a = "model-00001-of-00002.safetensors"
    file_b = "model-00002-of-00002.safetensors"
    tensors_a = {name: tensors[name] for name in names[:midpoint]}
    tensors_b = {name: tensors[name] for name in names[midpoint:]}
    save_file(tensors_a, str(directory / file_a))
    save_file(tensors_b, str(directory / file_b))

    weight_map = {name: (file_a if name in tensors_a else file_b) for name in names}
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    return directory


@pytest.fixture(scope="session")
def tiny_qwen2_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A saved tiny Qwen2 snapshot; treat as read-only (copy to mutate)."""
    directory = tmp_path_factory.mktemp("tiny-qwen2")
    Qwen2ForCausalLM(TINY_QWEN2_CONFIG).save_pretrained(directory, safe_serialization=True)
    return directory


@pytest.fixture
def tiny_qwen2_source(tiny_qwen2_dir: Path) -> ModelSource:
    return ModelSource(path=tiny_qwen2_dir, model_id="tiny/qwen2")
