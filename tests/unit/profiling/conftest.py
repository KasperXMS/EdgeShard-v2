"""Shared profiling fixtures: real torch.export graphs of tiny Qwen2 layers.

Export is slow enough that the graphs are built once per module and shared
between the extractor and normalizer tests; the tiny fixture models come
from the root ``tests/conftest.py`` (session-scoped, read-only).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from edgeshard.model.adapters.registry import default_registry, resolve_adapter_for_source
from edgeshard.model.layout import ModelLayout
from edgeshard.model.source import ModelSource
from edgeshard.profiling.domain.model import ModelCharacterization
from edgeshard.profiling.model.adapters.qwen import QwenProfilingAdapter
from edgeshard.profiling.model.characterization import characterize_model_source
from edgeshard.profiling.operator.extractor import RawOperatorGraph, TorchExportExtractor


def _tiny_qwen2_source(tiny_qwen2_dir: Path) -> ModelSource:
    return ModelSource(path=tiny_qwen2_dir, model_id="tiny/qwen2")


@pytest.fixture(scope="session")
def tiny_qwen2_checkpoint(tiny_qwen2_dir: Path) -> nn.Module:
    """Loaded tiny Qwen2 model in eval mode; treat as read-only."""
    return AutoModelForCausalLM.from_pretrained(tiny_qwen2_dir).eval()


@pytest.fixture(scope="session")
def tiny_qwen2_layout(tiny_qwen2_dir: Path) -> ModelLayout:
    source = _tiny_qwen2_source(tiny_qwen2_dir)
    return resolve_adapter_for_source(source, default_registry()).inspect(source)


@pytest.fixture(scope="session")
def qwen2_adapter() -> QwenProfilingAdapter:
    return QwenProfilingAdapter()


@pytest.fixture(scope="session")
def qwen2_characterization(tiny_qwen2_dir: Path) -> ModelCharacterization:
    return characterize_model_source(_tiny_qwen2_source(tiny_qwen2_dir), dtype="fp32")


@pytest.fixture(scope="module")
def qwen2_layer_exports(tiny_qwen2_dir: Path) -> tuple[RawOperatorGraph, ...]:
    """Export graphs of layers 0 and 1 of the tiny Qwen2 model.

    The layers are structurally identical, so their normalized signatures
    must deduplicate to one set (§20) — that is what the normalizer tests
    assert on real extractor output.
    """
    model = AutoModelForCausalLM.from_pretrained(tiny_qwen2_dir).eval()
    extractor = TorchExportExtractor()
    graphs: list[RawOperatorGraph] = []
    for layer in (model.model.layers[0], model.model.layers[1]):
        hidden = torch.randn(1, 8, model.config.hidden_size)
        positions = torch.arange(8).unsqueeze(0)
        with torch.no_grad():
            cos, sin = model.model.rotary_emb(hidden, positions)
        graphs.append(
            extractor.extract(layer, (hidden,), {"position_embeddings": (cos, sin)})
        )
    return tuple(graphs)
