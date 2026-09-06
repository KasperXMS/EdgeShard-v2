"""Shared profiling fixtures: real torch.export graphs of tiny Qwen2 layers.

Export is slow enough that the graphs are built once per module and shared
between the extractor and normalizer tests; the tiny fixture models come
from the root ``tests/conftest.py`` (session-scoped, read-only).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from edgeshard.profiling.operator.extractor import RawOperatorGraph, TorchExportExtractor


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
