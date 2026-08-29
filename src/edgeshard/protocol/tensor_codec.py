"""Tensor bundle codec (spec 5.5, 17.3).

Phase 0 encodes tensor bundles crossing shard boundaries with safetensors:
this intentionally trades performance for dtype correctness, shape
preservation, and BF16 support. Future transports may replace the codec
without changing inference semantics.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from safetensors import SafetensorError
from safetensors.torch import load, save

from edgeshard.protocol.domain import ProtocolError


def encode_tensors(tensors: Mapping[str, torch.Tensor]) -> bytes:
    """Encode named tensors into a safetensors bundle.

    An empty mapping encodes to an empty bundle (``b""``).
    """
    if not tensors:
        return b""
    return save(dict(tensors))


def decode_tensors(data: bytes) -> dict[str, torch.Tensor]:
    """Decode a safetensors bundle produced by :func:`encode_tensors`.

    An empty bundle decodes to an empty mapping. Malformed bundles fail
    explicitly with :class:`ProtocolError`.
    """
    if not data:
        return {}
    try:
        return load(data)
    except SafetensorError as exc:
        raise ProtocolError(f"invalid tensor bundle: {exc}") from exc
