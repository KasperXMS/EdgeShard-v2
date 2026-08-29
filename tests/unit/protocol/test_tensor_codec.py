"""Tensor codec: safetensors bundle roundtrips preserve dtype/shape (spec 5.5)."""

from __future__ import annotations

import pytest
import torch

from edgeshard.protocol.domain import ProtocolError
from edgeshard.protocol.tensor_codec import decode_tensors, encode_tensors


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64]
)
def test_roundtrip_preserves_dtype_shape_and_values(dtype: torch.dtype) -> None:
    tensor = torch.arange(14, dtype=dtype).reshape(1, 7, 2)
    decoded = decode_tensors(encode_tensors({"x": tensor}))

    assert set(decoded) == {"x"}
    result = decoded["x"]
    assert result.dtype is dtype
    assert result.shape == tensor.shape
    assert result.device == torch.device("cpu")
    assert torch.equal(result, tensor)


def test_roundtrip_carries_multiple_tensors() -> None:
    tensors = {
        "hidden_states": torch.randn(1, 5, 8),
        "positions": torch.arange(5).unsqueeze(0),
    }
    decoded = decode_tensors(encode_tensors(tensors))

    assert set(decoded) == set(tensors)
    for name, tensor in tensors.items():
        assert torch.equal(decoded[name], tensor)


def test_empty_bundle_roundtrip() -> None:
    assert encode_tensors({}) == b""
    assert decode_tensors(b"") == {}


def test_malformed_bundle_fails_explicitly() -> None:
    with pytest.raises(ProtocolError, match="invalid tensor bundle"):
        decode_tensors(b"garbage-bytes")

    truncated = encode_tensors({"x": torch.zeros(4)})[:5]
    with pytest.raises(ProtocolError, match="invalid tensor bundle"):
        decode_tensors(truncated)
