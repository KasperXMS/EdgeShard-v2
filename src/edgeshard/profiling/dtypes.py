"""Stable dtype labels for profiling signatures (Phase 2).

Signatures carry dtypes as *strings* so the domain stays torch-free
(P2A DoD). This module is the single place that translates backend dtype
representations into those labels — one vocabulary, no per-module
inventions:

- :func:`dtype_label` maps ``torch.dtype`` (real or fake tensors) to the
  project label (``bf16``, ``fp16``, ``fp32``, ...);
- :func:`kineto_dtype_label` maps the profiler's scalar-type strings
  (``float``, ``BFloat16``, ``Scalar``, ``""``, ...) into the same
  vocabulary. A label Kineto does not provide (``""`` for non-tensor
  arguments) becomes the explicit ``unknown`` — never a guessed dtype
  (§52.2). The profiler fallback is structural discovery (§18.2), so an
  ``unknown`` label honestly marks reduced fidelity.
"""

from __future__ import annotations

import torch

_TORCH_LABELS: dict[torch.dtype, str] = {
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.float32: "fp32",
    torch.float64: "fp64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

_KINETO_LABELS: dict[str, str] = {
    "float": "fp32",
    "double": "fp64",
    "half": "fp16",
    "bfloat16": "bf16",
    "float8_e4m3fn": "fp8_e4m3",
    "float8_e5m2": "fp8_e5m2",
    "long": "int64",
    "int": "int32",
    "short": "int16",
    "char": "int8",
    "byte": "uint8",
    "bool": "bool",
}

UNKNOWN_DTYPE = "unknown"


def dtype_label(dtype: torch.dtype) -> str:
    """Project label for a torch dtype; unlisted dtypes keep their name."""
    label = _TORCH_LABELS.get(dtype)
    if label is not None:
        return label
    return str(dtype).removeprefix("torch.")


def kineto_dtype_label(raw: str) -> str:
    """Project label for a profiler-reported scalar type string."""
    if not raw:
        return UNKNOWN_DTYPE
    lowered = raw.lower().removeprefix("c10::")
    return _KINETO_LABELS.get(lowered, lowered)
