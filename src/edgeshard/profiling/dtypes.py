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

The torch import is *deferred* into the two translation functions that
need real ``torch.dtype`` objects: the label vocabulary itself is pure
data, so torch-free consumers (the §36 payload formula, the Master-side
strategy) can import this module in a control-plane-only deployment
where the ``inference`` extra is not installed.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@cache
def _torch_labels() -> dict[torch.dtype, str]:
    """Torch dtype ↔ label table, built on first torch-facing use."""
    import torch

    return {
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
    label = _torch_labels().get(dtype)
    if label is not None:
        return label
    return str(dtype).removeprefix("torch.")


def kineto_dtype_label(raw: str) -> str:
    """Project label for a profiler-reported scalar type string."""
    if not raw:
        return UNKNOWN_DTYPE
    lowered = raw.lower().removeprefix("c10::")
    return _KINETO_LABELS.get(lowered, lowered)


def torch_dtype(label: str) -> torch.dtype:
    """Torch dtype for a project label (inverse of :func:`dtype_label`).

    Used when building benchmark inputs for a declared measurement dtype
    (P2D). Unknown labels raise ``ValueError`` — an input dtype is never
    guessed (§52.2).
    """
    for dtype, known in _torch_labels().items():
        if known == label:
            return dtype
    raise ValueError(f"unknown dtype label {label!r}")


_BYTES_BY_LABEL: dict[str, int] = {
    "fp64": 8,
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}


def dtype_byte_size(label: str) -> int:
    """Bytes per element for a project dtype label (torch-free).

    Used by the §36 network payload formula (``batch * seq * hidden *
    bytes_per_element``) without importing torch into pure computation.
    Unknown labels raise ``ValueError`` — a payload size is never guessed
    (§52.2).
    """
    size = _BYTES_BY_LABEL.get(label)
    if size is None:
        raise ValueError(f"unknown dtype label {label!r}")
    return size
