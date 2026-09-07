"""Synthetic operator microbenchmark workloads (spec §25-§27).

An operator workload materializes the tensors of one
:class:`~edgeshard.profiling.domain.signature.OperatorSignature` and
executes exactly that operation — operator profiling never runs a full
model (P2E DoD). The §25.1 protocol keeps workloads backend-shaped:
``prepare`` binds a signature to a device, ``run_once`` performs one
iteration, ``cleanup`` releases the materialized tensors.

Rules pinned here:

- only the *production-relevant* backend path is benchmarked (§26-§27):
  GEMM runs ``torch.mm`` with the recorded transpose layout (the cuBLAS
  call shape a production ``linear`` produces), attention runs torch SDPA
  — never a generic implementation standing in for a fused one;
- signature facts are never extended by guesses (§52.2): a workload whose
  declared dtype is ``unknown``, whose norm variant is ``OTHER``, or whose
  attention phase does not have exact synthetic semantics (chunked
  prefill, multi-token decode) fails typed (``UNSUPPORTED_OPERATOR``)
  instead of approximating;
- decode attention is benchmarked from its own signature (``q_len == 1``,
  full ``kv_len``) — it is never derived from prefill (§24).
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.signature import (
    AttentionSignature,
    GemmSignature,
    InferencePhase,
    NormSignature,
    NormVariant,
    OperatorSignature,
)
from edgeshard.profiling.dtypes import UNKNOWN_DTYPE, torch_dtype
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.operator.planning import parameters_dtype

RMS_NORM_EPS = 1e-6
"""Epsilon of the eager RMSNorm sequence (the common production default)."""


def typed_parameters[T](signature: OperatorSignature, expected: type[T]) -> T:
    """Narrow a signature's parameters to the workload's typed shape.

    The domain already couples kinds to parameter types; a mismatch here
    means a workload was resolved for the wrong signature and fails typed
    rather than reading foreign fields (§42).
    """
    if not isinstance(signature.parameters, expected):
        raise ProfilingError(
            ProfilingErrorCategory.UNSUPPORTED_OPERATOR,
            f"{expected.__name__} workload requires {expected.__name__} parameters, "
            f"got {type(signature.parameters).__name__}",
            {"kind": signature.kind.value},
        )
    return signature.parameters


def workload_dtype(label: str) -> torch.dtype:
    """Torch dtype for a signature's declared measurement dtype.

    An ``unknown`` label is an extractor fact, not a licensable guess:
    the workload cannot materialize inputs for it and fails typed (§52.2).
    """
    if label == UNKNOWN_DTYPE:
        raise ProfilingError(
            ProfilingErrorCategory.UNSUPPORTED_OPERATOR,
            "operator signature declares an unknown dtype; cannot materialize "
            "benchmark inputs without guessing (§52.2)",
            {"dtype": label},
        )
    return torch_dtype(label)


@runtime_checkable
class OperatorWorkload(Protocol):
    """One synthetic operator benchmark (spec §25.1).

    ``backend_family`` names the logical execution primitive the workload
    measures; the profiler refuses to run it against a signature recorded
    from a different backend family (§27 — measurements are only reusable
    within the same logical primitive).
    """

    backend_family: ClassVar[str]

    def prepare(self, signature: OperatorSignature, device: torch.device) -> None: ...

    def run_once(self) -> None: ...

    def cleanup(self) -> None: ...


class GemmWorkload:
    """``torch.mm`` at the signature's exact (M, N, K) and layout (§26).

    The transpose flags reproduce the memory layout the production call
    site presents to cuBLAS: a linear-shaped GEMM stores its weight as
    ``(N, K)`` and is benchmarked as ``mm(x, w.t())`` — the same op shape,
    not an equivalent-but-different kernel.
    """

    backend_family: ClassVar[str] = "torch"

    def __init__(self) -> None:
        self._left: torch.Tensor | None = None
        self._right: torch.Tensor | None = None

    def prepare(self, signature: OperatorSignature, device: torch.device) -> None:
        parameters = typed_parameters(signature, GemmSignature)
        dtype = workload_dtype(parameters.dtype)
        left_storage = (
            (parameters.k, parameters.m) if parameters.transpose_a
            else (parameters.m, parameters.k)
        )
        right_storage = (
            (parameters.n, parameters.k) if parameters.transpose_b
            else (parameters.k, parameters.n)
        )
        left = torch.randn(left_storage, dtype=dtype)
        right = torch.randn(right_storage, dtype=dtype)
        self._left = (left.t() if parameters.transpose_a else left).to(device)
        self._right = (right.t() if parameters.transpose_b else right).to(device)

    def run_once(self) -> None:
        left, right = self._left, self._right
        if left is None or right is None:
            raise RuntimeError("workload has not been prepared")
        with torch.no_grad():
            torch.mm(left, right)

    def cleanup(self) -> None:
        self._left = None
        self._right = None


class AttentionWorkload:
    """Torch SDPA at the signature's exact dimensions and phase (§27).

    Production attention for the supported decoder families is torch SDPA,
    so the microbenchmark is torch SDPA — including native GQA via
    ``enable_gqa`` rather than a materialized repeat. Phase semantics are
    exact or typed-unsupported: prefill is causal with ``q_len == kv_len``;
    decode is one query token attending to the full cache length.
    """

    backend_family: ClassVar[str] = "torch"

    def __init__(self) -> None:
        self._query: torch.Tensor | None = None
        self._key: torch.Tensor | None = None
        self._value: torch.Tensor | None = None
        self._causal = False

    def prepare(self, signature: OperatorSignature, device: torch.device) -> None:
        parameters = typed_parameters(signature, AttentionSignature)
        dtype = workload_dtype(parameters.dtype)
        if parameters.phase is InferencePhase.PREFILL:
            if parameters.q_len != parameters.kv_len:
                raise ProfilingError(
                    ProfilingErrorCategory.UNSUPPORTED_OPERATOR,
                    "prefill attention microbenchmark requires q_len == kv_len; "
                    "chunked prefill masking is not synthesized in v1 (§24)",
                    {"q_len": parameters.q_len, "kv_len": parameters.kv_len},
                )
            self._causal = True
        else:
            if parameters.q_len != 1:
                raise ProfilingError(
                    ProfilingErrorCategory.UNSUPPORTED_OPERATOR,
                    "decode attention microbenchmark requires q_len == 1, "
                    f"got {parameters.q_len} (§24: decode is never approximated)",
                    {"q_len": parameters.q_len},
                )
            self._causal = False
        self._query = torch.randn(
            (parameters.batch_size, parameters.num_heads, parameters.q_len,
             parameters.head_dim),
            dtype=dtype,
        ).to(device)
        key_value_shape = (
            parameters.batch_size, parameters.num_kv_heads, parameters.kv_len,
            parameters.head_dim,
        )
        self._key = torch.randn(key_value_shape, dtype=dtype).to(device)
        self._value = torch.randn(key_value_shape, dtype=dtype).to(device)

    def run_once(self) -> None:
        query, key, value = self._query, self._key, self._value
        if query is None or key is None or value is None:
            raise RuntimeError("workload has not been prepared")
        with torch.no_grad():
            F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=self._causal,
                enable_gqa=True,
            )

    def cleanup(self) -> None:
        self._query = None
        self._key = None
        self._value = None


class NormWorkload:
    """Normalization at the signature's shape and variant (§25).

    RMS reproduces the eager production sequence of the supported decoder
    families (fp32 variance, rsqrt, cast back, weight scale) — norm
    variants of real models are bandwidth-bound eager kernels, not fused
    primitives. LAYER runs native ``F.layer_norm``. ``OTHER`` has no
    defined synthetic semantics and fails typed.
    """

    backend_family: ClassVar[str] = "torch"

    def __init__(self) -> None:
        self._input: torch.Tensor | None = None
        self._weight: torch.Tensor | None = None
        self._variant: NormVariant | None = None

    def prepare(self, signature: OperatorSignature, device: torch.device) -> None:
        parameters = typed_parameters(signature, NormSignature)
        dtype = workload_dtype(parameters.dtype)
        if parameters.variant is NormVariant.OTHER:
            raise ProfilingError(
                ProfilingErrorCategory.UNSUPPORTED_OPERATOR,
                "norm variant 'other' has no synthetic workload; the normalized "
                "variant must be RMS or LAYER (§19)",
            )
        shape = (
            parameters.batch_size,
            parameters.sequence_length,
            parameters.hidden_size,
        )
        self._input = torch.randn(shape, dtype=dtype).to(device)
        self._weight = torch.ones(parameters.hidden_size, dtype=dtype).to(device)
        self._variant = parameters.variant

    def run_once(self) -> None:
        tensor, weight, variant = self._input, self._weight, self._variant
        if tensor is None or weight is None or variant is None:
            raise RuntimeError("workload has not been prepared")
        with torch.no_grad():
            if variant is NormVariant.RMS:
                promoted = tensor.float()
                variance = promoted.pow(2).mean(-1, keepdim=True)
                normalized = promoted * torch.rsqrt(variance + RMS_NORM_EPS)
                _ = normalized.to(tensor.dtype) * weight
            else:
                _ = F.layer_norm(tensor, (tensor.shape[-1],), weight)

    def cleanup(self) -> None:
        self._input = None
        self._weight = None
        self._variant = None


__all__ = [
    "AttentionWorkload",
    "GemmWorkload",
    "NormWorkload",
    "OperatorWorkload",
    "parameters_dtype",
    "typed_parameters",
    "workload_dtype",
]
