"""P2E synthetic operator workload tests (spec §25-§27).

The workloads are pinned at the backend-call level: spies on ``torch.mm``
/ SDPA / ``layer_norm`` / ``rsqrt`` verify that the *production-relevant*
primitive runs with the signature's exact dimensions, transpose layout,
causality, and dtype promotion — a microbenchmark must not substitute a
different logical primitive (§27). Fact gaps (unknown dtype, ``OTHER``
norm variant, chunked prefill, multi-token decode) fail typed instead of
approximating (§24, §52.2).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F

from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.signature import (
    AttentionSignature,
    CustomOperatorParameters,
    GemmSignature,
    InferencePhase,
    NormSignature,
    NormVariant,
    OperatorKind,
    OperatorSignature,
)
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.operator.workloads import (
    AttentionWorkload,
    GemmWorkload,
    NormWorkload,
    OperatorWorkload,
    parameters_dtype,
    typed_parameters,
    workload_dtype,
)

CPU = torch.device("cpu")


def _gemm(**overrides: Any) -> OperatorSignature:
    params: dict[str, Any] = {
        "m": 4,
        "n": 16,
        "k": 8,
        "dtype": "fp32",
        "transpose_a": False,
        "transpose_b": True,
    }
    params.update(overrides)
    return OperatorSignature(
        kind=OperatorKind.GEMM, parameters=GemmSignature(**params), backend_family="torch"
    )


def _attention(**overrides: Any) -> OperatorSignature:
    params: dict[str, Any] = {
        "batch_size": 1,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 16,
        "q_len": 8,
        "kv_len": 8,
        "dtype": "fp32",
        "phase": InferencePhase.PREFILL,
    }
    params.update(overrides)
    return OperatorSignature(
        kind=OperatorKind.ATTENTION,
        parameters=AttentionSignature(**params),
        backend_family="torch",
    )


def _norm(**overrides: Any) -> OperatorSignature:
    params: dict[str, Any] = {
        "batch_size": 1,
        "sequence_length": 8,
        "hidden_size": 64,
        "dtype": "fp32",
        "variant": NormVariant.RMS,
    }
    params.update(overrides)
    return OperatorSignature(
        kind=OperatorKind.NORM, parameters=NormSignature(**params), backend_family="torch"
    )


def _spy_mm(monkeypatch: pytest.MonkeyPatch) -> list[tuple[torch.Tensor, torch.Tensor]]:
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    real_mm = torch.mm

    def spy(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        calls.append((left, right))
        return real_mm(left, right)

    monkeypatch.setattr(torch, "mm", spy)
    return calls


def _spy_sdpa(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    real_sdpa = F.scaled_dot_product_attention

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append({"args": args, **kwargs})
        return real_sdpa(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    return calls


class TestProtocolConformance:
    @pytest.mark.parametrize(
        "workload", [GemmWorkload(), AttentionWorkload(), NormWorkload()]
    )
    def test_workloads_satisfy_protocol(self, workload: Any) -> None:
        assert isinstance(workload, OperatorWorkload)
        assert workload.backend_family == "torch"


class TestTypedParameters:
    def test_narrowing(self) -> None:
        signature = _gemm()
        params = typed_parameters(signature, GemmSignature)
        assert params.m == 4

    def test_foreign_parameters_fail_typed(self) -> None:
        with pytest.raises(ProfilingError) as excinfo:
            typed_parameters(_attention(), GemmSignature)
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_OPERATOR


class TestWorkloadDtype:
    def test_known_labels(self) -> None:
        assert workload_dtype("fp32") is torch.float32
        assert workload_dtype("bf16") is torch.bfloat16

    def test_unknown_label_fails_typed(self) -> None:
        with pytest.raises(ProfilingError) as excinfo:
            workload_dtype("unknown")
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_OPERATOR
        assert ("dtype", "unknown") in excinfo.value.to_failure().details

    def test_invalid_label_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown dtype label"):
            workload_dtype("fp7")


class TestParametersDtype:
    def test_typed_parameters_expose_dtype(self) -> None:
        assert parameters_dtype(GemmSignature(m=2, n=2, k=2, dtype="bf16")) == "bf16"

    def test_custom_parameters_fail_typed(self) -> None:
        params = CustomOperatorParameters(
            raw_name="aten.mystery.default",
            input_shapes=((2, 2),),
            input_dtypes=("fp32",),
        )
        with pytest.raises(ProfilingError) as excinfo:
            parameters_dtype(params)
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_OPERATOR


class TestGemmWorkload:
    def test_runs_and_releases(self) -> None:
        workload = GemmWorkload()
        workload.prepare(_gemm(), CPU)
        workload.run_once()
        workload.run_once()
        workload.cleanup()
        with pytest.raises(RuntimeError, match="not been prepared"):
            workload.run_once()

    def test_run_before_prepare_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="not been prepared"):
            GemmWorkload().run_once()

    def test_linear_layout_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """transpose_b reproduces the (N, K)-stored weight of a linear (§26)."""
        calls = _spy_mm(monkeypatch)
        workload = GemmWorkload()
        workload.prepare(_gemm(m=4, n=16, k=8, transpose_b=True), CPU)
        workload.run_once()
        left, right = calls[0]
        assert left.shape == (4, 8)
        assert right.shape == (8, 16)
        # Storage is (16, 8) row-major presented transposed — the exact
        # view a production F.linear gives cuBLAS, not a re-materialized
        # contiguous (8, 16) matrix.
        assert right.stride() == (1, 8)

    def test_plain_layout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _spy_mm(monkeypatch)
        workload = GemmWorkload()
        workload.prepare(_gemm(m=4, n=16, k=8, transpose_a=False, transpose_b=False), CPU)
        workload.run_once()
        left, right = calls[0]
        assert left.shape == (4, 8)
        assert right.shape == (8, 16)
        assert right.stride() == (16, 1)  # contiguous storage

    def test_transpose_a(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _spy_mm(monkeypatch)
        workload = GemmWorkload()
        workload.prepare(_gemm(m=4, k=8, transpose_a=True, transpose_b=False), CPU)
        workload.run_once()
        left, _ = calls[0]
        assert left.shape == (4, 8)
        assert left.stride() == (1, 4)  # stored (k, m), presented transposed

    def test_dtype_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _spy_mm(monkeypatch)
        workload = GemmWorkload()
        workload.prepare(_gemm(dtype="bf16"), CPU)
        workload.run_once()
        left, right = calls[0]
        assert left.dtype is torch.bfloat16 and right.dtype is torch.bfloat16

    def test_unknown_dtype_fails_typed(self) -> None:
        with pytest.raises(ProfilingError) as excinfo:
            GemmWorkload().prepare(_gemm(dtype="unknown"), CPU)
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_OPERATOR

    def test_foreign_signature_fails_typed(self) -> None:
        with pytest.raises(ProfilingError) as excinfo:
            GemmWorkload().prepare(_attention(), CPU)
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_OPERATOR


class TestAttentionWorkload:
    def test_prefill_is_causal_gqa(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _spy_sdpa(monkeypatch)
        workload = AttentionWorkload()
        workload.prepare(_attention(), CPU)
        workload.run_once()
        call = calls[0]
        query, key, value = call["args"]
        assert query.shape == (1, 4, 8, 16)
        assert key.shape == (1, 2, 8, 16)
        assert value.shape == (1, 2, 8, 16)
        assert call["is_causal"] is True
        assert call["enable_gqa"] is True

    def test_decode_single_query_full_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _spy_sdpa(monkeypatch)
        workload = AttentionWorkload()
        workload.prepare(
            _attention(q_len=1, kv_len=9, phase=InferencePhase.DECODE), CPU
        )
        workload.run_once()
        call = calls[0]
        query, key, _ = call["args"]
        assert query.shape == (1, 4, 1, 16)
        assert key.shape == (1, 2, 9, 16)
        assert call["is_causal"] is False

    def test_mha_runs_with_enable_gqa(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _spy_sdpa(monkeypatch)
        workload = AttentionWorkload()
        workload.prepare(_attention(num_heads=4, num_kv_heads=4), CPU)
        workload.run_once()
        assert calls[0]["enable_gqa"] is True

    def test_dtype_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _spy_sdpa(monkeypatch)
        workload = AttentionWorkload()
        workload.prepare(_attention(dtype="bf16"), CPU)
        workload.run_once()
        query, key, value = calls[0]["args"]
        assert query.dtype is torch.bfloat16
        assert key.dtype is torch.bfloat16 and value.dtype is torch.bfloat16

    def test_chunked_prefill_fails_typed(self) -> None:
        """q_len != kv_len prefill has no synthesized masking in v1 (§24)."""
        with pytest.raises(ProfilingError) as excinfo:
            AttentionWorkload().prepare(_attention(q_len=4, kv_len=8), CPU)
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_OPERATOR
        details = dict(excinfo.value.to_failure().details)
        assert details == {"q_len": 4, "kv_len": 8}

    def test_multi_token_decode_fails_typed(self) -> None:
        with pytest.raises(ProfilingError) as excinfo:
            AttentionWorkload().prepare(
                _attention(q_len=2, kv_len=9, phase=InferencePhase.DECODE), CPU
            )
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_OPERATOR

    def test_run_before_prepare_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="not been prepared"):
            AttentionWorkload().run_once()

    def test_cleanup_releases(self) -> None:
        workload = AttentionWorkload()
        workload.prepare(_attention(), CPU)
        workload.cleanup()
        with pytest.raises(RuntimeError, match="not been prepared"):
            workload.run_once()


class TestNormWorkload:
    def test_rms_promotes_to_fp32(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The eager production RMSNorm sequence upcasts before rsqrt."""
        seen: list[torch.Tensor] = []
        real_rsqrt = torch.rsqrt

        def spy(value: torch.Tensor) -> torch.Tensor:
            seen.append(value)
            return real_rsqrt(value)

        monkeypatch.setattr(torch, "rsqrt", spy)
        workload = NormWorkload()
        workload.prepare(_norm(dtype="bf16"), CPU)
        workload.run_once()
        assert len(seen) == 1
        assert seen[0].dtype is torch.float32
        assert seen[0].shape == (1, 8, 1)  # mean over hidden, keepdim

    def test_layer_variant_uses_native(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict[str, Any]] = []
        real_layer_norm = F.layer_norm

        def spy(*args: Any, **kwargs: Any) -> Any:
            calls.append({"args": args, **kwargs})
            return real_layer_norm(*args, **kwargs)

        monkeypatch.setattr(F, "layer_norm", spy)
        workload = NormWorkload()
        workload.prepare(_norm(variant=NormVariant.LAYER), CPU)
        workload.run_once()
        normalized_shape, weight = calls[0]["args"][1], calls[0]["args"][2]
        assert normalized_shape == (64,)
        assert torch.equal(weight, torch.ones(64))

    def test_other_variant_fails_typed(self) -> None:
        with pytest.raises(ProfilingError) as excinfo:
            NormWorkload().prepare(_norm(variant=NormVariant.OTHER), CPU)
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_OPERATOR

    def test_unknown_dtype_fails_typed(self) -> None:
        with pytest.raises(ProfilingError) as excinfo:
            NormWorkload().prepare(_norm(dtype="unknown"), CPU)
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_OPERATOR

    def test_run_before_prepare_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="not been prepared"):
            NormWorkload().run_once()

    def test_cleanup_releases(self) -> None:
        workload = NormWorkload()
        workload.prepare(_norm(), CPU)
        workload.cleanup()
        with pytest.raises(RuntimeError, match="not been prepared"):
            workload.run_once()
