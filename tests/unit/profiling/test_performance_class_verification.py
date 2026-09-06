"""P2E performance-class verification tests (spec §29, P2E DoD).

The verification suite is exactly what §29 asks for — one representative
GEMM, one attention case, one memory-sensitive norm — with every
dimension an explicit policy knob. The comparison is deliberately simple
(per-signature relative deviation against one shared tolerance, no
statistical certification): identical means are compatible, a candidate
outside tolerance is not, and comparisons only ever run between observed
latency means (§52.1) — records without latency are rejected, not
imputed. The end-to-end test runs the suite twice on this host through
the real profiler and verifies the verdict machinery on facts.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from edgeshard.profiling.benchmark.harness import InstrumentationBundle
from edgeshard.profiling.benchmark.sampling import DurationSamplingPolicy
from edgeshard.profiling.domain.experiment import ProfilingCase
from edgeshard.profiling.domain.measurement import (
    LatencyMetrics,
    MeasurementMetrics,
    MeasurementRecord,
    SampleSummary,
    TelemetryContextMetrics,
    TimeUnit,
)
from edgeshard.profiling.domain.signature import (
    AttentionSignature,
    GemmSignature,
    InferencePhase,
    NormSignature,
    NormVariant,
    OperatorKind,
    OperatorSignature,
    operator_signature_id,
)
from edgeshard.profiling.instrumentation.timing import WallClockTimer
from edgeshard.profiling.operator.profiler import (
    DEFAULT_VERIFICATION_TOLERANCE,
    OperatorProfiler,
    operator_case_spec,
    verification_suite,
    verify_performance_class,
)
from edgeshard.profiling.operator.registry import default_operator_workload_registry


def _record(mean: float | None) -> MeasurementRecord:
    now = datetime.now(UTC)
    latency = (
        LatencyMetrics(
            summary=SampleSummary(
                mean=mean,
                median=mean,
                stddev=0.0,
                minimum=mean,
                maximum=mean,
                p95=None,
            ),
            unit=TimeUnit.MILLISECONDS,
        )
        if mean is not None
        else None
    )
    metrics = MeasurementMetrics(
        latency=latency,
        telemetry=None if latency is not None else TelemetryContextMetrics(),
    )
    return MeasurementRecord(
        measurement_id="m1",
        case_id="c1",
        environment_fingerprint="e1",
        started_at=now,
        finished_at=now,
        sample_count=1,
        samples=(mean,) if mean is not None else None,
        metrics=metrics,
    )


def _tiny_suite() -> tuple[OperatorSignature, ...]:
    return verification_suite(
        dtype="fp32",
        gemm_dim=16,
        attention_heads=4,
        attention_kv_heads=2,
        attention_head_dim=8,
        attention_length=4,
        norm_tokens=4,
        norm_hidden=16,
    )


def _means(
    suite: tuple[OperatorSignature, ...], values: tuple[float, ...]
) -> dict[str, MeasurementRecord]:
    return {
        operator_signature_id(signature): _record(value)
        for signature, value in zip(suite, values, strict=True)
    }


class TestVerificationSuite:
    def test_exactly_three_workloads(self) -> None:
        suite = verification_suite()
        assert [signature.kind for signature in suite] == [
            OperatorKind.GEMM,
            OperatorKind.ATTENTION,
            OperatorKind.NORM,
        ]
        assert all(signature.backend_family == "torch" for signature in suite)

    def test_default_dimensions(self) -> None:
        gemm, attention, norm = verification_suite()
        assert isinstance(gemm.parameters, GemmSignature)
        assert gemm.parameters == GemmSignature(
            m=1024, n=1024, k=1024, dtype="bf16", transpose_b=True
        )
        assert isinstance(attention.parameters, AttentionSignature)
        assert attention.parameters.phase is InferencePhase.PREFILL
        assert attention.parameters.q_len == attention.parameters.kv_len == 256
        assert isinstance(norm.parameters, NormSignature)
        assert norm.parameters.variant is NormVariant.RMS
        assert norm.parameters.hidden_size == 4096

    def test_knobs_flow_into_signatures(self) -> None:
        gemm, attention, norm = verification_suite(
            dtype="fp32", gemm_dim=64, attention_length=32, norm_hidden=128
        )
        assert isinstance(gemm.parameters, GemmSignature)
        assert gemm.parameters.m == 64 and gemm.parameters.dtype == "fp32"
        assert isinstance(attention.parameters, AttentionSignature)
        assert attention.parameters.q_len == 32
        assert isinstance(norm.parameters, NormSignature)
        assert norm.parameters.hidden_size == 128 and norm.parameters.dtype == "fp32"

    def test_suite_is_benchmarkable_by_default_registry(self) -> None:
        registry = default_operator_workload_registry()
        assert all(registry.supports(signature.kind) for signature in _tiny_suite())

    def test_default_tolerance_is_a_policy_constant(self) -> None:
        assert DEFAULT_VERIFICATION_TOLERANCE == 0.15


class TestVerifyPerformanceClass:
    def test_identical_means_are_compatible(self) -> None:
        suite = _tiny_suite()
        references = _means(suite, (10.0, 20.0, 30.0))
        candidates = _means(suite, (10.0, 20.0, 30.0))
        verdict = verify_performance_class(references, candidates)
        assert verdict.compatible is True
        assert verdict.tolerance == DEFAULT_VERIFICATION_TOLERANCE
        assert verdict.max_relative_deviation == 0.0
        assert len(verdict.comparisons) == 3
        assert all(
            comparison.relative_deviation == 0.0 and comparison.within_tolerance
            for comparison in verdict.comparisons
        )

    def test_within_tolerance(self) -> None:
        suite = _tiny_suite()
        verdict = verify_performance_class(
            _means(suite, (10.0, 10.0, 10.0)), _means(suite, (11.0, 10.5, 9.0))
        )
        assert verdict.compatible is True
        assert verdict.max_relative_deviation == pytest.approx(0.1)

    def test_single_outlier_fails_the_verdict(self) -> None:
        suite = _tiny_suite()
        verdict = verify_performance_class(
            _means(suite, (10.0, 10.0, 10.0)), _means(suite, (10.0, 15.0, 10.0))
        )
        assert verdict.compatible is False
        assert verdict.max_relative_deviation == pytest.approx(0.5)
        assert [comparison.within_tolerance for comparison in verdict.comparisons].count(
            False
        ) == 1

    def test_tolerance_is_configurable_and_inclusive(self) -> None:
        references = {"sig-a": _record(10.0)}
        candidates = {"sig-a": _record(11.5)}
        verdict = verify_performance_class(references, candidates, tolerance=0.15)
        assert verdict.comparisons[0].relative_deviation == pytest.approx(0.15)
        assert verdict.compatible is True
        tighter = verify_performance_class(references, candidates, tolerance=0.1)
        assert tighter.compatible is False
        assert tighter.tolerance == pytest.approx(0.1)

    def test_comparisons_sorted_by_signature_id(self) -> None:
        references = {"zzz": _record(10.0), "aaa": _record(10.0)}
        candidates = {"zzz": _record(10.0), "aaa": _record(10.0)}
        verdict = verify_performance_class(references, candidates)
        assert [
            comparison.operator_signature_id for comparison in verdict.comparisons
        ] == ["aaa", "zzz"]

    def test_comparison_fields(self) -> None:
        verdict = verify_performance_class(
            {"sig": _record(8.0)}, {"sig": _record(10.0)}, tolerance=0.5
        )
        comparison = verdict.comparisons[0]
        assert comparison.operator_signature_id == "sig"
        assert comparison.reference_mean_ms == 8.0
        assert comparison.candidate_mean_ms == 10.0
        assert comparison.relative_deviation == pytest.approx(0.25)
        assert comparison.within_tolerance is True

    def test_empty_suite_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            verify_performance_class({}, {})

    def test_non_positive_tolerance_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            verify_performance_class({"sig": _record(1.0)}, {"sig": _record(1.0)}, tolerance=0.0)

    def test_mismatched_suites_rejected(self) -> None:
        with pytest.raises(ValueError, match="same suite"):
            verify_performance_class(
                {"sig-a": _record(1.0), "sig-b": _record(1.0)},
                {"sig-a": _record(1.0), "sig-c": _record(1.0)},
            )

    def test_records_without_latency_rejected(self) -> None:
        """Comparisons run on observed facts only — never imputed (§52.1)."""
        with pytest.raises(ValueError, match="no latency"):
            verify_performance_class({"sig": _record(1.0)}, {"sig": _record(None)})
        with pytest.raises(ValueError, match="no latency"):
            verify_performance_class({"sig": _record(None)}, {"sig": _record(1.0)})

    def test_non_positive_mean_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-positive"):
            verify_performance_class({"sig": _record(1.0)}, {"sig": _record(0.0)})


class TestVerificationEndToEnd:
    def test_same_host_passes_with_generous_tolerance(self) -> None:
        """§29 flow on real measurements: run the suite, compare, verdict.

        Two runs on one host describe the same device, so a generous
        tolerance must yield ``compatible``; the exact deviation is CPU
        timing noise and is not asserted beyond finiteness.
        """
        suite = _tiny_suite()
        profiler = OperatorProfiler(
            sampling_policy=DurationSamplingPolicy(
                min_warmups=1, min_runs=3, max_runs=3, target_duration_ms=1_000_000.0
            )
        )
        instrumentation = InstrumentationBundle(timer=WallClockTimer())

        def run_suite(fingerprint: str) -> dict[str, MeasurementRecord]:
            records: dict[str, MeasurementRecord] = {}
            for signature in suite:
                record = profiler.profile(
                    ProfilingCase.for_spec(
                        "worker-test", operator_case_spec(signature, device_id="cpu-0")
                    ),
                    instrumentation=instrumentation,
                    environment_fingerprint=fingerprint,
                )
                records[operator_signature_id(signature)] = record
            return records

        references = run_suite("env-reference")
        candidates = run_suite("env-candidate")
        verdict = verify_performance_class(references, candidates, tolerance=1.0)
        assert verdict.compatible is True
        assert len(verdict.comparisons) == 3
        assert all(
            0.0 <= comparison.relative_deviation < 1.0
            for comparison in verdict.comparisons
        )
