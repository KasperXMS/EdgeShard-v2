"""P2E incremental operator profiling tests (spec §28, P2E DoD).

The plan is pure computation over canonical signature ids: dedup, split
against the measured set, nothing dropped. The end-to-end tests use the
*real* extractor output of the tiny Qwen2 layers (module fixture) to pin
the DoD guarantees — operator profiling runs without model execution,
existing signatures are reused, and a new model contributes only its
missing shapes.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from typing import Any

from edgeshard.profiling.benchmark.harness import InstrumentationBundle
from edgeshard.profiling.benchmark.sampling import DurationSamplingPolicy
from edgeshard.profiling.domain.experiment import ProfilingCase
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
from edgeshard.profiling.operator.extractor import RawOperatorGraph
from edgeshard.profiling.operator.normalizer import (
    OperatorNormalizer,
    unique_operator_signatures,
)
from edgeshard.profiling.operator.profiler import (
    IncrementalOperatorProfiler,
    MeasuredSignatureIndex,
    OperatorProfiler,
    operator_case_spec,
    plan_incremental_profiling,
)
from edgeshard.profiling.operator.registry import default_operator_workload_registry


class StaticIndex:
    """Test double for the store-side reuse query (§28)."""

    def __init__(self, ids: AbstractSet[str] = frozenset()) -> None:
        self._ids = frozenset(ids)
        self.queries = 0

    def measured_signature_ids(self) -> AbstractSet[str]:
        self.queries += 1
        return self._ids


def _gemm(m: int, n: int = 16, k: int = 4) -> OperatorSignature:
    return OperatorSignature(
        kind=OperatorKind.GEMM,
        parameters=GemmSignature(m=m, n=n, k=k, dtype="fp32", transpose_b=True),
        backend_family="torch",
    )


def _norm() -> OperatorSignature:
    return OperatorSignature(
        kind=OperatorKind.NORM,
        parameters=NormSignature(1, 8, 64, "fp32", NormVariant.RMS),
        backend_family="torch",
    )


def _attention() -> OperatorSignature:
    return OperatorSignature(
        kind=OperatorKind.ATTENTION,
        parameters=AttentionSignature(1, 4, 2, 16, 8, 8, "fp32", InferencePhase.PREFILL),
        backend_family="torch",
    )


def _policy() -> DurationSamplingPolicy:
    return DurationSamplingPolicy(
        min_warmups=1, min_runs=3, max_runs=3, target_duration_ms=1_000_000.0
    )


def _ids(signatures: tuple[OperatorSignature, ...] | list[OperatorSignature]) -> frozenset[str]:
    return frozenset(operator_signature_id(signature) for signature in signatures)


class TestMeasuredSignatureIndexProtocol:
    def test_static_index_conforms(self) -> None:
        assert isinstance(StaticIndex(), MeasuredSignatureIndex)


class TestPlanIncrementalProfiling:
    def test_empty_input(self) -> None:
        plan = plan_incremental_profiling([], measured_ids=frozenset({"anything"}))
        assert plan.reused == ()
        assert plan.missing == ()

    def test_nothing_measured_yet(self) -> None:
        signatures = (_gemm(8), _attention(), _norm())
        plan = plan_incremental_profiling(signatures, measured_ids=frozenset())
        assert plan.reused == ()
        assert plan.missing == signatures  # order preserved

    def test_everything_measured(self) -> None:
        signatures = (_gemm(8), _attention(), _norm())
        plan = plan_incremental_profiling(signatures, measured_ids=_ids(signatures))
        assert plan.reused == signatures
        assert plan.missing == ()

    def test_split_preserves_order_and_loses_nothing(self) -> None:
        signatures = (_gemm(8), _attention(), _norm(), _gemm(32))
        plan = plan_incremental_profiling(
            signatures, measured_ids=_ids([_attention()])
        )
        assert plan.reused == (_attention(),)
        assert plan.missing == (_gemm(8), _norm(), _gemm(32))
        assert len(plan.reused) + len(plan.missing) == len(signatures)

    def test_duplicates_dedup_by_canonical_id(self) -> None:
        plan = plan_incremental_profiling(
            [_gemm(8), _gemm(8), _norm(), _gemm(8)], measured_ids=frozenset()
        )
        assert plan.missing == (_gemm(8), _norm())

    def test_equal_distinct_objects_dedup(self) -> None:
        """Identity is structural: separately built equal signatures are one."""
        plan = plan_incremental_profiling(
            [_gemm(8), _gemm(8, n=16, k=4)], measured_ids=_ids([_gemm(8)])
        )
        assert plan.reused == (_gemm(8),)
        assert plan.missing == ()

    def test_unmeasured_id_set_ignores_foreign_ids(self) -> None:
        plan = plan_incremental_profiling(
            [_gemm(8)], measured_ids=frozenset({"not-a-real-signature-id"})
        )
        assert plan.missing == (_gemm(8),)

    def test_new_model_adds_only_missing_shapes(self) -> None:
        """§28 guarantee: no full reprofiling when a new model arrives."""
        model_a = (_gemm(8), _attention(), _norm())
        model_b = (_gemm(8), _gemm(64), _attention())  # shares 2, adds 1
        measured_after_a = _ids(model_a)
        plan = plan_incremental_profiling(
            [*model_a, *model_b], measured_ids=measured_after_a
        )
        assert plan.missing == (_gemm(64),)
        assert len(plan.reused) == 3


class TestIncrementalOperatorProfiler:
    def test_benchmarks_only_missing(self) -> None:
        signatures = (_gemm(8), _attention(), _norm())
        index = StaticIndex(_ids([_gemm(8), _norm()]))
        result = IncrementalOperatorProfiler(
            index=index, profiler=OperatorProfiler(sampling_policy=_policy())
        ).run(
            signatures,
            worker_id="worker-test",
            device_id="cpu-0",
            environment_fingerprint="env-test",
            instrumentation=InstrumentationBundle(timer=WallClockTimer()),
        )
        assert index.queries == 1
        assert result.plan.reused == (_gemm(8), _norm())
        assert result.plan.missing == (_attention(),)
        assert len(result.records) == 1
        record = result.records[0]
        assert record.sample_count == 3
        assert record.environment_fingerprint == "env-test"
        metadata = record.metadata_mapping
        assert metadata["operator_kind"] == "attention"
        assert metadata["operator_signature_id"] == operator_signature_id(_attention())
        # The record's case id is the canonical request id for the missing
        # signature — the P2G store will append it under that identity.
        case = ProfilingCase.for_spec(
            "worker-test", operator_case_spec(_attention(), device_id="cpu-0")
        )
        assert record.case_id == case.case_id

    def test_fully_measured_input_produces_no_records(self) -> None:
        signatures = (_gemm(8), _norm())
        result = IncrementalOperatorProfiler(
            index=StaticIndex(_ids(signatures)),
            profiler=OperatorProfiler(sampling_policy=_policy()),
        ).run(
            signatures,
            worker_id="worker-test",
            device_id="cpu-0",
            environment_fingerprint="env-test",
            instrumentation=InstrumentationBundle(timer=WallClockTimer()),
        )
        assert result.plan.missing == ()
        assert result.records == ()

    def test_real_extracted_signatures_end_to_end(
        self, qwen2_layer_exports: tuple[RawOperatorGraph, ...]
    ) -> None:
        """DoD on real extractor output: dedup → plan → benchmark missing.

        Runs entirely without model execution — the exported graphs were
        produced once by the fixture; everything after normalization is
        synthetic microbenchmarking.
        """
        normalizer = OperatorNormalizer()
        graphs = [
            normalizer.normalize(graph, phase=InferencePhase.PREFILL)
            for graph in qwen2_layer_exports
        ]
        signatures = unique_operator_signatures(graphs)
        registry = default_operator_workload_registry()
        benchmarkable = tuple(
            signature for signature in signatures if registry.supports(signature.kind)
        )
        # The tiny Qwen2 layer exports exactly one SDPA occurrence and a
        # set of linear GEMMs; both dedup across the two identical layers.
        kinds = [signature.kind for signature in benchmarkable]
        assert kinds.count(OperatorKind.ATTENTION) == 1
        assert kinds.count(OperatorKind.GEMM) >= 1

        # First device: nothing measured yet → every supported signature runs.
        first = IncrementalOperatorProfiler(
            index=StaticIndex(), profiler=OperatorProfiler(sampling_policy=_policy())
        ).run(
            benchmarkable,
            worker_id="worker-test",
            device_id="cpu-0",
            environment_fingerprint="env-first",
            instrumentation=InstrumentationBundle(timer=WallClockTimer()),
        )
        assert len(first.records) == len(benchmarkable)
        measured = _ids(benchmarkable)

        # A second, structurally identical "new model" adds nothing (§28).
        second = IncrementalOperatorProfiler(
            index=StaticIndex(measured),
            profiler=OperatorProfiler(sampling_policy=_policy()),
        ).run(
            benchmarkable,
            worker_id="worker-test",
            device_id="cpu-0",
            environment_fingerprint="env-first",
            instrumentation=InstrumentationBundle(timer=WallClockTimer()),
        )
        assert second.plan.reused == benchmarkable
        assert second.records == ()

        # A genuinely new shape contributes exactly one missing signature.
        widened = (*benchmarkable, _gemm(128, n=64, k=32))
        third = IncrementalOperatorProfiler(
            index=StaticIndex(measured),
            profiler=OperatorProfiler(sampling_policy=_policy()),
        ).run(
            widened,
            worker_id="worker-test",
            device_id="cpu-0",
            environment_fingerprint="env-first",
            instrumentation=InstrumentationBundle(timer=WallClockTimer()),
        )
        assert third.plan.missing == (_gemm(128, n=64, k=32),)
        assert len(third.records) == 1
        assert (
            third.records[0].metadata_mapping["operator_signature_id"]
            == operator_signature_id(_gemm(128, n=64, k=32))
        )


class TestPlanDeterminism:
    def test_repeated_plans_identical(self) -> None:
        signatures = (_gemm(8), _attention(), _norm())
        index = StaticIndex(_ids([_attention()]))
        first = plan_incremental_profiling(
            signatures, measured_ids=index.measured_signature_ids()
        )
        second = plan_incremental_profiling(
            signatures, measured_ids=index.measured_signature_ids()
        )
        assert first == second

    def test_plan_accepts_any_iterable(self) -> None:
        generated: Any = (_gemm(m) for m in (4, 8, 4))
        plan = plan_incremental_profiling(generated, measured_ids=frozenset())
        assert len(plan.missing) == 2
