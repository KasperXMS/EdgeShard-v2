"""Immutable ProfileSnapshot (spec §46)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from edgeshard.profiling.domain.environment import (
    EnvironmentFingerprint,
    environment_fingerprint_id,
)
from edgeshard.profiling.domain.experiment import (
    ModelCaseSpec,
    ProfilingCase,
)
from edgeshard.profiling.domain.measurement import (
    LatencyMetrics,
    MeasurementMetrics,
    MeasurementRecord,
    SampleSummary,
    TimeUnit,
)
from edgeshard.profiling.domain.model import (
    ModelCharacterization,
    ModelReference,
    ModelStage,
    StageKind,
)
from edgeshard.profiling.domain.signature import (
    GemmSignature,
    OperatorKind,
    OperatorSignature,
    ProfilingGranularity,
)
from edgeshard.profiling.domain.snapshot import ProfileSnapshot

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)

ENVIRONMENT = EnvironmentFingerprint(
    backend="torch",
    profiling_implementation_revision="test",
    torch_version="test",
    dtype="fp32",
    worker_id="w-1",
    device_id="gpu-0",
)
OPERATOR_SIGNATURE = OperatorSignature(
    kind=OperatorKind.GEMM,
    parameters=GemmSignature(m=1, n=1, k=1, dtype="fp32"),
    backend_family="torch",
)
CASE = ProfilingCase.for_spec(
    "w-1",
    ModelCaseSpec(
        granularity=ProfilingGranularity.OPERATOR,
        device_ids=("gpu-0",),
        dtype="fp32",
        operator_signature=OPERATOR_SIGNATURE,
    ),
)


def _characterization(
    model_id: str = "Qwen/Qwen2.5-7B", revision: str | None = "r1"
) -> ModelCharacterization:
    return ModelCharacterization(
        model=ModelReference(model_id=model_id, revision=revision),
        architecture_family="qwen2",
        num_layers=4,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=128,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        dtype="bf16",
        quantization=None,
        tied_word_embeddings=False,
        stages=(ModelStage(kind=StageKind.TRANSFORMER_LAYER_GROUP, layer_count=4),),
    )


def _record(measurement_id: str) -> MeasurementRecord:
    return MeasurementRecord(
        measurement_id=measurement_id,
        case_id=CASE.case_id,
        environment_fingerprint=environment_fingerprint_id(ENVIRONMENT),
        environment=ENVIRONMENT,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
        sample_count=1,
        samples=(1.0,),
        metrics=MeasurementMetrics(
            latency=LatencyMetrics(
                summary=SampleSummary(1.0, 1.0, 0.0, 1.0, 1.0, None),
                unit=TimeUnit.MILLISECONDS,
            )
        ),
    )


def _snapshot(**overrides: object) -> ProfileSnapshot:
    base: dict[str, object] = {
        "snapshot_id": "snap-1",
        "created_at": NOW,
        "model_characterizations": (_characterization(),),
        "measurements": (_record("m-1"),),
        "network_measurements": (_record("m-net-1"),),
        "profiling_cases": (CASE,),
        "operator_signatures": (OPERATOR_SIGNATURE,),
        "environment_fingerprints": (ENVIRONMENT,),
    }
    base.update(overrides)
    return ProfileSnapshot(**base)  # type: ignore[arg-type]


def test_snapshot_constructs_and_is_frozen() -> None:
    snapshot = _snapshot()
    assert snapshot.snapshot_id == "snap-1"
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.snapshot_id = "mutated"  # type: ignore[misc]


def test_snapshot_rejects_naive_timestamp_and_empty_id() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _snapshot(created_at=datetime(2026, 9, 7, 12, 0))
    with pytest.raises(ValueError, match="snapshot_id"):
        _snapshot(snapshot_id="")


def test_snapshot_rejects_duplicate_model_characterizations() -> None:
    with pytest.raises(ValueError, match="duplicate model characterization"):
        _snapshot(model_characterizations=(_characterization(), _characterization()))
    # A different revision of the same model is a distinct characterization.
    _snapshot(
        model_characterizations=(_characterization(revision="r1"), _characterization(revision="r2"))
    )


def test_snapshot_rejects_duplicate_measurement_ids_across_collections() -> None:
    with pytest.raises(ValueError, match="duplicate measurement_id"):
        _snapshot(measurements=(_record("m-1"),), network_measurements=(_record("m-1"),))


def test_empty_snapshot_is_valid() -> None:
    """A cluster with nothing profiled yet yields an empty — not invalid — snapshot."""
    snapshot = _snapshot(
        model_characterizations=(), measurements=(), network_measurements=()
    )
    assert snapshot.measurements == ()


def test_snapshot_rejects_measurement_without_self_contained_facts() -> None:
    record = dataclasses.replace(_record("m-orphan"), environment=None)
    with pytest.raises(ValueError, match="full environment"):
        _snapshot(measurements=(record,), network_measurements=())
    with pytest.raises(ValueError, match="missing profiling cases"):
        _snapshot(profiling_cases=())
    with pytest.raises(ValueError, match="missing environment fingerprints"):
        _snapshot(environment_fingerprints=())
