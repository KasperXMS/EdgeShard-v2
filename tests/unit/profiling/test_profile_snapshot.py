"""Immutable ProfileSnapshot (spec §46)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

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
from edgeshard.profiling.domain.snapshot import ProfileSnapshot

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


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
        case_id="c-1",
        environment_fingerprint="f" * 64,
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
