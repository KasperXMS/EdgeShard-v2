"""Immutable ProfileSnapshot (Phase 2 spec §46).

The point-in-time view of everything Phase 2 knows: static model
characterizations plus empirical measurements, split into device/model
observations and network observations. The snapshot contains empirical and
static facts only — no estimates, predictions, placements, or composed
performance numbers (§1.1, §46). Phase 3 consumes this snapshot and must
never depend on a live ``ProfilingRunner``.

All members are frozen dataclasses over tuples, so a snapshot built from
the ProfileStore can never be mutated by measurements appended afterwards
— the same construction discipline as the Phase 1 ``ClusterSnapshot``.
Timestamps must be timezone-aware so snapshots from different Masters stay
comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from edgeshard.profiling.domain.measurement import MeasurementRecord
from edgeshard.profiling.domain.model import ModelCharacterization


def _require_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field} must be timezone-aware")


@dataclass(frozen=True)
class ProfileSnapshot:
    """Immutable view of all Phase 2 facts at one instant (spec §46).

    Validation: characterization identity is unique per
    (``model_id``, ``revision``) — one characterization per model snapshot —
    and measurement ids are unique across both measurement collections.
    """

    snapshot_id: str
    created_at: datetime
    model_characterizations: tuple[ModelCharacterization, ...]
    measurements: tuple[MeasurementRecord, ...]
    network_measurements: tuple[MeasurementRecord, ...]

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise ValueError("snapshot_id must not be empty")
        _require_aware(self.created_at, "created_at")

        seen_models: set[tuple[str, str | None]] = set()
        for characterization in self.model_characterizations:
            key = (
                characterization.model.model_id,
                characterization.model.revision,
            )
            if key in seen_models:
                raise ValueError(
                    f"duplicate model characterization for model_id={key[0]!r} "
                    f"revision={key[1]!r}"
                )
            seen_models.add(key)

        seen_ids: set[str] = set()
        for record in (*self.measurements, *self.network_measurements):
            if record.measurement_id in seen_ids:
                raise ValueError(
                    f"duplicate measurement_id {record.measurement_id!r} in snapshot"
                )
            seen_ids.add(record.measurement_id)
