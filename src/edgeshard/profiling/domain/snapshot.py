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

from edgeshard.profiling.domain.environment import (
    DevicePerformanceClass,
    DevicePerformanceClassMembership,
    EnvironmentFingerprint,
    device_performance_class_id,
    environment_fingerprint_id,
)
from edgeshard.profiling.domain.experiment import ProfilingCase
from edgeshard.profiling.domain.measurement import MeasurementRecord
from edgeshard.profiling.domain.model import ModelCharacterization
from edgeshard.profiling.domain.network import (
    NetworkEndpointProfile,
    NetworkPair,
    NetworkPathClass,
)
from edgeshard.profiling.domain.signature import (
    ModuleSignature,
    OperatorSignature,
    TransformerLayerSignature,
)


def _require_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field} must be timezone-aware")


@dataclass(frozen=True)
class NetworkPathProfile:
    """A persisted directed pair plus its current path classification."""

    pair: NetworkPair
    path_class: NetworkPathClass


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
    profiling_cases: tuple[ProfilingCase, ...] = ()
    layer_signatures: tuple[TransformerLayerSignature, ...] = ()
    module_signatures: tuple[ModuleSignature, ...] = ()
    operator_signatures: tuple[OperatorSignature, ...] = ()
    environment_fingerprints: tuple[EnvironmentFingerprint, ...] = ()
    device_performance_classes: tuple[DevicePerformanceClass, ...] = ()
    device_performance_class_memberships: tuple[
        DevicePerformanceClassMembership, ...
    ] = ()
    network_endpoints: tuple[NetworkEndpointProfile, ...] = ()
    network_paths: tuple[NetworkPathProfile, ...] = ()

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

        records = (*self.measurements, *self.network_measurements)
        seen_ids: set[str] = set()
        for record in records:
            if record.measurement_id in seen_ids:
                raise ValueError(
                    f"duplicate measurement_id {record.measurement_id!r} in snapshot"
                )
            seen_ids.add(record.measurement_id)

        missing_environments = [
            record.measurement_id for record in records if record.environment is None
        ]
        if missing_environments:
            raise ValueError(
                "snapshot measurements lack full environment fingerprints: "
                f"{sorted(missing_environments)}"
            )
        case_ids = {case.case_id for case in self.profiling_cases}
        missing_cases = {record.case_id for record in records if record.case_id not in case_ids}
        if missing_cases:
            raise ValueError(
                "snapshot measurements reference missing profiling cases: "
                f"{sorted(missing_cases)}"
            )

        fingerprint_ids = {
            environment_fingerprint_id(item) for item in self.environment_fingerprints
        }
        missing = {
            record.environment_fingerprint
            for record in records
            if record.environment_fingerprint not in fingerprint_ids
        }
        if missing:
            raise ValueError(
                "snapshot measurements reference missing environment "
                f"fingerprints: {sorted(missing)}"
            )
        class_ids = {
            device_performance_class_id(item)
            for item in self.device_performance_classes
        }
        for fingerprint in self.environment_fingerprints:
            class_id = fingerprint.device_performance_class_id
            if class_id is not None and class_id not in class_ids:
                raise ValueError(
                    "environment fingerprint references missing device "
                    f"performance class {class_id!r}"
                )
        for membership in self.device_performance_class_memberships:
            if membership.device_performance_class_id not in class_ids:
                raise ValueError(
                    "performance-class membership references missing class "
                    f"{membership.device_performance_class_id!r}"
                )
