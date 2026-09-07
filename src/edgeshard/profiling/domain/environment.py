"""Environment fingerprints and device performance classes (spec §9-10).

Two distinct identities that must never be conflated (§52.3):

* physical device identity (Phase 1 ``DeviceIdentity``) is **provenance** —
  where a measurement happened;
* ``DevicePerformanceClass`` is the **reuse identity** — which physically
  distinct devices are compatible enough to share measurements (three
  RTX 4090 UUIDs → one class), gated by the small verification suite of
  §29.

``EnvironmentFingerprint`` records the full measurement context needed to
judge reuse later (§9). Physical provenance and the host-wide
``capability_revision`` are excluded from ``environment_fingerprint_id``;
device-relevant software and hardware compatibility remain in it. Volatile
telemetry belongs to measurement context, never to the fingerprint (§9).

``MemoryModel`` mirrors ``edgeshard.cluster.capability.MemoryModel`` by
value; the domain package imports nothing but the stdlib (P2A DoD), so the
bridge lives in the implementation layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from edgeshard.profiling.domain.hashing import canonical_sha256, check_normalized_items


class MemoryModel(StrEnum):
    """How memory is physically attached to devices (mirrors Phase 1 §12)."""

    DISCRETE = "discrete"
    SHARED = "shared"


@dataclass(frozen=True)
class DevicePerformanceClass:
    """Compatibility class of physically distinct devices (spec §10).

    The class key intentionally stays small and is not overfitted: vendor,
    accelerator model, architecture/compute capability, memory model,
    backend family, an optional class-wide dtype regime, and the relevant
    software major versions as normalized ``(name, version)`` items.

    A new physical device may reuse an existing class only after passing
    the verification benchmark of §29; that gate is runner logic, not part
    of this identity.
    """

    vendor: str
    accelerator_model: str
    memory_model: MemoryModel
    backend_family: str
    architecture: str | None = None
    dtype: str | None = None
    software_versions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.vendor:
            raise ValueError("vendor must not be empty")
        if not self.accelerator_model:
            raise ValueError("accelerator_model must not be empty")
        if not self.backend_family:
            raise ValueError("backend_family must not be empty")
        if self.architecture is not None and not self.architecture:
            raise ValueError("architecture must not be empty when present")
        if self.dtype is not None and not self.dtype:
            raise ValueError("dtype must not be empty when present")
        check_normalized_items(self.software_versions, "software_versions")
        for _, version in self.software_versions:
            if not version:
                raise ValueError("software_versions values must not be empty")


def device_performance_class_id(performance_class: DevicePerformanceClass) -> str:
    """Canonical SHA-256 identity of a performance class (spec §7)."""
    return canonical_sha256(("device_performance_class", performance_class))


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """Full recorded context of one measurement environment (spec §9).

    ``backend`` names the execution/measurement backend (e.g. ``torch``,
    ``iperf3``); ``profiling_implementation_revision`` versions the
    profiling code itself so measurements produced by different profiler
    generations never mix silently.
    """

    backend: str
    profiling_implementation_revision: str

    device_performance_class_id: str | None = None
    device_performance_class: DevicePerformanceClass | None = None
    capability_revision: str | None = None

    torch_version: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    backend_revision: str | None = None

    model_revision: str | None = None
    dtype: str | None = None
    quantization: str | None = None

    worker_id: str | None = None
    """Provenance only — excluded from the fingerprint id (§9)."""

    device_id: str | None = None
    """Provenance only — excluded from the fingerprint id (§9)."""

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must not be empty")
        if not self.profiling_implementation_revision:
            raise ValueError("profiling_implementation_revision must not be empty")
        if self.device_performance_class is not None:
            expected = device_performance_class_id(self.device_performance_class)
            if self.device_performance_class_id != expected:
                raise ValueError(
                    "device_performance_class_id does not match the embedded "
                    "DevicePerformanceClass"
                )
        for field_name in (
            "device_performance_class_id",
            "capability_revision",
            "torch_version",
            "cuda_version",
            "driver_version",
            "backend_revision",
            "model_revision",
            "dtype",
            "quantization",
            "worker_id",
            "device_id",
        ):
            value = getattr(self, field_name)
            if value is not None and not value:
                raise ValueError(f"{field_name} must not be empty when present")


def environment_fingerprint_id(fingerprint: EnvironmentFingerprint) -> str:
    """Canonical SHA-256 identity of the compatibility context (spec §7, §9).

    Excludes provenance-only fields (``worker_id``, ``device_id``) and the
    host-wide ``capability_revision`` so unrelated NIC/container discovery
    changes do not invalidate device profiles. Reuse queries (§28) filter on
    the performance class and compatibility fields, never on volatile
    telemetry, which is not recorded here at all.
    """
    return canonical_sha256(
        (
            "environment_fingerprint",
            fingerprint.device_performance_class_id,
            fingerprint.torch_version,
            fingerprint.cuda_version,
            fingerprint.driver_version,
            fingerprint.backend,
            fingerprint.backend_revision,
            fingerprint.model_revision,
            fingerprint.dtype,
            fingerprint.quantization,
            fingerprint.profiling_implementation_revision,
        )
    )


def environment_instance_id(fingerprint: EnvironmentFingerprint) -> str:
    """Identity of the full provenance-bearing fingerprint record.

    ``environment_fingerprint_id`` is deliberately a compatibility hash and
    may be shared by multiple physical devices. Persistence therefore uses a
    second identity that includes provenance, while measurements keep the
    compatibility hash used by reuse queries.
    """
    return canonical_sha256(("environment_instance", fingerprint))


@dataclass(frozen=True)
class DevicePerformanceClassMembership:
    """Verified (or pending) membership of one physical device in a class."""

    device_performance_class_id: str
    worker_id: str
    device_id: str
    verified: bool
    verified_at: datetime | None = None
    evidence_measurement_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.device_performance_class_id:
            raise ValueError("device_performance_class_id must not be empty")
        if not self.worker_id or not self.device_id:
            raise ValueError("performance-class membership requires worker_id/device_id")
        if self.verified and self.verified_at is None:
            raise ValueError("verified membership requires verified_at")
        if not self.verified and self.verified_at is not None:
            raise ValueError("unverified membership must not carry verified_at")
        if self.verified_at is not None and (
            self.verified_at.tzinfo is None
            or self.verified_at.tzinfo.utcoffset(self.verified_at) is None
        ):
            raise ValueError("verified_at must be timezone-aware")
        if len(set(self.evidence_measurement_ids)) != len(
            self.evidence_measurement_ids
        ):
            raise ValueError("evidence_measurement_ids must be unique")
