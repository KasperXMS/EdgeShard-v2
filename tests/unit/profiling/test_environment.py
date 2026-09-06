"""Environment fingerprint and performance class identity (spec §9-10, §52.3)."""

from __future__ import annotations

import pytest

from edgeshard.profiling.domain.environment import (
    DevicePerformanceClass,
    EnvironmentFingerprint,
    MemoryModel,
    device_performance_class_id,
    environment_fingerprint_id,
)
from edgeshard.profiling.domain.hashing import normalized_items

GOLDEN_CLASS_ID = "7ed7b70ac1723246b6d1f0f07e0373ff2a206d7452ae6de0b6234750db8468dd"
GOLDEN_FINGERPRINT_ID = "edf7dfd13244ab096df908ce3907456ee2a88244eba5c1b0c0b824b85621dd4f"


def _rtx4090_class(**overrides: object) -> DevicePerformanceClass:
    base: dict[str, object] = {
        "vendor": "nvidia",
        "accelerator_model": "rtx4090",
        "memory_model": MemoryModel.DISCRETE,
        "backend_family": "cuda",
        "architecture": "8.9",
    }
    base.update(overrides)
    return DevicePerformanceClass(**base)  # type: ignore[arg-type]


def test_performance_class_id_is_pinned() -> None:
    assert device_performance_class_id(_rtx4090_class()) == GOLDEN_CLASS_ID


def test_distinct_physical_devices_share_one_class() -> None:
    """§10: three RTX 4090 UUIDs → one compatible performance class."""
    assert device_performance_class_id(_rtx4090_class()) == device_performance_class_id(
        _rtx4090_class()
    )
    jetson = device_performance_class_id(
        _rtx4090_class(
            accelerator_model="agx-orin-64gb",
            memory_model=MemoryModel.SHARED,
            architecture="8.7",
        )
    )
    assert jetson != GOLDEN_CLASS_ID


def test_software_versions_order_is_irrelevant() -> None:
    def with_versions(versions: dict[str, str]) -> str:
        return device_performance_class_id(
            _rtx4090_class(software_versions=normalized_items(versions, "s"))
        )

    assert with_versions({"cuda": "12.6", "torch": "2.8"}) == with_versions(
        {"torch": "2.8", "cuda": "12.6"}
    )
    assert with_versions({"cuda": "12.4", "torch": "2.8"}) != with_versions(
        {"cuda": "12.6", "torch": "2.8"}
    )


def test_performance_class_validation() -> None:
    with pytest.raises(ValueError, match="vendor"):
        _rtx4090_class(vendor="")
    with pytest.raises(ValueError, match="software_versions"):
        _rtx4090_class(software_versions=(("cuda", ""),))


def _fingerprint(**overrides: object) -> EnvironmentFingerprint:
    base: dict[str, object] = {
        "backend": "torch",
        "profiling_implementation_revision": "0.1.0",
        "torch_version": "2.8.0",
        "dtype": "bf16",
    }
    base.update(overrides)
    return EnvironmentFingerprint(**base)  # type: ignore[arg-type]


def test_fingerprint_id_is_pinned() -> None:
    assert environment_fingerprint_id(_fingerprint()) == GOLDEN_FINGERPRINT_ID


def test_fingerprint_id_excludes_provenance_only_fields() -> None:
    """§9: worker_id/device_id are provenance, not compatibility identity."""
    baseline = environment_fingerprint_id(_fingerprint())
    elsewhere = environment_fingerprint_id(
        _fingerprint(worker_id="worker-b", device_id="GPU-uuid-2")
    )
    assert baseline == elsewhere


def test_fingerprint_id_includes_compatibility_context() -> None:
    baseline = environment_fingerprint_id(_fingerprint())
    assert baseline != environment_fingerprint_id(_fingerprint(torch_version="2.7.0"))
    assert baseline != environment_fingerprint_id(_fingerprint(capability_revision="rev-2"))
    assert baseline != environment_fingerprint_id(
        _fingerprint(profiling_implementation_revision="0.2.0")
    )
    assert baseline != environment_fingerprint_id(_fingerprint(dtype="fp16"))


def test_fingerprint_validation() -> None:
    with pytest.raises(ValueError, match="backend"):
        _fingerprint(backend="")
    with pytest.raises(ValueError, match="profiling_implementation_revision"):
        _fingerprint(profiling_implementation_revision="")
    with pytest.raises(ValueError, match="cuda_version"):
        _fingerprint(cuda_version="")
