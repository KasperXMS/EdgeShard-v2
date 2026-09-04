"""Runtime and model inventory domain tests (Phase 1 spec §19-20)."""

from __future__ import annotations

import pytest

from edgeshard.cluster.inventory import (
    ModelAvailability,
    ModelInventoryEntry,
    RuntimeInstanceState,
    RuntimeStatus,
)


def make_runtime_instance(runtime_id: str = "runtime-1", backend: str = "edgeshard-shard"):
    return RuntimeInstanceState(
        runtime_id=runtime_id,
        backend=backend,
        execution_id="execution-1",
        status=RuntimeStatus.RUNNING,
        device_ids=("gpu-0",),
        container_id="abc123",
        endpoint="127.0.0.1:51051",
        model_local_name="tiny-llama",
    )


def test_runtime_status_values() -> None:
    assert [status.value for status in RuntimeStatus] == [
        "created",
        "running",
        "stopped",
        "failed",
        "unknown",
    ]


def test_runtime_instance_state_assembles() -> None:
    instance = make_runtime_instance()
    assert instance.runtime_id == "runtime-1"
    assert instance.status is RuntimeStatus.RUNNING
    assert instance.device_ids == ("gpu-0",)


def test_runtime_instance_optional_fields_may_be_absent() -> None:
    """An agent restart can observe containers with partial labels (spec §19)."""
    instance = RuntimeInstanceState(
        runtime_id="runtime-2",
        backend="vllm",
        execution_id=None,
        status=RuntimeStatus.UNKNOWN,
        device_ids=(),
        container_id=None,
        endpoint=None,
        model_local_name=None,
    )
    assert instance.execution_id is None
    assert instance.container_id is None


@pytest.mark.parametrize("field", ["runtime_id", "backend"])
def test_runtime_instance_rejects_empty_required_fields(field: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        make_runtime_instance(**{field: ""})


def test_model_availability_values() -> None:
    assert [status.value for status in ModelAvailability] == ["ready", "incomplete", "invalid"]


def test_model_inventory_entry_assembles() -> None:
    entry = ModelInventoryEntry(
        local_name="tiny-llama",
        model_id="tiny/llama",
        revision="main",
        size_bytes=1024,
        status=ModelAvailability.READY,
    )
    assert entry.local_name == "tiny-llama"
    assert entry.status is ModelAvailability.READY


def test_model_inventory_entry_rejects_empty_local_name() -> None:
    with pytest.raises(ValueError, match="local_name"):
        ModelInventoryEntry(
            local_name="",
            model_id=None,
            revision=None,
            size_bytes=None,
            status=ModelAvailability.INVALID,
        )


def test_model_inventory_entry_rejects_negative_size() -> None:
    with pytest.raises(ValueError, match="size_bytes"):
        ModelInventoryEntry(
            local_name="tiny-llama",
            model_id=None,
            revision=None,
            size_bytes=-1,
            status=ModelAvailability.INCOMPLETE,
        )
