"""Runtime inventory tests (Phase 1 spec §19)."""

from __future__ import annotations

from typing import Any

import pytest

from edgeshard.cluster.inventory import RuntimeStatus
from edgeshard.control.worker.runtime_inventory import scan_runtime_inventory
from edgeshard.runtime import labels


class FakeContainer:
    def __init__(
        self,
        container_labels: dict[str, str],
        *,
        status: str = "running",
        container_id: str = "a" * 64,
        exit_code: int | None = None,
    ) -> None:
        self.labels = container_labels
        self.status = status
        self.id = container_id
        state: dict[str, Any] = {}
        if exit_code is not None:
            state["ExitCode"] = exit_code
        self.attrs = {"State": state}


class FakeContainers:
    def __init__(self, containers: list[FakeContainer]) -> None:
        self._containers = containers
        self.last_list_kwargs: dict[str, Any] | None = None

    def list(self, **kwargs: Any) -> list[FakeContainer]:
        self.last_list_kwargs = kwargs
        return list(self._containers)


class FakeDockerClient:
    def __init__(self, containers: list[FakeContainer] | None = None) -> None:
        self.containers = FakeContainers(containers or [])


def managed_labels(
    *,
    runtime_id: str = "runtime-1",
    execution_id: str = "execution-1",
    backend: str = "edgeshard-shard",
) -> dict[str, str]:
    return labels.managed_container_labels(
        execution_id=execution_id, runtime_id=runtime_id, backend=backend
    )


def test_lists_only_managed_containers() -> None:
    client = FakeDockerClient()
    scan_runtime_inventory(client)
    assert client.containers.last_list_kwargs == {
        "all": True,
        "filters": {"label": labels.MANAGED},
    }


def test_running_container_maps_labels_and_state() -> None:
    container = FakeContainer(managed_labels())
    (instance,) = scan_runtime_inventory(FakeDockerClient([container]))
    assert instance.runtime_id == "runtime-1"
    assert instance.backend == "edgeshard-shard"
    assert instance.execution_id == "execution-1"
    assert instance.status is RuntimeStatus.RUNNING
    assert instance.container_id == "a" * 64
    # Device attribution lands with NVML/Jetson discovery (P1C/P1D).
    assert instance.device_ids == ()
    assert instance.endpoint is None
    assert instance.model_local_name is None


@pytest.mark.parametrize(
    ("status", "exit_code", "expected"),
    [
        ("created", None, RuntimeStatus.CREATED),
        ("running", None, RuntimeStatus.RUNNING),
        ("exited", 0, RuntimeStatus.STOPPED),
        ("exited", 137, RuntimeStatus.FAILED),
        ("exited", None, RuntimeStatus.STOPPED),
        ("dead", None, RuntimeStatus.FAILED),
        ("paused", None, RuntimeStatus.UNKNOWN),
        ("restarting", None, RuntimeStatus.UNKNOWN),
        ("", None, RuntimeStatus.UNKNOWN),
    ],
)
def test_status_mapping(status: str, exit_code: int | None, expected: RuntimeStatus) -> None:
    container = FakeContainer(managed_labels(), status=status, exit_code=exit_code)
    (instance,) = scan_runtime_inventory(FakeDockerClient([container]))
    assert instance.status is expected


def test_container_without_identity_labels_is_skipped() -> None:
    incomplete = {k: v for k, v in managed_labels().items() if k != labels.RUNTIME_ID}
    containers = [
        FakeContainer(incomplete, container_id="b" * 64),
        FakeContainer(managed_labels(runtime_id="runtime-2")),
    ]
    instances = scan_runtime_inventory(FakeDockerClient(containers))
    assert [instance.runtime_id for instance in instances] == ["runtime-2"]


def test_missing_execution_label_reports_none() -> None:
    without_execution = {
        k: v for k, v in managed_labels().items() if k != labels.EXECUTION_ID
    }
    (instance,) = scan_runtime_inventory(FakeDockerClient([FakeContainer(without_execution)]))
    assert instance.execution_id is None
