"""Runtime inventory tests (Phase 1 spec §19)."""

from __future__ import annotations

from typing import Any

import pytest

from edgeshard.cluster.inventory import RuntimeStatus
from edgeshard.control.worker.runtime_inventory import scan_runtime_inventory
from edgeshard.runtime import labels

GPU_UUID_1 = "GPU-69c27179-5df5-d790-4b75-6cf18a4d2b1c"
GPU_UUID_2 = "GPU-1c34a739-3c6f-4bd1-8e5d-889491d40bb1"


class FakeContainer:
    def __init__(
        self,
        container_labels: dict[str, str],
        *,
        status: str = "running",
        container_id: str = "a" * 64,
        exit_code: int | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self.labels = container_labels
        self.status = status
        self.id = container_id
        state: dict[str, Any] = {}
        if exit_code is not None:
            state["ExitCode"] = exit_code
        self.attrs: Any = {"State": state, **(attrs or {})}


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
    # Nothing observable in bare inspect data: attribution stays empty (§19).
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


# -- device attribution (§19: observed facts only, never guessed) -----------


def device_requests(*device_ids: str) -> dict[str, Any]:
    return {"HostConfig": {"DeviceRequests": [{"Driver": "nvidia", "DeviceIDs": list(device_ids)}]}}


def visible_devices(value: str) -> dict[str, Any]:
    return {"Config": {"Env": [f"NVIDIA_VISIBLE_DEVICES={value}", "PATH=/usr/bin"]}}


def test_device_request_ids_are_attributed() -> None:
    container = FakeContainer(
        managed_labels(), attrs=device_requests(GPU_UUID_1, GPU_UUID_2)
    )
    (instance,) = scan_runtime_inventory(FakeDockerClient([container]))
    assert instance.device_ids == (GPU_UUID_1, GPU_UUID_2)


def test_device_requests_win_over_environment() -> None:
    attrs = {**device_requests(GPU_UUID_1), **visible_devices(GPU_UUID_2)}
    container = FakeContainer(managed_labels(), attrs=attrs)
    (instance,) = scan_runtime_inventory(FakeDockerClient([container]))
    assert instance.device_ids == (GPU_UUID_1,)


def test_env_uuids_are_attributed_and_deduplicated() -> None:
    container = FakeContainer(
        managed_labels(),
        attrs=visible_devices(f"{GPU_UUID_1},{GPU_UUID_2},{GPU_UUID_1}"),
    )
    (instance,) = scan_runtime_inventory(FakeDockerClient([container]))
    assert instance.device_ids == (GPU_UUID_1, GPU_UUID_2)


def test_env_bare_uuid_without_prefix_is_attributed() -> None:
    bare = GPU_UUID_1.removeprefix("GPU-")
    container = FakeContainer(managed_labels(), attrs=visible_devices(bare))
    (instance,) = scan_runtime_inventory(FakeDockerClient([container]))
    assert instance.device_ids == (bare,)


@pytest.mark.parametrize(
    "value",
    [
        "all",  # shorthand: cannot be resolved to stable ids (§11)
        "none",
        "0,1",  # CUDA ordinals are never identity
        f"{GPU_UUID_1},0",  # mixed: ambiguous, report nothing
        "",
        "void",
    ],
)
def test_env_shorthands_are_never_guessed(value: str) -> None:
    container = FakeContainer(managed_labels(), attrs=visible_devices(value))
    (instance,) = scan_runtime_inventory(FakeDockerClient([container]))
    assert instance.device_ids == ()


def test_empty_device_request_falls_through_to_env() -> None:
    attrs: dict[str, Any] = {
        "HostConfig": {"DeviceRequests": [{"Driver": "nvidia", "DeviceIDs": []}]},
        **visible_devices(GPU_UUID_1),
    }
    container = FakeContainer(managed_labels(), attrs=attrs)
    (instance,) = scan_runtime_inventory(FakeDockerClient([container]))
    assert instance.device_ids == (GPU_UUID_1,)


def test_malformed_inspect_data_never_crashes() -> None:
    attrs: dict[str, Any] = {
        "HostConfig": {"DeviceRequests": ["not-a-dict", {"DeviceIDs": [None, 3, ""]}]},
        "Config": {"Env": "NVIDIA_VISIBLE_DEVICES=all"},  # env is not even a list
    }
    container = FakeContainer(managed_labels(), attrs=attrs)
    (instance,) = scan_runtime_inventory(FakeDockerClient([container]))
    assert instance.device_ids == ()

    class NoAttrs:
        labels = managed_labels()
        status = "running"
        id = "c" * 64

    (bare,) = scan_runtime_inventory(FakeDockerClient([NoAttrs()]))
    assert bare.status is RuntimeStatus.RUNNING
    assert bare.device_ids == ()


# -- endpoint attribution (§19: only when unambiguous) ----------------------


def ports(*bindings: tuple[str, str | None, str]) -> dict[str, Any]:
    """bindings: (container_port, host_ip, host_port); protocol in the key."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for container_port, host_ip, host_port in bindings:
        entry: dict[str, Any] = {"HostPort": host_port}
        if host_ip is not None:
            entry["HostIp"] = host_ip
        grouped.setdefault(container_port, []).append(entry)
    return {"NetworkSettings": {"Ports": grouped}}


def test_single_tcp_binding_maps_endpoint() -> None:
    container = FakeContainer(
        managed_labels(), attrs=ports(("8000/tcp", "192.168.1.5", "32768"))
    )
    (instance,) = scan_runtime_inventory(FakeDockerClient([container]))
    assert instance.endpoint == "192.168.1.5:32768"


@pytest.mark.parametrize("wildcard", ["", "0.0.0.0", "::"])
def test_wildcard_host_ip_normalizes_to_loopback(wildcard: str) -> None:
    container = FakeContainer(
        managed_labels(), attrs=ports(("8000/tcp", wildcard, "32768"))
    )
    (instance,) = scan_runtime_inventory(FakeDockerClient([container]))
    assert instance.endpoint == "127.0.0.1:32768"


def test_multiple_tcp_bindings_report_no_endpoint() -> None:
    container = FakeContainer(
        managed_labels(),
        attrs=ports(("8000/tcp", "", "32768"), ("8001/tcp", "", "32769")),
    )
    (instance,) = scan_runtime_inventory(FakeDockerClient([container]))
    assert instance.endpoint is None  # no single answer, never a guess


def test_udp_and_unpublished_ports_are_ignored() -> None:
    container = FakeContainer(
        managed_labels(),
        attrs=ports(
            ("8000/tcp", "", "32768"),
            ("9000/udp", "", "32770"),  # not tcp
            ("8001/tcp", "", ""),  # published without a host port
            ("8002/tcp", None, "32771"),  # no HostIp key at all: still counts
        ),
    )
    # Two usable tcp bindings (8000, 8002): ambiguous, so None.
    (instance,) = scan_runtime_inventory(FakeDockerClient([container]))
    assert instance.endpoint is None

    only_one = FakeContainer(
        managed_labels(), attrs=ports(("9000/udp", "", "1"), ("8000/tcp", "", "32768"))
    )
    (instance,) = scan_runtime_inventory(FakeDockerClient([only_one]))
    assert instance.endpoint == "127.0.0.1:32768"


def test_null_port_bindings_report_no_endpoint() -> None:
    """Docker reports unpublished ports as ``"8000/tcp": None``."""
    container = FakeContainer(
        managed_labels(), attrs={"NetworkSettings": {"Ports": {"8000/tcp": None}}}
    )
    (instance,) = scan_runtime_inventory(FakeDockerClient([container]))
    assert instance.endpoint is None


# -- model attribution (§19: operator label, never path guessing) -----------


def test_model_local_name_label_is_reported() -> None:
    container_labels = {**managed_labels(), labels.MODEL_LOCAL_NAME: "tiny-llama"}
    (instance,) = scan_runtime_inventory(
        FakeDockerClient([FakeContainer(container_labels)])
    )
    assert instance.model_local_name == "tiny-llama"


def test_empty_model_local_name_label_reports_none() -> None:
    container_labels = {**managed_labels(), labels.MODEL_LOCAL_NAME: ""}
    (instance,) = scan_runtime_inventory(
        FakeDockerClient([FakeContainer(container_labels)])
    )
    assert instance.model_local_name is None
