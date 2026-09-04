"""Cluster identity domain tests (Phase 1 spec §10-11, §51)."""

from __future__ import annotations

import dataclasses
import uuid

import pytest

from edgeshard.cluster.identity import DeviceIdentity, DeviceKind, WorkerIdentity

WORKER_ID = str(uuid.uuid4())


def test_worker_identity_holds_fields() -> None:
    identity = WorkerIdentity(
        worker_id=WORKER_ID, hostname="host-a", agent_version="0.1.0", protocol_version="1"
    )
    assert identity.worker_id == WORKER_ID
    assert identity.hostname == "host-a"
    assert identity.agent_version == "0.1.0"
    assert identity.protocol_version == "1"


def test_worker_identity_is_frozen() -> None:
    identity = WorkerIdentity(
        worker_id=WORKER_ID, hostname="host-a", agent_version="0.1.0", protocol_version="1"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.hostname = "host-b"  # type: ignore[misc]


@pytest.mark.parametrize(
    "field", ["worker_id", "hostname", "agent_version", "protocol_version"]
)
def test_worker_identity_rejects_empty_fields(field: str) -> None:
    kwargs = {
        "worker_id": WORKER_ID,
        "hostname": "host-a",
        "agent_version": "0.1.0",
        "protocol_version": "1",
        field: "",
    }
    with pytest.raises(ValueError, match="must not be empty"):
        WorkerIdentity(**kwargs)  # type: ignore[arg-type]


def test_device_kind_values() -> None:
    assert [kind.value for kind in DeviceKind] == ["cpu", "gpu", "npu", "other"]


def test_device_identity_holds_fields() -> None:
    identity = DeviceIdentity(device_id="GPU-abc", kind=DeviceKind.GPU, local_locator="cuda:0")
    assert identity.device_id == "GPU-abc"
    assert identity.kind is DeviceKind.GPU
    assert identity.local_locator == "cuda:0"


@pytest.mark.parametrize("field", ["device_id", "local_locator"])
def test_device_identity_rejects_empty_fields(field: str) -> None:
    kwargs = {"device_id": "GPU-abc", "kind": DeviceKind.GPU, "local_locator": "cuda:0", field: ""}
    with pytest.raises(ValueError, match="must not be empty"):
        DeviceIdentity(**kwargs)
