"""IdentityManager and identity helper tests (Phase 1 spec §10-11, §51)."""

from __future__ import annotations

import socket
import uuid
from pathlib import Path

import pytest

import edgeshard
from edgeshard.control.worker.identity import (
    CONTROL_PROTOCOL_VERSION,
    IdentityError,
    IdentityManager,
    build_worker_identity,
    derive_cpu_device_id,
    new_instance_id,
)


def test_worker_id_created_once_and_persists_across_restarts(tmp_path: Path) -> None:
    """Spec §51: worker_id created once, persists across Agent restart."""
    path = tmp_path / "nested" / "worker-id"
    first = IdentityManager(path).load_or_create()
    second = IdentityManager(path).load_or_create()  # a restarted Agent
    assert first == second
    assert uuid.UUID(first).version == 4
    assert path.read_text(encoding="utf-8").strip() == first


def test_existing_identity_file_is_reused(tmp_path: Path) -> None:
    path = tmp_path / "worker-id"
    known = str(uuid.uuid4())
    path.write_text(known + "\n", encoding="utf-8")
    assert IdentityManager(path).load_or_create() == known


def test_distinct_paths_get_distinct_ids(tmp_path: Path) -> None:
    first = IdentityManager(tmp_path / "a" / "worker-id").load_or_create()
    second = IdentityManager(tmp_path / "b" / "worker-id").load_or_create()
    assert first != second


def test_corrupt_identity_file_fails_loudly(tmp_path: Path) -> None:
    """Spec §47: cannot load/create identity is fatal, never regenerated."""
    path = tmp_path / "worker-id"
    path.write_text("not-a-uuid", encoding="utf-8")
    with pytest.raises(IdentityError, match="valid UUID"):
        IdentityManager(path).load_or_create()


def test_unreadable_identity_path_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "somewhere"
    path.mkdir()  # a directory is neither readable nor writable as a file
    with pytest.raises(IdentityError):
        IdentityManager(path).load_or_create()


def test_instance_id_changes_each_process_start() -> None:
    """Spec §51: instance_id changes each process start."""
    assert new_instance_id() != new_instance_id()
    uuid.UUID(new_instance_id())


def test_derive_cpu_device_id_is_stable_per_worker() -> None:
    """Spec §11: worker-derived device identity, stable across reboots."""
    worker_id = str(uuid.uuid4())
    assert derive_cpu_device_id(worker_id) == derive_cpu_device_id(worker_id)
    assert derive_cpu_device_id(worker_id) != derive_cpu_device_id(str(uuid.uuid4()))
    uuid.UUID(derive_cpu_device_id(worker_id))


def test_build_worker_identity_fields() -> None:
    worker_id = str(uuid.uuid4())
    identity = build_worker_identity(worker_id, hostname="host-a")
    assert identity.worker_id == worker_id
    assert identity.hostname == "host-a"
    assert identity.agent_version == edgeshard.__version__
    assert identity.protocol_version == CONTROL_PROTOCOL_VERSION


def test_build_worker_identity_defaults_hostname() -> None:
    identity = build_worker_identity(str(uuid.uuid4()))
    assert identity.hostname == socket.gethostname()


def test_build_worker_identity_rejects_empty_worker_id() -> None:
    with pytest.raises(ValueError, match="worker_id"):
        build_worker_identity("")
