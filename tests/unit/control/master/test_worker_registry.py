"""WorkerRegistry tests (Phase 1 spec §34, §52 Test C semantics)."""

from __future__ import annotations

import dataclasses

import pytest

from edgeshard.control.master.registry import WorkerRecord, WorkerRegistry
from factories import finalize, make_jetson_capability, make_rtx_capability, make_worker_identity


def test_upsert_then_get() -> None:
    registry = WorkerRegistry()
    identity = make_worker_identity()
    capability = make_rtx_capability()

    record = registry.upsert(identity, capability)

    assert record.identity == identity
    assert record.capability == capability
    assert record.worker_id == identity.worker_id
    assert record.capability_revision == capability.capability_revision
    assert registry.get(identity.worker_id) == record


def test_get_unknown_worker_raises() -> None:
    registry = WorkerRegistry()
    with pytest.raises(KeyError):
        registry.get("never-registered")
    assert registry.find("never-registered") is None


def test_reregistration_updates_same_entry() -> None:
    """§52 Test C: same worker_id → still the same registry entry."""
    registry = WorkerRegistry()
    identity = make_worker_identity()
    registry.upsert(identity, make_rtx_capability())

    # The Agent restarted on a new image: capability facts changed, identity persists.
    evolved = finalize(
        dataclasses.replace(
            make_rtx_capability(),
            runtime_platforms=make_rtx_capability().runtime_platforms[:1],
        )
    )
    registry.upsert(identity, evolved)

    assert len(registry) == 1
    assert registry.get(identity.worker_id).capability == evolved


def test_list_workers_sorted_and_never_shrinks() -> None:
    """§32: OFFLINE Workers remain in the registry; nothing is deleted."""
    registry = WorkerRegistry()
    identities = [make_worker_identity() for _ in range(3)]
    for identity in identities:
        registry.upsert(identity, make_jetson_capability())

    listed = registry.list_workers()
    assert [record.worker_id for record in listed] == sorted(
        identity.worker_id for identity in identities
    )
    # The registry offers no removal API at all — entries are permanent in Phase 1.
    assert len(registry) == 3


def test_record_rejects_empty_worker_id() -> None:
    with pytest.raises(ValueError, match="worker_id"):
        # WorkerIdentity itself already rejects the empty id; WorkerRecord
        # re-asserts it so a hand-built record cannot sneak past.
        identity = dataclasses.replace(make_worker_identity(), worker_id="")
        WorkerRecord(identity=identity, capability=make_rtx_capability())
