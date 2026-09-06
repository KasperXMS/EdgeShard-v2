"""SessionManager tests (Phase 1 spec §35, §30, §51 Master sessions/Heartbeat order)."""

from __future__ import annotations

import itertools

import pytest

from edgeshard.control.master.sessions import Rejection, SessionInfo, SessionManager
from edgeshard.protocol.control.mapper import RejectionReason


def rejection(reason: RejectionReason) -> Rejection:
    """The canonical Rejection for a reason (details are fixed strings)."""
    details = {
        RejectionReason.UNKNOWN_WORKER: "unknown worker",
        RejectionReason.STALE_SESSION: "stale session",
        RejectionReason.INSTANCE_MISMATCH: "instance mismatch",
        RejectionReason.OUT_OF_ORDER: "out-of-order heartbeat",
    }
    return Rejection(reason, details[reason])

WORKER = "worker-1"
INSTANCE = "instance-1"


def counting_factory() -> tuple[itertools.count, SessionManager]:
    counter = itertools.count(1)
    return counter, SessionManager(session_factory=lambda: f"session-{next(counter)}")


def test_open_session_starts_at_sequence_zero() -> None:
    manager = SessionManager()
    info = manager.open_session(WORKER, INSTANCE)

    assert info.worker_id == WORKER
    assert info.instance_id == INSTANCE
    assert info.session_id
    assert info.last_accepted_sequence == 0
    assert manager.current(WORKER) == info
    assert manager.current("other-worker") is None


def test_second_registration_invalidates_old_session_immediately() -> None:
    """§51: registration creates session; second registration invalidates old."""
    _counter, manager = counting_factory()
    first = manager.open_session(WORKER, INSTANCE)
    second = manager.open_session(WORKER, "instance-2")

    assert second.session_id != first.session_id
    assert manager.current(WORKER) == second
    # The old session is rejected the moment the new one exists (§35),
    # even though the old instance_id was also valid once. The typed reason
    # tells the superseded Agent to stop rather than re-register (§30).
    assert manager.check_heartbeat(WORKER, INSTANCE, first.session_id, 1) == rejection(
        RejectionReason.STALE_SESSION
    )


def test_check_heartbeat_acceptance_then_rejections() -> None:
    manager = SessionManager()
    info = manager.open_session(WORKER, INSTANCE)

    assert manager.check_heartbeat(WORKER, INSTANCE, info.session_id, 1) is None
    manager.record_accepted_sequence(WORKER, 1)

    assert manager.check_heartbeat("unknown", INSTANCE, info.session_id, 2) == rejection(
        RejectionReason.UNKNOWN_WORKER
    )
    assert manager.check_heartbeat(WORKER, INSTANCE, "bogus-session", 2) == rejection(
        RejectionReason.STALE_SESSION
    )
    assert manager.check_heartbeat(WORKER, "bogus-instance", info.session_id, 2) == rejection(
        RejectionReason.INSTANCE_MISMATCH
    )
    assert manager.check_heartbeat(WORKER, INSTANCE, info.session_id, 1) == rejection(
        RejectionReason.OUT_OF_ORDER
    )


def test_heartbeat_order_per_spec_51() -> None:
    """§51 Heartbeat order: seq=5 accepted, 4 rejected, duplicate 5 rejected, 6 accepted."""
    manager = SessionManager()
    info = manager.open_session(WORKER, INSTANCE)

    def check(sequence: int) -> Rejection | None:
        return manager.check_heartbeat(WORKER, INSTANCE, info.session_id, sequence)

    out_of_order = rejection(RejectionReason.OUT_OF_ORDER)
    assert check(5) is None
    manager.record_accepted_sequence(WORKER, 5)

    assert check(4) == out_of_order
    assert check(5) == out_of_order  # duplicate, not just older
    assert check(6) is None
    manager.record_accepted_sequence(WORKER, 6)

    assert manager.current(WORKER) is not None
    assert manager.current(WORKER).last_accepted_sequence == 6


def test_sequence_restarts_with_new_session() -> None:
    """After re-registration, heartbeats begin from 1 again (§30)."""
    manager = SessionManager()
    first = manager.open_session(WORKER, INSTANCE)
    manager.record_accepted_sequence(WORKER, 7)

    second = manager.open_session(WORKER, "instance-2")
    assert second.last_accepted_sequence == 0
    assert manager.check_heartbeat(WORKER, "instance-2", second.session_id, 1) is None
    assert first.session_id != second.session_id


def test_record_accepted_sequence_refuses_to_rewind() -> None:
    manager = SessionManager()
    manager.open_session(WORKER, INSTANCE)
    manager.record_accepted_sequence(WORKER, 5)

    with pytest.raises(ValueError, match="does not advance"):
        manager.record_accepted_sequence(WORKER, 5)
    with pytest.raises(ValueError, match="does not advance"):
        manager.record_accepted_sequence(WORKER, 4)
    assert manager.current(WORKER) is not None
    assert manager.current(WORKER).last_accepted_sequence == 5


def test_check_session_for_updates() -> None:
    manager = SessionManager()
    info = manager.open_session(WORKER, INSTANCE)

    assert manager.check_session(WORKER, INSTANCE, info.session_id) is None
    assert manager.check_session("unknown", INSTANCE, info.session_id) == rejection(
        RejectionReason.UNKNOWN_WORKER
    )
    assert manager.check_session(WORKER, INSTANCE, "bogus") == rejection(
        RejectionReason.STALE_SESSION
    )
    assert manager.check_session(WORKER, "bogus", info.session_id) == rejection(
        RejectionReason.INSTANCE_MISMATCH
    )


def test_session_info_validation() -> None:
    with pytest.raises(ValueError, match="instance_id"):
        SessionInfo(worker_id=WORKER, instance_id="", session_id="s", last_accepted_sequence=0)
    with pytest.raises(ValueError, match="negative"):
        SessionInfo(
            worker_id=WORKER, instance_id=INSTANCE, session_id="s", last_accepted_sequence=-1
        )
