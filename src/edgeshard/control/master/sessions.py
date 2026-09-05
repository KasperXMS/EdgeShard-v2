"""Registration session tracking (Phase 1 spec §35).

One session per registration: the SessionManager stores, per ``worker_id``,
the current ``instance_id``, the Master-awarded ``session_id``, and the last
accepted heartbeat sequence number.

A new registration for an existing Worker invalidates the previous session
*immediately* (spec §35): the old ``session_id`` stops being current the
moment ``open_session`` returns, so an old Agent process heartbeating with
it is rejected as stale and can never update state again (spec §30, §52
Test C). Sequence numbers restart from zero with every new session —
heartbeats begin at 1 after registration (spec §30).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SessionInfo:
    """The one valid session of a Worker (spec §35)."""

    worker_id: str
    instance_id: str
    session_id: str
    last_accepted_sequence: int

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id must not be empty")
        if not self.instance_id:
            raise ValueError("instance_id must not be empty")
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.last_accepted_sequence < 0:
            raise ValueError("last_accepted_sequence must not be negative")


class SessionManager:
    """In-memory current-session bookkeeping (spec §35).

    Validation order matters for the rejection details: session identity
    is checked before instance identity, so a heartbeat from a superseded
    Agent process reports ``stale session`` even though its instance_id is
    also outdated.
    """

    def __init__(
        self, session_factory: Callable[[], str] = lambda: str(uuid.uuid4())
    ) -> None:
        self._sessions: dict[str, SessionInfo] = {}
        self._session_factory = session_factory

    def open_session(self, worker_id: str, instance_id: str) -> SessionInfo:
        """Award a fresh session, invalidating any previous one immediately."""
        info = SessionInfo(
            worker_id=worker_id,
            instance_id=instance_id,
            session_id=self._session_factory(),
            last_accepted_sequence=0,
        )
        self._sessions[worker_id] = info
        return info

    def current(self, worker_id: str) -> SessionInfo | None:
        """The Worker's current session, or ``None`` if it never registered."""
        return self._sessions.get(worker_id)

    def check_heartbeat(
        self,
        worker_id: str,
        instance_id: str,
        session_id: str,
        sequence_number: int,
    ) -> str | None:
        """Validate a heartbeat against the current session (spec §30).

        Returns ``None`` when the heartbeat must be accepted, otherwise the
        rejection detail. Acceptance requires all four conditions: the
        Worker exists, the session is current, the instance matches, and
        the sequence strictly increases over the last accepted one — a
        duplicate or late sequence never overwrites newer state.
        """
        info = self._sessions.get(worker_id)
        if info is None:
            return "unknown worker"
        if info.session_id != session_id:
            return "stale session"
        if info.instance_id != instance_id:
            return "instance mismatch"
        if sequence_number <= info.last_accepted_sequence:
            return "out-of-order heartbeat"
        return None

    def check_session(
        self, worker_id: str, instance_id: str, session_id: str
    ) -> str | None:
        """Session-only validation for non-heartbeat RPCs (UpdateCapability)."""
        info = self._sessions.get(worker_id)
        if info is None:
            return "unknown worker"
        if info.session_id != session_id:
            return "stale session"
        if info.instance_id != instance_id:
            return "instance mismatch"
        return None

    def record_accepted_sequence(self, worker_id: str, sequence_number: int) -> None:
        """Persist the sequence of an accepted heartbeat.

        Callers must only invoke this after :meth:`check_heartbeat` returned
        ``None``; the monotonicity invariant is re-asserted here so a
        programming error fails loudly instead of rewinding state.
        """
        info = self._sessions[worker_id]
        if sequence_number <= info.last_accepted_sequence:
            raise ValueError(
                f"sequence {sequence_number} does not advance "
                f"last accepted {info.last_accepted_sequence}"
            )
        self._sessions[worker_id] = replace(info, last_accepted_sequence=sequence_number)
