"""MasterService — the Master-side control-plane facade (spec §33, §29-30).

Implements the :class:`~edgeshard.protocol.control.grpc_server.WorkerRegistryHandler`
protocol over the four P1F components, so it plugs directly into the
gRPC transport::

    MasterService
    ├── WorkerRegistry    (stable identity + capability)
    ├── SessionManager    (current session + heartbeat sequence)
    ├── StateStore        (latest accepted state + receive timestamps)
    ├── LivenessManager   (derived ONLINE/SUSPECT/OFFLINE)
    └── SnapshotBuilder   (immutable ClusterSnapshot views, P1H)

Semantics follow spec §29-30 exactly:

* registration upserts the stable record, invalidates any previous
  session immediately, stores the initial state with Master-local receive
  timestamps, and answers with a fresh ``session_id`` plus the configured
  heartbeat interval — leaving the Worker ONLINE by construction;
* a heartbeat is accepted only when the Worker is known, its session and
  instance are current, and its sequence strictly advances; every
  rejection travels as ``accepted=False`` with a mandatory detail and
  never touches stored state;
* capability updates require a current session and replace the stable
  capability in place.

Protocol validation proper (version, redundant-field agreement, enum
decoding) already happened in the mapper before a domain request reaches
this class (spec §40, §47); what remains here is the integrity check that
a reported ``capability_revision`` really fingerprints the reported
capability (spec §16).

Concurrency: everything is in-memory and single-event-loop — no method
awaits between reading and writing component state, so each RPC applies
atomically (spec §34: no database, no locks needed in Phase 1).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime

from edgeshard.cluster.capability import WorkerCapability, compute_capability_revision
from edgeshard.cluster.snapshot import ClusterSnapshot
from edgeshard.cluster.state import WorkerStatus
from edgeshard.control.master.config import MasterConfig
from edgeshard.control.master.liveness import LivenessManager
from edgeshard.control.master.registry import WorkerRegistry
from edgeshard.control.master.sessions import SessionManager
from edgeshard.control.master.snapshot import SnapshotBuilder
from edgeshard.control.master.state_store import StateStore
from edgeshard.protocol.control.mapper import (
    CONTROL_PROTOCOL_VERSION,
    ControlProtocolError,
    HeartbeatRequest,
    HeartbeatResponse,
    RegisterWorkerRequest,
    RegisterWorkerResponse,
    UpdateCapabilityRequest,
    UpdateCapabilityResponse,
)

logger = logging.getLogger("master.service")


def _utc_now() -> datetime:
    return datetime.now(UTC)


class MasterService:
    """In-memory Master state implementing the §29-30 acceptance rules."""

    def __init__(
        self,
        config: MasterConfig | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall: Callable[[], datetime] = _utc_now,
        session_factory: Callable[[], str] | None = None,
        snapshot_factory: Callable[[], str] | None = None,
    ) -> None:
        self._config = config or MasterConfig()
        self._monotonic = monotonic
        self._wall = wall
        self._registry = WorkerRegistry()
        if session_factory is None:
            self._sessions = SessionManager()
        else:
            self._sessions = SessionManager(session_factory=session_factory)
        self._states = StateStore()
        self._liveness = LivenessManager(self._states, self._config, monotonic=monotonic)
        if snapshot_factory is None:
            self._snapshot_builder = SnapshotBuilder(
                self._registry, self._sessions, self._states, self._liveness, wall=wall
            )
        else:
            self._snapshot_builder = SnapshotBuilder(
                self._registry,
                self._sessions,
                self._states,
                self._liveness,
                wall=wall,
                snapshot_factory=snapshot_factory,
            )

    # -- component access (read-oriented; SnapshotBuilder consumes these) ----

    @property
    def config(self) -> MasterConfig:
        return self._config

    @property
    def registry(self) -> WorkerRegistry:
        return self._registry

    @property
    def sessions(self) -> SessionManager:
        return self._sessions

    @property
    def states(self) -> StateStore:
        return self._states

    @property
    def liveness(self) -> LivenessManager:
        return self._liveness

    @property
    def snapshot_builder(self) -> SnapshotBuilder:
        return self._snapshot_builder

    def worker_status(self, worker_id: str) -> WorkerStatus:
        """Derived liveness of one Worker (spec §32); ``KeyError`` if unknown."""
        return self._liveness.status_of(worker_id)

    def build_snapshot(self) -> ClusterSnapshot:
        """Immutable point-in-time view of the whole cluster (spec §38).

        Synchronous, so it is atomic against registration/heartbeat on the
        single event loop: the returned snapshot never changes afterward.
        """
        return self._snapshot_builder.build()

    # -- WorkerRegistryHandler protocol (spec §29-30) ----------------------

    async def register_worker(
        self, request: RegisterWorkerRequest
    ) -> RegisterWorkerResponse:
        self._check_capability_revision(request.capability)
        worker_id = request.identity.worker_id

        # §29: upsert identity + capability, invalidate the old session,
        # create a new one, store the initial state, mark ONLINE.
        self._registry.upsert(request.identity, request.capability)
        session = self._sessions.open_session(worker_id, request.instance_id)
        self._states.record(
            worker_id,
            request.initial_state,
            monotonic=self._monotonic(),
            wall=self._wall(),
        )
        logger.info(
            "registration worker_id=%s instance_id=%s session_id=%s revision=%s",
            worker_id,
            request.instance_id,
            session.session_id,
            request.capability.capability_revision,
        )
        return RegisterWorkerResponse(
            session_id=session.session_id,
            heartbeat_interval_ms=self._config.heartbeat_interval_ms,
            server_protocol_version=CONTROL_PROTOCOL_VERSION,
        )

    async def heartbeat(self, request: HeartbeatRequest) -> HeartbeatResponse:
        detail = self._sessions.check_heartbeat(
            request.worker_id,
            request.instance_id,
            request.session_id,
            request.sequence_number,
        )
        if detail is not None:
            # §30: rejected heartbeats never overwrite newer state.
            logger.info(
                "heartbeat rejected worker_id=%s sequence=%d detail=%s",
                request.worker_id,
                request.sequence_number,
                detail,
            )
            return HeartbeatResponse(accepted=False, detail=detail)

        self._sessions.record_accepted_sequence(request.worker_id, request.sequence_number)
        self._states.record(
            request.worker_id,
            request.state,
            monotonic=self._monotonic(),
            wall=self._wall(),
        )
        self._warn_on_revision_drift(request.worker_id, request.capability_revision)
        logger.info(
            "heartbeat worker_id=%s session_id=%s sequence=%d",
            request.worker_id,
            request.session_id,
            request.sequence_number,
        )
        return HeartbeatResponse(accepted=True)

    async def update_capability(
        self, request: UpdateCapabilityRequest
    ) -> UpdateCapabilityResponse:
        detail = self._sessions.check_session(
            request.worker_id, request.instance_id, request.session_id
        )
        if detail is not None:
            logger.info(
                "capability update rejected worker_id=%s detail=%s",
                request.worker_id,
                detail,
            )
            return UpdateCapabilityResponse(accepted=False, detail=detail)

        self._check_capability_revision(request.capability)
        record = self._registry.get(request.worker_id)
        self._registry.upsert(record.identity, request.capability)
        logger.info(
            "capability updated worker_id=%s revision=%s",
            request.worker_id,
            request.capability.capability_revision,
        )
        return UpdateCapabilityResponse(accepted=True)

    # -- liveness task lifecycle (spec §37; used by `master serve` in P1G) --

    async def start(self) -> None:
        """Start the periodic liveness evaluation task."""
        await self._liveness.start()

    async def stop(self) -> None:
        """Stop the periodic liveness evaluation task."""
        await self._liveness.stop()

    async def __aenter__(self) -> MasterService:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    # -- integrity helpers --------------------------------------------------

    @staticmethod
    def _check_capability_revision(capability: WorkerCapability) -> None:
        """Fail loudly when the reported revision does not fingerprint the capability (§16, §47)."""
        expected = compute_capability_revision(capability)
        if capability.capability_revision != expected:
            raise ControlProtocolError(
                f"capability_revision mismatch: reported {capability.capability_revision!r}, "
                f"recomputed {expected!r}"
            )

    def _warn_on_revision_drift(self, worker_id: str, reported_revision: str) -> None:
        record = self._registry.find(worker_id)
        if record is not None and record.capability_revision != reported_revision:
            logger.warning(
                "heartbeat worker_id=%s reports capability_revision=%s but Master "
                "stores %s; capability drift — expected an UpdateCapability",
                worker_id,
                reported_revision,
                record.capability_revision,
            )
