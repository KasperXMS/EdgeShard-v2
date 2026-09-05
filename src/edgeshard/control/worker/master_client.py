"""Worker Agent loops against the Master (Phase 1 spec §27, §44, §56 P1G).

``WorkerAgent.run`` implements the second half of the §27 lifecycle:
register with the Master, then heartbeat a fresh local inspection at the
cadence the Master dictates. On Master loss the Agent backs off
exponentially and re-registers; on session invalidation it re-registers
immediately. A stale registration session is never restored (§27) — every
reconnect performs a full ``RegisterWorker`` with the *current* local state
under a stable per-process ``instance_id`` (§10.2) and a fresh sequence
starting at 1 (§30).

Capability revisions are checked every cycle: if local discovery now
produces a different revision than the Master acknowledged, a full
``UpdateCapability`` is sent before the next heartbeat (§16).

Transport failures (unreachable Master, deadline exceeded, connection
reset) are transient and trigger backoff. ``INVALID_ARGUMENT`` aborts are
*not* transient — they mean the Agent itself violated the protocol
(version mismatch, malformed request) and must fail loudly (§47).

The Master-provided ``heartbeat_interval_ms`` from registration is
authoritative for the heartbeat cadence; the config's
``heartbeat_interval_s`` is only the Worker's preference and a difference
is logged once at registration.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import grpc

from edgeshard.control.worker.agent import LocalInspection, inspect_local_worker
from edgeshard.control.worker.config import WorkerConfig
from edgeshard.control.worker.identity import new_instance_id
from edgeshard.model.errors import EdgeShardError
from edgeshard.protocol.control.grpc_client import WorkerRegistryClient
from edgeshard.protocol.control.mapper import (
    CONTROL_PROTOCOL_VERSION,
    HeartbeatRequest,
    HeartbeatResponse,
    RegisterWorkerRequest,
    RegisterWorkerResponse,
    UpdateCapabilityRequest,
    UpdateCapabilityResponse,
)

logger = logging.getLogger("worker.master_client")

DEFAULT_RPC_TIMEOUT_S = 10.0
"""Per-RPC deadline: a Master that accepts TCP but never answers must not
hang the Agent forever; exceeding it is treated as Master loss (§27)."""


class WorkerAgentError(EdgeShardError):
    """Fatal Agent-side error: missing endpoint or protocol rejection."""


class _MasterUnavailable(Exception):
    """Transient transport failure: back off, then register again (§27)."""


class _SessionInvalid(Exception):
    """Master rejected session or sequence: re-register immediately (§27)."""


class ControlClient(Protocol):
    """The transport surface ``WorkerAgent`` needs (spec §28)."""

    async def register_worker(
        self, request: RegisterWorkerRequest, *, timeout: float | None = None
    ) -> RegisterWorkerResponse: ...

    async def heartbeat(
        self, request: HeartbeatRequest, *, timeout: float | None = None
    ) -> HeartbeatResponse: ...

    async def update_capability(
        self, request: UpdateCapabilityRequest, *, timeout: float | None = None
    ) -> UpdateCapabilityResponse: ...

    async def close(self) -> None: ...


Inspector = Callable[[WorkerConfig], Awaitable[LocalInspection]]
Sleeper = Callable[[float], Awaitable[None]]
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class _Session:
    """What the Master granted at registration; discarded on any loss."""

    worker_id: str
    session_id: str
    heartbeat_interval_s: float
    acknowledged_revision: str


class WorkerAgent:
    """Registration / heartbeat / reconnect loops of the Worker Agent (§44).

    ``client``, ``inspector``, ``sleeper`` and ``clock`` are injectable for
    tests; an injected client is owned by the caller, a default-constructed
    :class:`WorkerRegistryClient` is closed by ``run``.
    """

    def __init__(
        self,
        config: WorkerConfig,
        *,
        client: ControlClient | None = None,
        inspector: Inspector = inspect_local_worker,
        sleeper: Sleeper = asyncio.sleep,
        clock: Clock = _utc_now,
        rpc_timeout_s: float = DEFAULT_RPC_TIMEOUT_S,
    ) -> None:
        if config.worker.master is None:
            raise WorkerAgentError(
                "worker.master.endpoint is required to serve a Worker; "
                "see examples/configs/worker.yaml"
            )
        self._config = config
        self._endpoint = config.worker.master.endpoint
        self._client = client
        self._inspector = inspector
        self._sleeper = sleeper
        self._clock = clock
        self._rpc_timeout_s = rpc_timeout_s
        self._instance_id = new_instance_id()
        self._stopping = False

    @property
    def instance_id(self) -> str:
        """Stable for the lifetime of this process (§10.2, §51)."""
        return self._instance_id

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def request_stop(self) -> None:
        """Ask ``run`` to exit at the next checkpoint (between RPCs)."""
        self._stopping = True

    async def run(self) -> None:
        """Serve until stopped or a fatal error occurs (§27 outer loop)."""
        client = self._client
        owns_client = client is None
        if client is None:
            client = WorkerRegistryClient(self._endpoint)
        reconnect = self._config.worker.reconnect
        delay = reconnect.initial_delay_s
        try:
            while not self._stopping:
                try:
                    session = await self._register(client)
                except _MasterUnavailable as exc:
                    logger.warning(
                        "master unreachable at %s (%s); retrying in %.1fs",
                        self._endpoint,
                        exc,
                        delay,
                    )
                    await self._sleeper(delay)
                    delay = min(delay * 2, reconnect.max_delay_s)
                    continue
                delay = reconnect.initial_delay_s  # backoff resets on success
                try:
                    await self._heartbeat_loop(client, session)
                except _MasterUnavailable as exc:
                    logger.warning(
                        "lost master connection (%s); re-registering in %.1fs",
                        exc,
                        delay,
                    )
                    await self._sleeper(delay)
                    delay = min(delay * 2, reconnect.max_delay_s)
                except _SessionInvalid as exc:
                    # §27: never restore a stale session; re-register at once.
                    logger.warning(
                        "session invalidated (%s); re-registering immediately", exc
                    )
        finally:
            if owns_client:
                await client.close()

    # ------------------------------------------------------------------
    # Loop bodies
    # ------------------------------------------------------------------

    async def _register(self, client: ControlClient) -> _Session:
        """One full §27 registration: fresh inspection, fresh session."""
        inspection = await self._inspector(self._config)
        request = RegisterWorkerRequest(
            protocol_version=CONTROL_PROTOCOL_VERSION,
            instance_id=self._instance_id,
            identity=inspection.identity,
            capability=inspection.capability,
            initial_state=inspection.state,
        )
        response = await self._call(client.register_worker, request, "registration")

        interval_s = response.heartbeat_interval_ms / 1000.0
        configured = self._config.worker.heartbeat_interval_s
        if interval_s != configured:
            logger.info(
                "master cadence %.3fs overrides configured heartbeat_interval_s=%.3fs",
                interval_s,
                configured,
            )
        logger.info(
            "registered worker_id=%s instance_id=%s session_id=%s revision=%s",
            inspection.identity.worker_id,
            self._instance_id,
            response.session_id,
            inspection.capability.capability_revision,
        )
        return _Session(
            worker_id=inspection.identity.worker_id,
            session_id=response.session_id,
            heartbeat_interval_s=interval_s,
            acknowledged_revision=inspection.capability.capability_revision,
        )

    async def _heartbeat_loop(self, client: ControlClient, session: _Session) -> None:
        """Heartbeat fresh state until stopped or the session ends (§27)."""
        sequence = 1  # §30: sequences restart at 1 with every new session
        while not self._stopping:
            # Sleep first: registration already carried the current state.
            await self._sleeper(session.heartbeat_interval_s)
            if self._stopping:
                return
            inspection = await self._inspector(self._config)
            revision = inspection.capability.capability_revision
            if revision != session.acknowledged_revision:
                await self._update_capability(client, session, inspection)
                session.acknowledged_revision = revision
            request = HeartbeatRequest(
                worker_id=session.worker_id,
                instance_id=self._instance_id,
                session_id=session.session_id,
                sequence_number=sequence,
                capability_revision=revision,
                state=inspection.state,
                worker_reported_at=self._clock(),
            )
            response = await self._call(client.heartbeat, request, "heartbeat")
            if not response.accepted:
                raise _SessionInvalid(f"heartbeat rejected: {response.detail}")
            logger.info(
                "heartbeat accepted worker_id=%s sequence=%d revision=%s",
                session.worker_id,
                sequence,
                revision,
            )
            sequence += 1

    async def _update_capability(
        self, client: ControlClient, session: _Session, inspection: LocalInspection
    ) -> None:
        """§16: a revision change requires a full capability retransmission."""
        revision = inspection.capability.capability_revision
        logger.info(
            "capability revision changed %s -> %s; sending UpdateCapability",
            session.acknowledged_revision,
            revision,
        )
        request = UpdateCapabilityRequest(
            worker_id=session.worker_id,
            instance_id=self._instance_id,
            session_id=session.session_id,
            capability=inspection.capability,
        )
        response = await self._call(
            client.update_capability, request, "capability update"
        )
        if not response.accepted:
            raise _SessionInvalid(f"capability update rejected: {response.detail}")

    # ------------------------------------------------------------------
    # Transport error classification (§47: transient vs fatal)
    # ------------------------------------------------------------------

    async def _call[RequestT, ResponseT](
        self,
        method: Callable[..., Awaitable[ResponseT]],
        request: RequestT,
        label: str,
    ) -> ResponseT:
        try:
            return await method(request, timeout=self._rpc_timeout_s)
        except grpc.aio.AioRpcError as exc:
            if exc.code() is grpc.StatusCode.INVALID_ARGUMENT:
                raise WorkerAgentError(
                    f"master rejected {label} as a protocol violation: {exc.details()}"
                ) from exc
            raise _MasterUnavailable(f"{exc.code().name}: {exc.details()}") from exc
        except OSError as exc:
            raise _MasterUnavailable(str(exc) or exc.__class__.__name__) from exc
