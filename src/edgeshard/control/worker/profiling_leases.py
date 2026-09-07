"""Lightweight device leases for Worker profiling (Phase 2 spec §39).

Before a profiling session benchmarks anything it *reserves* its target
devices against the Worker's own freshly reported state, so a case never
runs on a GPU that is already serving a runtime, is saturated, or is
claimed by another profiling session. The reservation is deliberately
lightweight and Worker-local — it is not a Master-visible state transition
and it never *corrects* a measurement for background load (§39: v1 refuses
a busy device instead of trying to subtract its interference).

The checks read only what :class:`~edgeshard.cluster.state.DeviceState`
actually reports (§52.2 — a metric the backend cannot supply is ``None``
and never implies "busy"): a device must be ``AVAILABLE``, must carry no
``running_runtime_ids``, and its utilization (when reported) must sit below
a configurable floor. An optional memory-pressure floor is applied only
when a caller supplies a device→memory-pool resolver and the pool reports
``available_bytes``; absent either, no memory fact is invented.

Leases are the runner's responsibility to release on *every* exit path —
success, failure, cancellation, session close, and runner shutdown — so the
manager keeps them keyed by session id and exposes :meth:`release_session`
and :meth:`release_all`. A leaked lease would permanently shrink the
profileable device set, so the release paths are exercised directly by
tests (spec §39 DoD).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from edgeshard.cluster.state import DeviceAvailability, DeviceState, WorkerState
from edgeshard.model.errors import EdgeShardError

logger = logging.getLogger("worker.profiling.leases")

DEFAULT_UTILIZATION_FLOOR = 5.0
"""Percent utilization above which a device is considered busy (§39).

A near-idle reading (driver bookkeeping, a momentary blip) must not block a
reservation, but a device doing real work must. The floor is intentionally
low: §39 refuses rather than correcting, so a genuinely busy device is
deferred, never measured through.
"""

MemoryPoolResolver = Callable[[str], tuple[str, ...]]
"""Maps a device id to the memory-pool ids whose pressure gates it (§14.2).

Returns ``()`` when the device has no modeled pool (CPU, or a backend that
reports no VRAM pool) — the lease then applies no memory floor rather than
guessing one (§52.2).
"""


class DeviceBusyError(EdgeShardError):
    """A target device cannot be reserved right now (spec §39).

    Carries the offending ``device_id`` and the specific reason so the
    runner can turn it into a typed ``DEVICE_BUSY`` refusal with a useful
    detail (§41) instead of a generic rejection.
    """

    def __init__(self, device_id: str, reason: str) -> None:
        super().__init__(f"device {device_id!r} is not reservable: {reason}")
        self.device_id = device_id
        self.reason = reason


@dataclass(frozen=True)
class DeviceLease:
    """One session's reservation of one device (spec §39)."""

    session_id: str
    device_id: str


class DeviceLeaseManager:
    """Reserves profileable devices against fresh Worker state (§39).

    The manager holds no thread/async locks of its own: the Worker profiling
    runner serializes reservation and release on the event loop, so the
    in-memory maps here are only ever touched from one coroutine at a time.
    ``utilization_floor`` and ``min_available_bytes``/``memory_pool_resolver``
    are injectable so tests can drive the busy verdicts without real GPUs.
    """

    def __init__(
        self,
        *,
        utilization_floor: float = DEFAULT_UTILIZATION_FLOOR,
        min_available_bytes: int | None = None,
        memory_pool_resolver: MemoryPoolResolver | None = None,
    ) -> None:
        if not 0.0 <= utilization_floor <= 100.0:
            raise ValueError(
                f"utilization_floor must be within [0, 100], got {utilization_floor}"
            )
        if min_available_bytes is not None and min_available_bytes < 0:
            raise ValueError(
                f"min_available_bytes must not be negative, got {min_available_bytes}"
            )
        if min_available_bytes is not None and memory_pool_resolver is None:
            raise ValueError(
                "a memory floor requires a memory_pool_resolver to locate the pool"
            )
        self._utilization_floor = utilization_floor
        self._min_available_bytes = min_available_bytes
        self._memory_pool_resolver = memory_pool_resolver
        # device_id -> session_id (a device is leased by at most one session).
        self._leased_by: dict[str, str] = {}

    @property
    def leased_device_ids(self) -> tuple[str, ...]:
        """Devices currently reserved, in acquisition order."""
        return tuple(self._leased_by)

    def leases_for_session(self, session_id: str) -> tuple[DeviceLease, ...]:
        """Every lease held by one session (spec §39 release accounting)."""
        return tuple(
            DeviceLease(session_id=session_id, device_id=device_id)
            for device_id, holder in self._leased_by.items()
            if holder == session_id
        )

    def acquire(
        self,
        session_id: str,
        device_ids: Iterable[str],
        state: WorkerState,
    ) -> tuple[DeviceLease, ...]:
        """Reserve ``device_ids`` for ``session_id`` against fresh ``state``.

        All-or-nothing: the first device that fails any §39 check aborts the
        whole reservation and rolls back the devices already marked, so a
        partial lease never strands a device. On success the leases are held
        until :meth:`release_session` / :meth:`release_all`.
        """
        if not session_id:
            raise ValueError("session_id must not be empty")
        requested = tuple(dict.fromkeys(device_ids))  # dedupe, preserve order
        if not requested:
            raise ValueError("a device lease requires at least one device_id")
        states = {device.device_id: device for device in state.device_states}

        acquired: list[str] = []
        try:
            for device_id in requested:
                self._check_device(session_id, device_id, states.get(device_id), state)
                self._leased_by[device_id] = session_id
                acquired.append(device_id)
        except BaseException:
            for device_id in acquired:  # roll back the partial reservation
                self._leased_by.pop(device_id, None)
            raise
        leases = tuple(
            DeviceLease(session_id=session_id, device_id=device_id)
            for device_id in acquired
        )
        logger.info(
            "leased devices for session %s: %s",
            session_id,
            ", ".join(lease.device_id for lease in leases),
        )
        return leases

    def release_session(self, session_id: str) -> tuple[str, ...]:
        """Release every device held by ``session_id``; idempotent (§39).

        Returns the released device ids (empty when the session held none),
        so the caller can log/confirm the release on success, failure,
        cancellation and close alike.
        """
        released = tuple(
            device_id
            for device_id, holder in list(self._leased_by.items())
            if holder == session_id
        )
        for device_id in released:
            del self._leased_by[device_id]
        if released:
            logger.info(
                "released devices for session %s: %s", session_id, ", ".join(released)
            )
        return released

    def release_all(self) -> tuple[str, ...]:
        """Release every outstanding lease (runner shutdown, §39)."""
        released = tuple(self._leased_by)
        self._leased_by.clear()
        if released:
            logger.info("released all profiling leases: %s", ", ".join(released))
        return released

    # ------------------------------------------------------------------
    # §39 busy inspection
    # ------------------------------------------------------------------

    def _check_device(
        self,
        session_id: str,
        device_id: str,
        device: DeviceState | None,
        state: WorkerState,
    ) -> None:
        if device is None:
            # Device compatibility: a device the Worker does not report is
            # not one this runner may benchmark (§39), never a silent pass.
            raise DeviceBusyError(device_id, "device is not reported by this Worker")
        holder = self._leased_by.get(device_id)
        if holder is not None and holder != session_id:
            raise DeviceBusyError(
                device_id, f"device is already leased by profiling session {holder!r}"
            )
        if device.availability is not DeviceAvailability.AVAILABLE:
            raise DeviceBusyError(
                device_id, f"device availability is {device.availability.value!r}"
            )
        if device.running_runtime_ids:
            raise DeviceBusyError(
                device_id,
                "device is serving running runtime(s): "
                + ", ".join(device.running_runtime_ids),
            )
        if (
            device.utilization is not None
            and device.utilization > self._utilization_floor
        ):
            raise DeviceBusyError(
                device_id,
                f"device utilization {device.utilization:.1f}% exceeds the "
                f"{self._utilization_floor:.1f}% reservation floor",
            )
        self._check_memory(device_id, state)

    def _check_memory(self, device_id: str, state: WorkerState) -> None:
        if self._min_available_bytes is None or self._memory_pool_resolver is None:
            return
        available_by_pool = {
            pool.memory_pool_id: pool.available_bytes for pool in state.memory_states
        }
        for pool_id in self._memory_pool_resolver(device_id):
            available = available_by_pool.get(pool_id)
            # A pool with no reported available_bytes is not a memory fact we
            # can gate on (§52.2) — skip it rather than assume pressure.
            if available is not None and available < self._min_available_bytes:
                raise DeviceBusyError(
                    device_id,
                    f"memory pool {pool_id!r} has {available} bytes available, "
                    f"below the {self._min_available_bytes}-byte floor",
                )


__all__ = [
    "DEFAULT_UTILIZATION_FLOOR",
    "DeviceBusyError",
    "DeviceLease",
    "DeviceLeaseManager",
    "MemoryPoolResolver",
]
