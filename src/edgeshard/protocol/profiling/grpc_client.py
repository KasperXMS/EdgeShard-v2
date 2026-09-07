"""Async gRPC client for the Worker profiling plane (Phase 2 spec §41).

The Master's ProfilingController speaks only domain objects across this
API: requests and responses are the mapper's frozen dataclasses, never
protobuf DTOs (mirroring the Phase 1 ``WorkerRegistryClient`` discipline).
The channel is insecure today (trusted LAN / overlay, spec §48);
construction funnels through one place so TLS credentials can be added
later without touching callers.
"""

from __future__ import annotations

import grpc

from edgeshard.protocol.profiling import mapper
from edgeshard.protocol.profiling.pb import profiling_pb2_grpc as pb_grpc

MAX_PROFILING_MESSAGE_BYTES: int = 8 * 1024 * 1024
"""Profiling-plane send/receive ceiling: payload strings carry measurement
records with per-iteration samples — kilobyte-to-megabyte scale — so a
generous but explicit bound fails loudly on pathological messages."""


def profiling_channel_options() -> list[tuple[str, int]]:
    """gRPC channel/server options shared by both sides of the profiling plane."""
    return [
        ("grpc.max_send_message_length", MAX_PROFILING_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", MAX_PROFILING_MESSAGE_BYTES),
    ]


class WorkerProfilingClient:
    """Client for one Worker's ``WorkerProfilingService`` endpoint."""

    def __init__(self, endpoint: str) -> None:
        if not endpoint:
            raise ValueError("profiling endpoint must not be empty")
        self._endpoint = endpoint
        # TLS: this insecure channel is the single transport chokepoint
        # (spec §48); swap for a secure channel when credentials land.
        self._channel = grpc.aio.insecure_channel(
            endpoint, options=profiling_channel_options()
        )
        # Generated grpc stub code is untyped (only pb2 ships .pyi stubs).
        self._stub = pb_grpc.WorkerProfilingServiceStub(self._channel)  # type: ignore[no-untyped-call]

    @property
    def endpoint(self) -> str:
        return self._endpoint

    async def prepare_profiling_session(
        self,
        request: mapper.PrepareProfilingSessionRequest,
        *,
        timeout: float | None = None,
    ) -> mapper.PrepareProfilingSessionResponse:
        wire = await self._stub.PrepareProfilingSession(
            mapper.prepare_request_to_wire(request), timeout=timeout
        )
        return mapper.prepare_response_from_wire(wire)

    async def run_profiling_case(
        self,
        request: mapper.RunProfilingCaseRequest,
        *,
        timeout: float | None = None,
    ) -> mapper.RunProfilingCaseResponse:
        wire = await self._stub.RunProfilingCase(
            mapper.run_case_request_to_wire(request), timeout=timeout
        )
        return mapper.run_case_response_from_wire(wire)

    async def get_profiling_case(
        self,
        request: mapper.GetProfilingCaseRequest,
        *,
        timeout: float | None = None,
    ) -> mapper.GetProfilingCaseResponse:
        wire = await self._stub.GetProfilingCase(
            mapper.get_case_request_to_wire(request), timeout=timeout
        )
        return mapper.get_case_response_from_wire(wire)

    async def cancel_profiling_case(
        self,
        request: mapper.CancelProfilingCaseRequest,
        *,
        timeout: float | None = None,
    ) -> mapper.CancelProfilingCaseResponse:
        wire = await self._stub.CancelProfilingCase(
            mapper.cancel_case_request_to_wire(request), timeout=timeout
        )
        return mapper.cancel_case_response_from_wire(wire)

    async def close_profiling_session(
        self,
        request: mapper.CloseProfilingSessionRequest,
        *,
        timeout: float | None = None,
    ) -> mapper.CloseProfilingSessionResponse:
        wire = await self._stub.CloseProfilingSession(
            mapper.close_session_request_to_wire(request), timeout=timeout
        )
        return mapper.close_session_response_from_wire(wire)

    async def close(self) -> None:
        await self._channel.close()

    async def __aenter__(self) -> WorkerProfilingClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


class ProfilingAdminClient:
    """Client for the Master's ``ProfilingAdminService`` endpoint (spec §49).

    Spoken by the CLI only: it submits intents and reads back status and
    snapshots, never generated protobuf objects.
    """

    def __init__(self, endpoint: str) -> None:
        if not endpoint:
            raise ValueError("admin endpoint must not be empty")
        self._endpoint = endpoint
        self._channel = grpc.aio.insecure_channel(
            endpoint, options=profiling_channel_options()
        )
        # Generated grpc stub code is untyped (only pb2 ships .pyi stubs).
        self._stub = pb_grpc.ProfilingAdminServiceStub(self._channel)  # type: ignore[no-untyped-call]

    @property
    def endpoint(self) -> str:
        return self._endpoint

    async def start_experiment(
        self,
        request: mapper.StartExperimentRequest,
        *,
        timeout: float | None = None,
    ) -> mapper.StartExperimentResponse:
        wire = await self._stub.StartExperiment(
            mapper.start_experiment_request_to_wire(request), timeout=timeout
        )
        return mapper.start_experiment_response_from_wire(wire)

    async def get_experiment(
        self,
        request: mapper.GetExperimentRequest,
        *,
        timeout: float | None = None,
    ) -> mapper.GetExperimentResponse:
        wire = await self._stub.GetExperiment(
            mapper.get_experiment_request_to_wire(request), timeout=timeout
        )
        return mapper.get_experiment_response_from_wire(wire)

    async def cancel_experiment(
        self,
        request: mapper.CancelExperimentRequest,
        *,
        timeout: float | None = None,
    ) -> mapper.CancelExperimentResponse:
        wire = await self._stub.CancelExperiment(
            mapper.cancel_experiment_request_to_wire(request), timeout=timeout
        )
        return mapper.cancel_experiment_response_from_wire(wire)

    async def build_profile_snapshot(
        self,
        request: mapper.BuildProfileSnapshotRequest,
        *,
        timeout: float | None = None,
    ) -> mapper.BuildProfileSnapshotResponse:
        wire = await self._stub.BuildProfileSnapshot(
            mapper.build_snapshot_request_to_wire(request), timeout=timeout
        )
        return mapper.build_snapshot_response_from_wire(wire)

    async def close(self) -> None:
        await self._channel.close()

    async def __aenter__(self) -> ProfilingAdminClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
