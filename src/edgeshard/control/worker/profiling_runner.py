"""Worker ProfilingRunner: the WorkerProfilingService behavior (§37-39, §41).

The long-lived runner the Worker Agent hosts next to its inspector (§37):
it reuses the process, the inspector's fresh state, and — per session — the
loaded checkpoint and reserved devices, instead of creating and destroying
any of them per case. Structure::

    Worker Agent
    ├── LocalWorkerInspector          (Phase 1, shared)
    └── WorkerProfilingRunner         (this module)
        ├── ProfilingSessionManager   (§38 sessions, §44 ledger, §41 gate)
        ├── DeviceLeaseManager        (§39 reservations)
        ├── ModelSessionLoader        (§38 load-once checkpoints)
        └── TransformerLayer / Module / Operator / Network profilers

Execution model (v1): ``RunProfilingCase`` is *synchronous* — the benchmark
runs inside the RPC (in a worker thread, so the event loop keeps serving
heartbeats and cancels) and the response carries the terminal outcome.
``CancelProfilingCase`` therefore wins in two ways: a case cancelled before
its run replays the recorded ``CANCELLED`` decision (§44/§50 — a duplicate
run never re-benchmarks), and a case cancelled *mid-run* keeps the
cancellation while the finished measurement is discarded, never published
(§41). There is no mid-benchmark abort in v1.

Failure discipline (§42): benchmark problems surface as ``accepted=True``
responses whose :class:`CaseOutcome` carries a typed
:class:`ProfilingFailure` — an RPC-level refusal (``accepted=False`` with a
:class:`ProfilingRejection`) is reserved for transport-level verdicts:
stale registrations, unknown/closed sessions, kind mismatches and §39 lease
refusals. Unexpected exceptions are logged and converted to
``INTERNAL_ERROR`` outcomes rather than crashing the RPC.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import socket
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Protocol

import torch

from edgeshard.control.worker.agent import LocalInspection
from edgeshard.control.worker.identity import (
    derive_cpu_device_id,
    derive_jetson_gpu_device_id,
)
from edgeshard.control.worker.profiling_leases import (
    DeviceBusyError,
    DeviceLeaseManager,
)
from edgeshard.control.worker.profiling_model_loader import (
    LoadedModelSession,
    ModelSessionLoader,
    TorchModelSessionLoader,
    resolve_model_source,
)
from edgeshard.control.worker.profiling_sessions import (
    ProfilingSessionManager,
    ProfilingSessionRecord,
    RegistrationTokens,
    SessionRefused,
    utc_now,
)
from edgeshard.profiling.benchmark.harness import InstrumentationBundle
from edgeshard.profiling.domain.environment import (
    DevicePerformanceClass,
    EnvironmentFingerprint,
    MemoryModel,
    device_performance_class_id,
    environment_fingerprint_id,
)
from edgeshard.profiling.domain.experiment import (
    CaseOutcome,
    CaseState,
    ModelCaseSpec,
    NetworkCaseSpec,
    ProfilingCase,
    ProfilingErrorCategory,
    ProfilingFailure,
)
from edgeshard.profiling.domain.model import ModelCharacterization
from edgeshard.profiling.domain.network import ProbeKind
from edgeshard.profiling.domain.session import ProfilingSessionKind
from edgeshard.profiling.domain.signature import ProfilingGranularity
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.instrumentation.memory import (
    CudaAllocatorMemoryProbe,
    PhysicalMemoryProbe,
)
from edgeshard.profiling.instrumentation.telemetry import (
    DeviceObservation,
    TelemetryContextCollector,
)
from edgeshard.profiling.instrumentation.timing import CudaEventTimer, WallClockTimer
from edgeshard.profiling.model.adapters.base import module_device
from edgeshard.profiling.model.layer_profiler import TransformerLayerProfiler
from edgeshard.profiling.model.module_profiler import ModuleProfiler
from edgeshard.profiling.network.classifier import WorkerNetworkFacts
from edgeshard.profiling.network.profiler import NetworkProfiler
from edgeshard.profiling.operator.profiler import OperatorProfiler
from edgeshard.protocol.profiling.mapper import (
    CancelProfilingCaseRequest,
    CancelProfilingCaseResponse,
    CloseProfilingSessionRequest,
    CloseProfilingSessionResponse,
    GetProfilingCaseRequest,
    GetProfilingCaseResponse,
    PrepareIperfServerRequest,
    PrepareIperfServerResponse,
    PrepareProfilingSessionRequest,
    PrepareProfilingSessionResponse,
    ProfilingRejection,
    RunProfilingCaseRequest,
    RunProfilingCaseResponse,
    StopIperfServerRequest,
    StopIperfServerResponse,
)
from edgeshard.runtime.model_store import ModelStore

logger = logging.getLogger("worker.profiling.runner")

PROFILING_IMPLEMENTATION_REVISION = "0.1.0"
"""Versions the profiling code itself inside every environment fingerprint
(§9): measurements produced by different runner generations never mix."""


class StateInspector(Protocol):
    """The fresh-state source the runner shares with the Worker Agent (§37)."""

    async def inspect(self) -> LocalInspection: ...


DeviceResolver = Callable[[str, str], torch.device]
"""Maps ``(device_id, worker_id)`` to a torch device; typed failures only."""

InstrumentationFactory = Callable[[torch.device], InstrumentationBundle]
ModelLoaderFactory = Callable[[torch.device], ModelSessionLoader]


def resolve_torch_device(device_id: str, worker_id: str) -> torch.device:
    """Default device resolution over the Phase 1 stable device ids (§11).

    The id spaces are exactly the ones the Worker's own probes report: the
    derived CPU/Jetson ids, NVML GPU UUIDs, and the ``nvml-gpu-<index>``
    fallback. Anything unresolvable is a typed ``INTERNAL_ERROR`` — a device
    the Worker cannot name to torch is never silently remapped (§52.2).
    """
    if device_id == derive_cpu_device_id(worker_id):
        return torch.device("cpu")
    if device_id == derive_jetson_gpu_device_id(worker_id):
        return _require_cuda_device(0, device_id)
    if device_id.startswith("nvml-gpu-"):
        try:
            return _require_cuda_device(int(device_id.removeprefix("nvml-gpu-")), device_id)
        except ValueError as exc:
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                f"fallback device id {device_id!r} carries no usable CUDA index",
                {"device_id": device_id},
            ) from exc
    return _require_cuda_device(_cuda_index_for_uuid(device_id), device_id)


def _require_cuda_device(index: int, device_id: str) -> torch.device:
    if not torch.cuda.is_available() or not 0 <= index < torch.cuda.device_count():
        raise ProfilingError(
            ProfilingErrorCategory.INTERNAL_ERROR,
            f"device {device_id!r} maps to CUDA index {index} but no such CUDA "
            "device is available on this worker",
            {"device_id": device_id, "cuda_index": index},
        )
    return torch.device("cuda", index)


def _cuda_index_for_uuid(device_id: str) -> int:
    """NVML ordinal of one GPU UUID device id (host-local, never guessed)."""
    from edgeshard.control.worker.discovery.nvidia import import_nvml

    try:
        nvml = import_nvml()
        handle = nvml.nvmlDeviceGetHandleByUUID(device_id)
        return int(nvml.nvmlDeviceGetIndex(handle))
    except Exception as exc:
        raise ProfilingError(
            ProfilingErrorCategory.INTERNAL_ERROR,
            f"cannot resolve GPU device {device_id!r} to a CUDA index via NVML: {exc}",
            {"device_id": device_id},
        ) from exc


def _available_tcp_port(bind_address: str | None) -> int:
    """Choose a destination-local ephemeral port for a temporary server."""
    family = socket.AF_INET6 if bind_address and ":" in bind_address else socket.AF_INET
    host = bind_address or ("::" if family == socket.AF_INET6 else "0.0.0.0")
    with socket.socket(family, socket.SOCK_STREAM) as candidate:
        candidate.bind((host, 0))
        return int(candidate.getsockname()[1])


class _StateTelemetryInstrumentation:
    """Adapter from the Phase 1 device state to profiling context metrics."""

    def __init__(
        self, device_id: str, record: ProfilingSessionRecord
    ) -> None:
        self._device_id = device_id
        self._record = record

    def capture(self) -> DeviceObservation | None:
        if self._record.worker_state_source is None:
            return None
        state = self._record.worker_state_source()
        if state is None:
            return None
        device_state = next(
            (item for item in state.device_states if item.device_id == self._device_id),
            None,
        )
        if device_state is None:
            return None
        memory_used: int | None = None
        capability = self._record.capability
        if capability is not None:
            device = next(
                (
                    item
                    for item in capability.devices
                    if item.identity.device_id == self._device_id
                ),
                None,
            )
            pool = next(
                (
                    item
                    for item in capability.memory_pools
                    if device is not None
                    and item.memory_pool_id == device.memory_pool_id
                ),
                None,
            )
            pool_state = next(
                (
                    item
                    for item in state.memory_states
                    if pool is not None
                    and item.memory_pool_id == pool.memory_pool_id
                ),
                None,
            )
            if (
                pool is not None
                and pool_state is not None
                and pool_state.available_bytes is not None
            ):
                memory_used = pool.total_bytes - pool_state.available_bytes
        return DeviceObservation(
            device_id=self._device_id,
            utilization=device_state.utilization,
            temperature_c=device_state.temperature_c,
            power_w=device_state.power_w,
            memory_used_bytes=memory_used,
        )


def default_instrumentation(
    device: torch.device,
    *,
    device_id: str | None = None,
    record: ProfilingSessionRecord | None = None,
) -> InstrumentationBundle:
    """The v1 bundle: device-correct timing, allocator memory on CUDA (§11-14).

    Physical-memory and telemetry instruments reuse the Worker inspector's
    live Phase 1 backend through ``worker_state_source``. Each open/poll/close
    obtains a fresh sample; Jetson keeps the same long-lived tegrastats reader.
    """
    timer = CudaEventTimer(device.index) if device.type == "cuda" else WallClockTimer()
    allocator = CudaAllocatorMemoryProbe(device.index) if device.type == "cuda" else None
    physical: PhysicalMemoryProbe | None = None
    telemetry: TelemetryContextCollector | None = None
    if (
        record is not None
        and device_id is not None
        and record.worker_state_source is not None
    ):
        capability = record.capability
        device_capability = (
            next(
                (
                    item
                    for item in capability.devices
                    if item.identity.device_id == device_id
                ),
                None,
            )
            if capability is not None
            else None
        )
        pool = (
            next(
                (
                    item
                    for item in capability.memory_pools
                    if item.memory_pool_id == device_capability.memory_pool_id
                ),
                None,
            )
            if capability is not None and device_capability is not None
            else None
        )
        if pool is not None:
            def read_available() -> int | None:
                assert record.worker_state_source is not None
                current_state = record.worker_state_source()
                if current_state is None:
                    return None
                current = next(
                    (
                        item
                        for item in current_state.memory_states
                        if item.memory_pool_id == pool.memory_pool_id
                    ),
                    None,
                )
                return current.available_bytes if current is not None else None

            physical = PhysicalMemoryProbe(
                pool.memory_pool_id,
                total_bytes=pool.total_bytes,
                read_available_bytes=read_available,
            )
        telemetry = TelemetryContextCollector(
            _StateTelemetryInstrumentation(device_id, record)
        )
    return InstrumentationBundle(
        timer=timer,
        memory=allocator,
        physical_memory=physical,
        telemetry=telemetry,
    )


def default_model_loader(device: torch.device) -> ModelSessionLoader:
    return TorchModelSessionLoader(device=device)


class WorkerProfilingRunner:
    """``WorkerProfilingHandler`` implementation of the Worker plane (§37).

    Everything expensive or platform-specific is injectable — profilers,
    loader factory, device resolver, instrumentation factory, clock, seed —
    so unit tests drive the full RPC lifecycle on a CPU-only host with stub
    checkpoints; the defaults are the real measurement stack.
    """

    def __init__(
        self,
        *,
        sessions: ProfilingSessionManager,
        inspector: StateInspector,
        model_store_root: Path,
        leases: DeviceLeaseManager | None = None,
        model_loader_factory: ModelLoaderFactory | None = None,
        layer_profiler: TransformerLayerProfiler | None = None,
        module_profiler: ModuleProfiler | None = None,
        operator_profiler: OperatorProfiler | None = None,
        network_profiler: NetworkProfiler | None = None,
        device_resolver: DeviceResolver | None = None,
        instrumentation_factory: InstrumentationFactory | None = None,
        clock: Callable[[], datetime] = utc_now,
        seed: int | None = None,
    ) -> None:
        self._sessions = sessions
        self._inspector = inspector
        self._model_store_root = model_store_root
        self._leases = leases if leases is not None else DeviceLeaseManager()
        self._model_loader_factory = (
            model_loader_factory if model_loader_factory is not None else default_model_loader
        )
        self._layer_profiler = (
            layer_profiler if layer_profiler is not None else TransformerLayerProfiler()
        )
        self._module_profiler = (
            module_profiler if module_profiler is not None else ModuleProfiler()
        )
        self._operator_profiler = (
            operator_profiler if operator_profiler is not None else OperatorProfiler()
        )
        self._network_profiler = (
            network_profiler if network_profiler is not None else NetworkProfiler()
        )
        self._device_resolver = (
            device_resolver if device_resolver is not None else resolve_torch_device
        )
        self._instrumentation_factory = instrumentation_factory
        self._clock = clock
        self._seed = seed
        self._iperf_servers: dict[
            str, tuple[asyncio.subprocess.Process, asyncio.Task[None], int]
        ] = {}

    @property
    def leases(self) -> DeviceLeaseManager:
        return self._leases

    @property
    def sessions(self) -> ProfilingSessionManager:
        return self._sessions

    # ------------------------------------------------------------------
    # PrepareProfilingSession (§38)
    # ------------------------------------------------------------------

    async def prepare_profiling_session(
        self, request: PrepareProfilingSessionRequest
    ) -> PrepareProfilingSessionResponse:
        try:
            tokens = self._validate(request)
        except SessionRefused as exc:
            return PrepareProfilingSessionResponse(
                accepted=False, detail=exc.detail, reason=exc.reason
            )

        session_id = request.profiling_session_id
        existing = self._sessions.peek(session_id)
        if existing is not None:
            # A retried prepare (e.g. after a lost response) replays the
            # prepared session instead of loading a second checkpoint (§37).
            if existing.closed:
                return _prepare_refused(
                    ProfilingRejection.SESSION_CLOSED,
                    f"profiling session {session_id!r} is closed; prepare a new id",
                )
            if existing.kind is not request.session_request.kind:
                return _prepare_refused(
                    ProfilingRejection.SESSION_KIND_MISMATCH,
                    f"session {session_id!r} was prepared as "
                    f"{existing.kind.value!r}, not "
                    f"{request.session_request.kind.value!r}",
                )
            return PrepareProfilingSessionResponse(
                accepted=True, session_facts=existing.session_facts
            )

        session_request = request.session_request
        inspection = await self._inspector.inspect()
        network_facts: Mapping[str, WorkerNetworkFacts] = {}
        if session_request.kind is ProfilingSessionKind.NETWORK:
            network_facts = {facts.worker_id: facts for facts in request.network_facts}
            if tokens.worker_id not in network_facts:
                # §47: the Master resolves the executing worker's own probe
                # addresses; completing them locally would guess facts the
                # protocol says only the Master may supply (§52.2).
                raise ValueError(
                    "network session facts must include the executing worker "
                    f"{tokens.worker_id!r}"
                )
        else:
            try:
                self._leases.acquire(
                    session_id, session_request.device_ids, inspection.state
                )
            except DeviceBusyError as exc:
                return _prepare_refused(ProfilingRejection.DEVICE_BUSY, str(exc))

        model_handle: LoadedModelSession | None = None
        try:
            if session_request.kind is ProfilingSessionKind.MODEL:
                model_handle = await self._load_model_session(
                    session_id, request, inspection, tokens
                )
        except Exception as exc:
            # Preparation failed: release the reservation (§39) and answer
            # with the typed failure — the category survives the wire (§42).
            self._leases.release_session(session_id)
            failure = (
                exc.to_failure()
                if isinstance(exc, ProfilingError)
                else ProfilingFailure(
                    category=ProfilingErrorCategory.INTERNAL_ERROR,
                    message=f"session preparation failed: {exc}",
                )
            )
            if not isinstance(exc, ProfilingError):
                logger.exception("preparation of session %s failed", session_id)
            return PrepareProfilingSessionResponse(
                accepted=False, detail=str(exc), failure=failure
            )

        record = self._sessions.create_session(
            session_id,
            session_request,
            now=self._clock(),
            capability_revision=inspection.capability.capability_revision or None,
            capability=inspection.capability,
            worker_state=inspection.state,
            worker_state_source=getattr(
                self._inspector, "sample_fresh_state", None
            ),
            session_facts=model_handle.facts if model_handle is not None else None,
            network_facts=network_facts,
            model_handle=model_handle,
            model_cleanup=model_handle.close if model_handle is not None else None,
        )
        if model_handle is not None:
            characterization = model_handle.facts.characterization
            record.session_facts = dataclasses.replace(
                model_handle.facts,
                environment=self._fingerprint(
                    record,
                    tokens,
                    backend=session_request.backend,
                    device_id=session_request.device_ids[0],
                    dtype=session_request.dtype,
                    quantization=characterization.quantization,
                    model_revision=characterization.model.revision,
                ),
            )
        return PrepareProfilingSessionResponse(
            accepted=True, session_facts=record.session_facts
        )

    async def _load_model_session(
        self,
        session_id: str,
        request: PrepareProfilingSessionRequest,
        inspection: LocalInspection,
        tokens: RegistrationTokens,
    ) -> LoadedModelSession:
        """Resolve → load → enumerate one MODEL session (§38, load once)."""
        session_request = request.session_request
        assert session_request.model is not None  # domain guarantees (§38)
        device = self._device_resolver(
            session_request.device_ids[0], tokens.worker_id
        )  # v1 loads the checkpoint onto the session's first leased device
        source = resolve_model_source(
            session_request.model.model_id,
            session_request.model.revision,
            inspection.state.models,
            ModelStore(model_root=self._model_store_root),
        )
        loader = self._model_loader_factory(device)
        logger.info(
            "loading model %s for session %s on %s",
            session_request.model.model_id,
            session_id,
            device,
        )
        return await asyncio.to_thread(loader.load, session_request, source)

    # ------------------------------------------------------------------
    # RunProfilingCase (§39, §42, §44)
    # ------------------------------------------------------------------

    async def run_profiling_case(
        self, request: RunProfilingCaseRequest
    ) -> RunProfilingCaseResponse:
        try:
            tokens = self._validate(request)
            record = self._sessions.require_open_session(request.profiling_session_id)
        except SessionRefused as exc:
            return RunProfilingCaseResponse(
                accepted=False, detail=exc.detail, reason=exc.reason
            )

        case = request.case
        if case.worker_id != tokens.worker_id:
            # §8.2: a case assigned elsewhere must never execute here; this
            # is a Master-side protocol violation, not a semantic refusal.
            raise ValueError(
                f"case {case.case_id!r} is assigned to worker "
                f"{case.worker_id!r}, not this worker ({tokens.worker_id!r})"
            )
        mismatch = _kind_mismatch(record, case)
        if mismatch is not None:
            return RunProfilingCaseResponse(
                accepted=False, detail=mismatch, reason=ProfilingRejection.SESSION_KIND_MISMATCH
            )

        entry = record.cases.get(case.case_id)
        if entry is not None:
            if entry.terminal:
                assert entry.outcome is not None  # §44 ledger invariant
                # Duplicate dispatch (retry after a lost response, §50):
                # replay the recorded decision, never re-benchmark.
                return RunProfilingCaseResponse(accepted=True, outcome=entry.outcome)
            return RunProfilingCaseResponse(
                accepted=False,
                detail=f"case {case.case_id!r} is already running in this session",
                reason=ProfilingRejection.DEVICE_BUSY,
            )

        if record.kind is not ProfilingSessionKind.NETWORK:
            assert isinstance(case.spec, ModelCaseSpec)  # _kind_mismatch guarantees
            leased = {
                lease.device_id
                for lease in self._leases.leases_for_session(record.session_id)
            }
            missing = [
                device_id for device_id in case.spec.device_ids if device_id not in leased
            ]
            if missing:
                # §39: the reservation is the run's authority; a case whose
                # devices the session does not hold is refused, not measured.
                return RunProfilingCaseResponse(
                    accepted=False,
                    detail=(
                        f"session {record.session_id!r} holds no lease on "
                        f"device(s): {', '.join(missing)}"
                    ),
                    reason=ProfilingRejection.DEVICE_BUSY,
                )

        self._sessions.mark_running(record.session_id, case.case_id)
        try:
            outcome = await self._execute(
                record,
                case,
                tokens,
                iperf_server_port=request.iperf_server_port,
            )
        except ProfilingError as exc:
            outcome = CaseOutcome.from_failure(exc.to_failure())
        except Exception as exc:
            logger.exception("case %s failed unexpectedly", case.case_id)
            outcome = CaseOutcome.from_failure(
                ProfilingFailure(
                    category=ProfilingErrorCategory.INTERNAL_ERROR,
                    message=f"unexpected runner failure: {exc}",
                )
            )
        state = CaseState.COMPLETED if outcome.succeeded else CaseState.FAILED
        try:
            self._sessions.decide_case(
                record.session_id, case.case_id, state, outcome, now=self._clock()
            )
        except ValueError:
            # A concurrent CancelProfilingCase decided the case mid-run
            # (§41): the cancellation stands and the finished measurement is
            # discarded — a cancelled case never publishes results.
            cancelled = self._sessions.case_entry(record.session_id, case.case_id)
            if cancelled is None or cancelled.outcome is None:
                raise
            logger.warning(
                "case %s was cancelled mid-run; discarding its measurement",
                case.case_id,
            )
            outcome = cancelled.outcome
        return RunProfilingCaseResponse(accepted=True, outcome=outcome)

    async def _execute(
        self,
        record: ProfilingSessionRecord,
        case: ProfilingCase,
        tokens: RegistrationTokens,
        *,
        iperf_server_port: int | None = None,
    ) -> CaseOutcome:
        spec = case.spec
        if isinstance(spec, NetworkCaseSpec):
            return await self._execute_network(
                record,
                case,
                spec,
                tokens,
                iperf_server_port=iperf_server_port,
            )
        assert isinstance(spec, ModelCaseSpec)  # _kind_mismatch guarantees
        if spec.granularity is ProfilingGranularity.OPERATOR:
            return await self._execute_operator(record, case, spec, tokens)
        return await self._execute_model(record, case, spec, tokens)

    async def _execute_network(
        self,
        record: ProfilingSessionRecord,
        case: ProfilingCase,
        spec: NetworkCaseSpec,
        tokens: RegistrationTokens,
        *,
        iperf_server_port: int | None,
    ) -> CaseOutcome:
        if spec.destination_worker_id not in record.network_facts:
            raise ProfilingError(
                ProfilingErrorCategory.NETWORK_UNREACHABLE,
                f"no Master-resolved network facts for destination worker "
                f"{spec.destination_worker_id!r} in session {record.session_id!r}",
                {"destination_worker_id": spec.destination_worker_id},
            )
        backend = "iperf3" if spec.probe_kind is ProbeKind.BANDWIDTH else "ping"
        fingerprint = self._fingerprint(record, tokens, backend=backend)
        result = await self._network_profiler.profile(
            case,
            network_facts=record.network_facts,
            environment_fingerprint=environment_fingerprint_id(fingerprint),
            iperf_server_port=iperf_server_port,
        )
        return CaseOutcome.from_record(
            dataclasses.replace(
                result,
                environment=fingerprint,
                environment_fingerprint=environment_fingerprint_id(fingerprint),
            )
        )

    async def _execute_operator(
        self,
        record: ProfilingSessionRecord,
        case: ProfilingCase,
        spec: ModelCaseSpec,
        tokens: RegistrationTokens,
    ) -> CaseOutcome:
        device_id = spec.device_ids[0]  # v1 operator benchmarks are single-device
        device = self._device_resolver(device_id, tokens.worker_id)
        instrumentation = self._instrumentation(record, device_id, device)
        fingerprint = self._fingerprint(
            record, tokens, backend=spec.backend, device_id=device_id, dtype=spec.dtype
        )
        result = await asyncio.to_thread(
            self._operator_profiler.profile,
            case,
            instrumentation=instrumentation,
            environment_fingerprint=environment_fingerprint_id(fingerprint),
            device=device,
        )
        return CaseOutcome.from_record(
            dataclasses.replace(
                result,
                environment=fingerprint,
                environment_fingerprint=environment_fingerprint_id(fingerprint),
            )
        )

    async def _execute_model(
        self,
        record: ProfilingSessionRecord,
        case: ProfilingCase,
        spec: ModelCaseSpec,
        tokens: RegistrationTokens,
    ) -> CaseOutcome:
        handle = record.model_handle
        if not isinstance(handle, LoadedModelSession):
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                f"model session {record.session_id!r} carries no loaded checkpoint",
            )
        facts = record.session_facts
        characterization = handle.facts.characterization
        if spec.granularity is ProfilingGranularity.TRANSFORMER_LAYER:
            if spec.layer_index is None:
                raise ProfilingError(
                    ProfilingErrorCategory.UNSUPPORTED_GRANULARITY,
                    "transformer-layer cases require the layer_index they "
                    "measure; the runner never defaults to a layer (§52.2)",
                    {"case_id": case.case_id},
                )
            layer = handle.layer_at(spec.layer_index)
            if facts is not None:
                declared = next(
                    (
                        entry.signature
                        for entry in facts.layer_entries
                        if entry.index == layer.index
                    ),
                    None,
                )
                if declared is not None and spec.layer_signature != declared:
                    # The case contradicts the facts this session published:
                    # measuring anyway would publish a mislabeled identity
                    # (§41), so it fails loudly instead.
                    raise ValueError(
                        f"case {case.case_id!r} declares a layer signature that "
                        f"contradicts session {record.session_id!r} facts at "
                        f"layer_index {layer.index}"
                    )
            device = module_device(layer.layer)
            self._require_case_device(
                device, spec.device_ids[0], tokens.worker_id, case.case_id
            )
            instrumentation = self._instrumentation(
                record, spec.device_ids[0], device
            )
            fingerprint = self._model_fingerprint(record, tokens, spec, characterization)
            result = await asyncio.to_thread(
                self._layer_profiler.profile,
                case,
                handle.model,
                layer,
                handle.layout,
                handle.adapter,
                instrumentation=instrumentation,
                environment_fingerprint=environment_fingerprint_id(fingerprint),
                seed=self._seed,
            )
            return CaseOutcome.from_record(
                dataclasses.replace(
                    result,
                    environment=fingerprint,
                    environment_fingerprint=environment_fingerprint_id(fingerprint),
                )
            )

        assert spec.module_signature is not None  # domain guarantees for MODULE
        module = handle.module_for(spec.module_signature)
        device = module_device(module.module)
        self._require_case_device(
            device, spec.device_ids[0], tokens.worker_id, case.case_id
        )
        instrumentation = self._instrumentation(record, spec.device_ids[0], device)
        fingerprint = self._model_fingerprint(record, tokens, spec, characterization)
        result = await asyncio.to_thread(
            self._module_profiler.profile,
            case,
            handle.model,
            module,
            handle.layout,
            handle.adapter,
            instrumentation=instrumentation,
            environment_fingerprint=environment_fingerprint_id(fingerprint),
            seed=self._seed,
        )
        return CaseOutcome.from_record(
            dataclasses.replace(
                result,
                environment=fingerprint,
                environment_fingerprint=environment_fingerprint_id(fingerprint),
            )
        )

    def _require_case_device(
        self,
        actual: torch.device,
        device_id: str,
        worker_id: str,
        case_id: str,
    ) -> None:
        expected = self._device_resolver(device_id, worker_id)
        if actual != expected:
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                f"case {case_id!r} targets {device_id!r} ({expected}) but "
                f"the loaded module is on {actual}",
                {
                    "case_id": case_id,
                    "device_id": device_id,
                    "expected_torch_device": str(expected),
                    "actual_torch_device": str(actual),
                },
            )

    def _instrumentation(
        self,
        record: ProfilingSessionRecord,
        device_id: str,
        device: torch.device,
    ) -> InstrumentationBundle:
        if self._instrumentation_factory is not None:
            return self._instrumentation_factory(device)
        return default_instrumentation(device, device_id=device_id, record=record)

    # ------------------------------------------------------------------
    # Get / Cancel / Close (§41, §44)
    # ------------------------------------------------------------------

    async def get_profiling_case(
        self, request: GetProfilingCaseRequest
    ) -> GetProfilingCaseResponse:
        try:
            self._validate(request)
            record = self._sessions.require_open_session(request.profiling_session_id)
        except SessionRefused as exc:
            return GetProfilingCaseResponse(
                accepted=False, detail=exc.detail, reason=exc.reason
            )
        entry = record.cases.get(request.case_id)
        if entry is None:
            return GetProfilingCaseResponse(
                accepted=False,
                detail=(
                    f"session {record.session_id!r} never dispatched case "
                    f"{request.case_id!r}"
                ),
                reason=ProfilingRejection.UNKNOWN_CASE,
            )
        return GetProfilingCaseResponse(
            accepted=True, case_state=entry.state, outcome=entry.outcome
        )

    async def cancel_profiling_case(
        self, request: CancelProfilingCaseRequest
    ) -> CancelProfilingCaseResponse:
        try:
            self._validate(request)
            record = self._sessions.require_open_session(request.profiling_session_id)
        except SessionRefused as exc:
            return CancelProfilingCaseResponse(
                accepted=False, detail=exc.detail, reason=exc.reason
            )
        entry = self._sessions.cancel_case(
            record.session_id, request.case_id, now=self._clock()
        )
        return CancelProfilingCaseResponse(accepted=True, case_state=entry.state)

    async def prepare_iperf_server(
        self, request: PrepareIperfServerRequest
    ) -> PrepareIperfServerResponse:
        try:
            self._validate(request)
        except SessionRefused as exc:
            return PrepareIperfServerResponse(
                accepted=False, detail=exc.detail, reason=exc.reason
            )
        existing = self._iperf_servers.get(request.server_id)
        if existing is not None:
            return PrepareIperfServerResponse(accepted=True, port=existing[2])
        port = request.port or _available_tcp_port(request.bind_address)
        try:
            process = await self._network_profiler.start_iperf_server(
                port=port, bind_address=request.bind_address
            )
        except Exception as exc:
            return PrepareIperfServerResponse(
                accepted=False,
                detail=f"could not start temporary iperf3 server: {exc}",
                reason=ProfilingRejection.DEVICE_BUSY,
            )
        timeout_task = asyncio.create_task(
            self._expire_iperf_server(request.server_id, request.timeout_s),
            name=f"iperf-server-timeout-{request.server_id[:12]}",
        )
        self._iperf_servers[request.server_id] = (process, timeout_task, port)
        return PrepareIperfServerResponse(accepted=True, port=port)

    async def stop_iperf_server(
        self, request: StopIperfServerRequest
    ) -> StopIperfServerResponse:
        try:
            self._validate(request)
        except SessionRefused as exc:
            return StopIperfServerResponse(
                accepted=False, detail=exc.detail, reason=exc.reason
            )
        await self._stop_iperf_server(request.server_id)
        return StopIperfServerResponse(accepted=True)

    async def _expire_iperf_server(self, server_id: str, timeout_s: float) -> None:
        try:
            await asyncio.sleep(timeout_s)
            await self._stop_iperf_server(server_id, cancel_timeout=False)
        except asyncio.CancelledError:
            pass

    async def _stop_iperf_server(
        self, server_id: str, *, cancel_timeout: bool = True
    ) -> None:
        handle = self._iperf_servers.pop(server_id, None)
        if handle is None:
            return
        process, timeout_task, _port = handle
        if cancel_timeout:
            timeout_task.cancel()
        await self._network_profiler.stop_iperf_server(process)

    async def close_profiling_session(
        self, request: CloseProfilingSessionRequest
    ) -> CloseProfilingSessionResponse:
        try:
            self._validate(request)
        except SessionRefused as exc:
            return CloseProfilingSessionResponse(
                accepted=False, detail=exc.detail, reason=exc.reason
            )
        # Idempotent close (§38): unknown/already-closed sessions still
        # answer accepted, and the lease release is unconditional so a close
        # can never strand a reservation (§39).
        self._sessions.close_session(request.profiling_session_id)
        self._leases.release_session(request.profiling_session_id)
        return CloseProfilingSessionResponse(accepted=True)

    async def shutdown(self) -> None:
        """Release every session and lease (runner shutdown, §39).

        Leases MUST NOT outlive the runner: a leaked reservation would
        permanently shrink the profileable device set of the next serve.
        """
        for session_id in self._sessions.close_all():
            self._leases.release_session(session_id)
        for server_id in tuple(self._iperf_servers):
            await self._stop_iperf_server(server_id)
        self._leases.release_all()
        logger.info("profiling runner shut down")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate(
        self,
        request: (
            PrepareProfilingSessionRequest
            | RunProfilingCaseRequest
            | GetProfilingCaseRequest
            | CancelProfilingCaseRequest
            | CloseProfilingSessionRequest
            | PrepareIperfServerRequest
            | StopIperfServerRequest
        ),
    ) -> RegistrationTokens:
        """§41: every RPC rides the *current* registration or is refused."""
        return self._sessions.validate_tokens(
            worker_id=request.worker_id,
            instance_id=request.instance_id,
            registration_session_id=request.registration_session_id,
        )

    def _model_fingerprint(
        self,
        record: ProfilingSessionRecord,
        tokens: RegistrationTokens,
        spec: ModelCaseSpec,
        characterization: ModelCharacterization,
    ) -> EnvironmentFingerprint:
        return self._fingerprint(
            record,
            tokens,
            backend=spec.backend,
            device_id=spec.device_ids[0],
            dtype=spec.dtype,
            quantization=characterization.quantization,
            model_revision=characterization.model.revision,
        )

    def _fingerprint(
        self,
        record: ProfilingSessionRecord,
        tokens: RegistrationTokens,
        *,
        backend: str,
        device_id: str | None = None,
        dtype: str | None = None,
        quantization: str | None = None,
        model_revision: str | None = None,
    ) -> EnvironmentFingerprint:
        """The §9 compatibility identity every record of this session carries.

        Volatile telemetry is not part of it (§9); facts this Worker cannot
        observe (driver version, a verified performance class) stay absent
        rather than approximated (§52.2).
        """
        performance_class = self._device_performance_class(
            record, device_id=device_id, backend=backend, dtype=dtype
        )
        fingerprint = EnvironmentFingerprint(
            backend=backend,
            profiling_implementation_revision=PROFILING_IMPLEMENTATION_REVISION,
            device_performance_class_id=(
                device_performance_class_id(performance_class)
                if performance_class is not None
                else None
            ),
            device_performance_class=performance_class,
            capability_revision=record.capability_revision,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            driver_version=self._driver_version(record, device_id),
            model_revision=model_revision,
            dtype=dtype,
            quantization=quantization,
            worker_id=tokens.worker_id,
            device_id=device_id,
        )
        return fingerprint

    @staticmethod
    def _driver_version(
        record: ProfilingSessionRecord, device_id: str | None
    ) -> str | None:
        if record.capability is None or device_id is None:
            return None
        device = next(
            (
                candidate
                for candidate in record.capability.devices
                if candidate.identity.device_id == device_id
            ),
            None,
        )
        return device.driver_version if device is not None else None

    @staticmethod
    def _device_performance_class(
        record: ProfilingSessionRecord,
        *,
        device_id: str | None,
        backend: str,
        dtype: str | None,
    ) -> DevicePerformanceClass | None:
        """Build the candidate reuse class from device-relevant facts only.

        Membership in this structural class is not itself permission for
        cross-device reuse; the store's verified-membership gate owns that
        decision. Host-wide capability revisions and NIC/container changes
        deliberately do not participate.
        """
        capability = record.capability
        if capability is None or device_id is None:
            return None
        device = next(
            (
                candidate
                for candidate in capability.devices
                if candidate.identity.device_id == device_id
            ),
            None,
        )
        if device is None:
            return None
        pool = next(
            (
                candidate
                for candidate in capability.memory_pools
                if candidate.memory_pool_id == device.memory_pool_id
            ),
            None,
        )
        memory_model = (
            MemoryModel(pool.model.value) if pool is not None else MemoryModel.DISCRETE
        )
        versions = {"torch": torch.__version__.split("+")[0]}
        if torch.version.cuda is not None:
            versions["cuda"] = torch.version.cuda
        return DevicePerformanceClass(
            vendor=device.vendor,
            accelerator_model=device.model,
            memory_model=memory_model,
            backend_family=backend,
            architecture=device.compute_capability,
            dtype=dtype,
            software_versions=tuple(sorted(versions.items())),
        )


def _kind_mismatch(record: ProfilingSessionRecord, case: ProfilingCase) -> str | None:
    """Detail string when a case does not belong to this session kind (§41)."""
    spec = case.spec
    if record.kind is ProfilingSessionKind.NETWORK:
        if not isinstance(spec, NetworkCaseSpec):
            return (
                f"network session {record.session_id!r} cannot run "
                f"{type(spec).__name__} cases"
            )
        return None
    if not isinstance(spec, ModelCaseSpec):
        return (
            f"{record.kind.value} session {record.session_id!r} cannot run "
            "network cases"
        )
    if record.kind is ProfilingSessionKind.OPERATOR:
        if spec.granularity is not ProfilingGranularity.OPERATOR:
            return (
                f"operator session {record.session_id!r} cannot run "
                f"{spec.granularity.value} cases"
            )
        return None
    if spec.granularity is ProfilingGranularity.OPERATOR:
        return (
            f"model session {record.session_id!r} does not run model-free "
            "operator cases (§38: they belong to operator sessions)"
        )
    return None


def _prepare_refused(
    reason: ProfilingRejection, detail: str
) -> PrepareProfilingSessionResponse:
    return PrepareProfilingSessionResponse(accepted=False, detail=detail, reason=reason)


__all__ = [
    "PROFILING_IMPLEMENTATION_REVISION",
    "DeviceResolver",
    "InstrumentationFactory",
    "ModelLoaderFactory",
    "StateInspector",
    "WorkerProfilingRunner",
    "default_instrumentation",
    "default_model_loader",
    "resolve_torch_device",
]
