"""Control-plane domain ↔ wire mapping (Phase 1 spec §40).

This module is the only boundary that translates between the cluster domain
dataclasses and the control protobuf DTOs; generated protobuf objects never
reach WorkerRegistry, StateStore, SnapshotBuilder, or any future scheduler
(spec §40, §59).

Fail-loudly policy (spec §47): semantic protocol violations — wrong protocol
version, identity/revision mismatches between redundant wire fields, unknown
enum values — raise :class:`ControlProtocolError` at the boundary instead of
producing silently degraded domain objects. Missing telemetry is never an
error: proto3 ``optional`` fields distinguish absence (``None``) from zero,
exactly as the domain model requires (spec §17).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from edgeshard.cluster.capability import (
    ContainerRuntimeCapability,
    DeviceCapability,
    MemoryModel,
    MemoryPoolCapability,
    NetworkInterfaceCapability,
    OSInfo,
    RuntimePlatformCapability,
    WorkerCapability,
)
from edgeshard.cluster.identity import DeviceIdentity, DeviceKind, WorkerIdentity
from edgeshard.cluster.inventory import (
    ModelAvailability,
    ModelInventoryEntry,
    RuntimeInstanceState,
    RuntimeStatus,
)
from edgeshard.cluster.state import (
    DeviceAvailability,
    DeviceState,
    MemoryPoolState,
    WorkerState,
)
from edgeshard.model.errors import EdgeShardError
from edgeshard.protocol.control.pb import worker_control_pb2 as pb

CONTROL_PROTOCOL_VERSION = "1"
"""Control-plane protocol revision carried in every registration (spec §29)."""


class ControlProtocolError(EdgeShardError):
    """Control protocol violation: version, identity, revision, or wire shape."""


class RejectionReason(StrEnum):
    """Why the Master rejected a heartbeat or capability update (spec §30, §35).

    The Worker Agent's recovery keys off this enum, never off the detail
    string: ``UNKNOWN_WORKER``/``REREGISTER_REQUIRED`` mean "register again
    with a fresh inspection"; ``STALE_SESSION`` means a newer registration
    superseded this Agent, so it must stop instead of fighting for the
    session (two same-worker_id Agents must not ping-pong);
    ``INSTANCE_MISMATCH``/``OUT_OF_ORDER`` are Agent-side protocol
    violations and must fail loudly (§47).
    """

    UNKNOWN_WORKER = "unknown_worker"
    STALE_SESSION = "stale_session"
    INSTANCE_MISMATCH = "instance_mismatch"
    OUT_OF_ORDER = "out_of_order"
    REREGISTER_REQUIRED = "reregister_required"


# ---------------------------------------------------------------------------
# Domain request/response objects (spec §29-30)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegisterWorkerRequest:
    """One Worker registration (spec §29).

    ``worker_id`` and ``capability_revision`` travel both top-level and
    inside ``identity``/``capability`` on the wire; the mapper cross-checks
    the redundant copies and rejects any mismatch (spec §47).

    ``profiling_endpoint`` is the additive Phase 2 field (Phase 2 spec §41):
    the "host:port" of the Worker-hosted ``WorkerProfilingService``, or
    ``None`` when the Worker does not host profiling. It lives outside
    ``WorkerCapability`` so frozen capability revisions never change.
    """

    protocol_version: str
    instance_id: str
    identity: WorkerIdentity
    capability: WorkerCapability
    initial_state: WorkerState
    profiling_endpoint: str | None = None

    def __post_init__(self) -> None:
        if self.protocol_version != CONTROL_PROTOCOL_VERSION:
            raise ControlProtocolError(
                f"unsupported control protocol version "
                f"{self.protocol_version!r} (expected {CONTROL_PROTOCOL_VERSION!r})"
            )
        if self.identity.protocol_version != self.protocol_version:
            # The redundant copies must agree (spec §47): an identity built
            # for another protocol version never rides a v1 registration.
            raise ControlProtocolError(
                f"identity protocol_version {self.identity.protocol_version!r} "
                f"does not match registration protocol_version "
                f"{self.protocol_version!r}"
            )
        if not self.instance_id:
            raise ValueError("instance_id must not be empty")
        if self.identity.worker_id != self.initial_state.worker_id:
            raise ValueError(
                f"identity/state worker_id mismatch: "
                f"{self.identity.worker_id!r} vs {self.initial_state.worker_id!r}"
            )
        if self.profiling_endpoint is not None and not self.profiling_endpoint:
            raise ValueError("profiling_endpoint must not be empty when present")


@dataclass(frozen=True)
class RegisterWorkerResponse:
    """The Master's registration answer: session and cadence (spec §29)."""

    session_id: str
    heartbeat_interval_ms: int
    server_protocol_version: str

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.heartbeat_interval_ms <= 0:
            raise ValueError(
                f"heartbeat_interval_ms must be positive, got {self.heartbeat_interval_ms}"
            )
        if self.server_protocol_version != CONTROL_PROTOCOL_VERSION:
            raise ControlProtocolError(
                f"invalid registration response: server protocol version "
                f"{self.server_protocol_version!r} "
                f"(expected {CONTROL_PROTOCOL_VERSION!r})"
            )


@dataclass(frozen=True)
class HeartbeatRequest:
    """One dynamic-state report (spec §30).

    Sequence numbers start at 1 after registration and increase
    monotonically; ``worker_reported_at`` is a debugging aid only and must
    never feed Master liveness (spec §31).
    """

    worker_id: str
    instance_id: str
    session_id: str
    sequence_number: int
    capability_revision: str
    state: WorkerState
    worker_reported_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id must not be empty")
        if not self.instance_id:
            raise ValueError("instance_id must not be empty")
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.sequence_number < 1:
            raise ValueError(
                f"sequence_number must start at 1 and increase, got {self.sequence_number}"
            )
        if not self.capability_revision:
            raise ValueError("capability_revision must not be empty")
        if self.worker_id != self.state.worker_id:
            raise ValueError(
                f"heartbeat/state worker_id mismatch: "
                f"{self.worker_id!r} vs {self.state.worker_id!r}"
            )
        if self.worker_reported_at is not None and (
            self.worker_reported_at.tzinfo is None
            or self.worker_reported_at.tzinfo.utcoffset(self.worker_reported_at) is None
        ):
            raise ValueError("worker_reported_at must be timezone-aware")


@dataclass(frozen=True)
class HeartbeatResponse:
    """Acceptance verdict for one heartbeat (spec §30: ignore/reject).

    A rejection carries both a human-readable ``detail`` (§46) and the
    machine-readable ``reason`` the Worker Agent's recovery keys off.
    """

    accepted: bool
    detail: str = ""
    reason: RejectionReason | None = None

    def __post_init__(self) -> None:
        if not self.accepted:
            if not self.detail:
                raise ValueError("a rejected heartbeat must carry a detail")
            if self.reason is None:
                raise ValueError("a rejected heartbeat must carry a reason")
        elif self.reason is not None:
            raise ValueError("an accepted heartbeat must not carry a rejection reason")


@dataclass(frozen=True)
class UpdateCapabilityRequest:
    """A full capability retransmission after a revision change (spec §16).

    Carries the state sampled atomically with the new capability so the
    Master can apply both in one step: a snapshot must never pair a new
    capability with an old state that references devices/pools it no
    longer contains (§38 cross-validation).
    """

    worker_id: str
    instance_id: str
    session_id: str
    capability: WorkerCapability
    state: WorkerState

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id must not be empty")
        if not self.instance_id:
            raise ValueError("instance_id must not be empty")
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.worker_id != self.state.worker_id:
            raise ValueError(
                f"capability update/state worker_id mismatch: "
                f"{self.worker_id!r} vs {self.state.worker_id!r}"
            )


@dataclass(frozen=True)
class UpdateCapabilityResponse:
    """Acceptance verdict for a capability update."""

    accepted: bool
    detail: str = ""
    reason: RejectionReason | None = None

    def __post_init__(self) -> None:
        if not self.accepted:
            if not self.detail:
                raise ValueError("a rejected capability update must carry a detail")
            if self.reason is None:
                raise ValueError("a rejected capability update must carry a reason")
        elif self.reason is not None:
            raise ValueError("an accepted capability update must not carry a reason")


# ---------------------------------------------------------------------------
# Enum translation tables
# ---------------------------------------------------------------------------

# The generated .pyi types wire enums as pb enum classes (subclasses of int),
# so the tables are typed against them directly; at runtime both sides are
# plain ints and lookups behave accordingly.

_DEVICE_KIND_TO_WIRE: dict[DeviceKind, pb.DeviceKind] = {
    DeviceKind.CPU: pb.DEVICE_KIND_CPU,
    DeviceKind.GPU: pb.DEVICE_KIND_GPU,
    DeviceKind.NPU: pb.DEVICE_KIND_NPU,
    DeviceKind.OTHER: pb.DEVICE_KIND_OTHER,
}
_WIRE_TO_DEVICE_KIND: dict[pb.DeviceKind, DeviceKind] = {
    v: k for k, v in _DEVICE_KIND_TO_WIRE.items()
}

_MEMORY_MODEL_TO_WIRE: dict[MemoryModel, pb.MemoryModel] = {
    MemoryModel.DISCRETE: pb.MEMORY_MODEL_DISCRETE,
    MemoryModel.SHARED: pb.MEMORY_MODEL_SHARED,
}
_WIRE_TO_MEMORY_MODEL: dict[pb.MemoryModel, MemoryModel] = {
    v: k for k, v in _MEMORY_MODEL_TO_WIRE.items()
}

_AVAILABILITY_TO_WIRE: dict[DeviceAvailability, pb.DeviceAvailability] = {
    DeviceAvailability.AVAILABLE: pb.DEVICE_AVAILABILITY_AVAILABLE,
    DeviceAvailability.UNAVAILABLE: pb.DEVICE_AVAILABILITY_UNAVAILABLE,
    DeviceAvailability.UNKNOWN: pb.DEVICE_AVAILABILITY_UNKNOWN,
}
_WIRE_TO_AVAILABILITY: dict[pb.DeviceAvailability, DeviceAvailability] = {
    v: k for k, v in _AVAILABILITY_TO_WIRE.items()
}

_RUNTIME_STATUS_TO_WIRE: dict[RuntimeStatus, pb.RuntimeStatus] = {
    RuntimeStatus.CREATED: pb.RUNTIME_STATUS_CREATED,
    RuntimeStatus.RUNNING: pb.RUNTIME_STATUS_RUNNING,
    RuntimeStatus.STOPPED: pb.RUNTIME_STATUS_STOPPED,
    RuntimeStatus.FAILED: pb.RUNTIME_STATUS_FAILED,
    RuntimeStatus.UNKNOWN: pb.RUNTIME_STATUS_UNKNOWN,
}
_WIRE_TO_RUNTIME_STATUS: dict[pb.RuntimeStatus, RuntimeStatus] = {
    v: k for k, v in _RUNTIME_STATUS_TO_WIRE.items()
}

_MODEL_AVAILABILITY_TO_WIRE: dict[ModelAvailability, pb.ModelAvailability] = {
    ModelAvailability.READY: pb.MODEL_AVAILABILITY_READY,
    ModelAvailability.INCOMPLETE: pb.MODEL_AVAILABILITY_INCOMPLETE,
    ModelAvailability.INVALID: pb.MODEL_AVAILABILITY_INVALID,
}
_WIRE_TO_MODEL_AVAILABILITY: dict[pb.ModelAvailability, ModelAvailability] = {
    v: k for k, v in _MODEL_AVAILABILITY_TO_WIRE.items()
}

_REJECTION_REASON_TO_WIRE: dict[RejectionReason, pb.RejectionReason] = {
    RejectionReason.UNKNOWN_WORKER: pb.REJECTION_REASON_UNKNOWN_WORKER,
    RejectionReason.STALE_SESSION: pb.REJECTION_REASON_STALE_SESSION,
    RejectionReason.INSTANCE_MISMATCH: pb.REJECTION_REASON_INSTANCE_MISMATCH,
    RejectionReason.OUT_OF_ORDER: pb.REJECTION_REASON_OUT_OF_ORDER,
    RejectionReason.REREGISTER_REQUIRED: pb.REJECTION_REASON_REREGISTER_REQUIRED,
}
_WIRE_TO_REJECTION_REASON: dict[pb.RejectionReason, RejectionReason] = {
    v: k for k, v in _REJECTION_REASON_TO_WIRE.items()
}

def _decode_enum[WireEnum, DomainEnum](
    table: Mapping[WireEnum, DomainEnum],
    wire_value: WireEnum,
    field: str,
) -> DomainEnum:
    decoded = table.get(wire_value)
    if decoded is None:
        raise ControlProtocolError(f"unknown wire value for {field}: {wire_value!r}")
    return decoded


# ---------------------------------------------------------------------------
# Identity mapping (spec §10)
# ---------------------------------------------------------------------------


def identity_to_wire(identity: WorkerIdentity) -> pb.WorkerIdentity:
    return pb.WorkerIdentity(
        worker_id=identity.worker_id,
        hostname=identity.hostname,
        agent_version=identity.agent_version,
        protocol_version=identity.protocol_version,
    )


def identity_from_wire(wire: pb.WorkerIdentity) -> WorkerIdentity:
    return WorkerIdentity(
        worker_id=wire.worker_id,
        hostname=wire.hostname,
        agent_version=wire.agent_version,
        protocol_version=wire.protocol_version,
    )


# ---------------------------------------------------------------------------
# Capability mapping (spec §12-16)
# ---------------------------------------------------------------------------


def capability_to_wire(capability: WorkerCapability) -> pb.WorkerCapability:
    wire = pb.WorkerCapability(
        architecture=capability.architecture,
        os=_os_to_wire(capability.os),
        network_interfaces=[
            _network_interface_to_wire(interface)
            for interface in capability.network_interfaces
        ],
        devices=[_device_to_wire(device) for device in capability.devices],
        memory_pools=[_memory_pool_to_wire(pool) for pool in capability.memory_pools],
        runtime_platforms=[
            _runtime_platform_to_wire(platform)
            for platform in capability.runtime_platforms
        ],
        capability_revision=capability.capability_revision,
    )
    if capability.container_runtime is not None:
        wire.container_runtime.CopyFrom(
            _container_runtime_to_wire(capability.container_runtime)
        )
    return wire


def capability_from_wire(wire: pb.WorkerCapability) -> WorkerCapability:
    return WorkerCapability(
        architecture=wire.architecture,
        os=_os_from_wire(wire.os),
        container_runtime=(
            _container_runtime_from_wire(wire.container_runtime)
            if wire.HasField("container_runtime")
            else None
        ),
        network_interfaces=tuple(
            _network_interface_from_wire(interface)
            for interface in wire.network_interfaces
        ),
        devices=tuple(_device_from_wire(device) for device in wire.devices),
        memory_pools=tuple(
            _memory_pool_from_wire(pool) for pool in wire.memory_pools
        ),
        runtime_platforms=tuple(
            _runtime_platform_from_wire(platform)
            for platform in wire.runtime_platforms
        ),
        capability_revision=wire.capability_revision,
    )


def _os_to_wire(os_info: OSInfo) -> pb.OSInfo:
    wire = pb.OSInfo(name=os_info.name)
    if os_info.version is not None:
        wire.version = os_info.version
    if os_info.kernel is not None:
        wire.kernel = os_info.kernel
    return wire


def _os_from_wire(wire: pb.OSInfo) -> OSInfo:
    return OSInfo(
        name=wire.name,
        version=wire.version if wire.HasField("version") else None,
        kernel=wire.kernel if wire.HasField("kernel") else None,
    )


def _container_runtime_to_wire(
    runtime: ContainerRuntimeCapability,
) -> pb.ContainerRuntimeCapability:
    wire = pb.ContainerRuntimeCapability(
        runtime=runtime.runtime,
        nvidia_runtime_available=runtime.nvidia_runtime_available,
    )
    if runtime.version is not None:
        wire.version = runtime.version
    return wire


def _container_runtime_from_wire(
    wire: pb.ContainerRuntimeCapability,
) -> ContainerRuntimeCapability:
    return ContainerRuntimeCapability(
        runtime=wire.runtime,
        version=wire.version if wire.HasField("version") else None,
        nvidia_runtime_available=wire.nvidia_runtime_available,
    )


def _network_interface_to_wire(
    interface: NetworkInterfaceCapability,
) -> pb.NetworkInterfaceCapability:
    wire = pb.NetworkInterfaceCapability(
        interface_id=interface.interface_id,
        name=interface.name,
        addresses=list(interface.addresses),
    )
    if interface.mtu is not None:
        wire.mtu = interface.mtu
    return wire


def _network_interface_from_wire(
    wire: pb.NetworkInterfaceCapability,
) -> NetworkInterfaceCapability:
    return NetworkInterfaceCapability(
        interface_id=wire.interface_id,
        name=wire.name,
        addresses=tuple(wire.addresses),
        mtu=wire.mtu if wire.HasField("mtu") else None,
    )


def _runtime_platform_to_wire(
    platform: RuntimePlatformCapability,
) -> pb.RuntimePlatformCapability:
    wire = pb.RuntimePlatformCapability(
        backend=platform.backend, platform=platform.platform
    )
    if platform.image is not None:
        wire.image = platform.image
    return wire


def _runtime_platform_from_wire(
    wire: pb.RuntimePlatformCapability,
) -> RuntimePlatformCapability:
    return RuntimePlatformCapability(
        backend=wire.backend,
        platform=wire.platform,
        image=wire.image if wire.HasField("image") else None,
    )


def _memory_pool_to_wire(pool: MemoryPoolCapability) -> pb.MemoryPoolCapability:
    return pb.MemoryPoolCapability(
        memory_pool_id=pool.memory_pool_id,
        model=_MEMORY_MODEL_TO_WIRE[pool.model],
        total_bytes=pool.total_bytes,
    )


def _memory_pool_from_wire(wire: pb.MemoryPoolCapability) -> MemoryPoolCapability:
    model = _decode_enum(_WIRE_TO_MEMORY_MODEL, wire.model, "memory model")
    return MemoryPoolCapability(
        memory_pool_id=wire.memory_pool_id,
        model=model,
        total_bytes=wire.total_bytes,
    )


def _device_to_wire(device: DeviceCapability) -> pb.DeviceCapability:
    wire = pb.DeviceCapability(
        identity=pb.DeviceIdentity(
            device_id=device.identity.device_id,
            kind=_DEVICE_KIND_TO_WIRE[device.identity.kind],
            local_locator=device.identity.local_locator,
        ),
        vendor=device.vendor,
        model=device.model,
        supported_dtypes=list(device.supported_dtypes),
        platform_tags=list(device.platform_tags),
    )
    if device.compute_capability is not None:
        wire.compute_capability = device.compute_capability
    if device.memory_pool_id is not None:
        wire.memory_pool_id = device.memory_pool_id
    if device.driver_version is not None:
        wire.driver_version = device.driver_version
    return wire


def _device_from_wire(wire: pb.DeviceCapability) -> DeviceCapability:
    kind = _decode_enum(_WIRE_TO_DEVICE_KIND, wire.identity.kind, "device kind")
    return DeviceCapability(
        identity=DeviceIdentity(
            device_id=wire.identity.device_id,
            kind=kind,
            local_locator=wire.identity.local_locator,
        ),
        vendor=wire.vendor,
        model=wire.model,
        compute_capability=(
            wire.compute_capability if wire.HasField("compute_capability") else None
        ),
        memory_pool_id=(
            wire.memory_pool_id if wire.HasField("memory_pool_id") else None
        ),
        supported_dtypes=tuple(wire.supported_dtypes),
        driver_version=(
            wire.driver_version if wire.HasField("driver_version") else None
        ),
        platform_tags=tuple(wire.platform_tags),
    )


# ---------------------------------------------------------------------------
# Dynamic state mapping (spec §17-20)
# ---------------------------------------------------------------------------


def state_to_wire(state: WorkerState) -> pb.WorkerState:
    return pb.WorkerState(
        worker_id=state.worker_id,
        device_states=[_device_state_to_wire(s) for s in state.device_states],
        memory_states=[_memory_state_to_wire(s) for s in state.memory_states],
        runtime_instances=[
            _runtime_instance_to_wire(instance)
            for instance in state.runtime_instances
        ],
        models=[_model_entry_to_wire(entry) for entry in state.models],
    )


def state_from_wire(wire: pb.WorkerState) -> WorkerState:
    return WorkerState(
        worker_id=wire.worker_id,
        device_states=tuple(
            _device_state_from_wire(s) for s in wire.device_states
        ),
        memory_states=tuple(
            _memory_state_from_wire(s) for s in wire.memory_states
        ),
        runtime_instances=tuple(
            _runtime_instance_from_wire(instance)
            for instance in wire.runtime_instances
        ),
        models=tuple(_model_entry_from_wire(entry) for entry in wire.models),
    )


def _device_state_to_wire(state: DeviceState) -> pb.DeviceState:
    wire = pb.DeviceState(
        device_id=state.device_id,
        availability=_AVAILABILITY_TO_WIRE[state.availability],
        running_runtime_ids=list(state.running_runtime_ids),
    )
    if state.utilization is not None:
        wire.utilization = state.utilization
    if state.temperature_c is not None:
        wire.temperature_c = state.temperature_c
    if state.power_w is not None:
        wire.power_w = state.power_w
    return wire


def _device_state_from_wire(wire: pb.DeviceState) -> DeviceState:
    availability = _decode_enum(
        _WIRE_TO_AVAILABILITY, wire.availability, "device availability"
    )
    return DeviceState(
        device_id=wire.device_id,
        utilization=wire.utilization if wire.HasField("utilization") else None,
        temperature_c=wire.temperature_c if wire.HasField("temperature_c") else None,
        power_w=wire.power_w if wire.HasField("power_w") else None,
        availability=availability,
        running_runtime_ids=tuple(wire.running_runtime_ids),
    )


def _memory_state_to_wire(state: MemoryPoolState) -> pb.MemoryPoolState:
    wire = pb.MemoryPoolState(memory_pool_id=state.memory_pool_id)
    if state.available_bytes is not None:
        wire.available_bytes = state.available_bytes
    return wire


def _memory_state_from_wire(wire: pb.MemoryPoolState) -> MemoryPoolState:
    return MemoryPoolState(
        memory_pool_id=wire.memory_pool_id,
        available_bytes=wire.available_bytes if wire.HasField("available_bytes") else None,
    )


def _runtime_instance_to_wire(
    instance: RuntimeInstanceState,
) -> pb.RuntimeInstanceState:
    wire = pb.RuntimeInstanceState(
        runtime_id=instance.runtime_id,
        backend=instance.backend,
        status=_RUNTIME_STATUS_TO_WIRE[instance.status],
        device_ids=list(instance.device_ids),
    )
    if instance.execution_id is not None:
        wire.execution_id = instance.execution_id
    if instance.container_id is not None:
        wire.container_id = instance.container_id
    if instance.endpoint is not None:
        wire.endpoint = instance.endpoint
    if instance.model_local_name is not None:
        wire.model_local_name = instance.model_local_name
    return wire


def _runtime_instance_from_wire(wire: pb.RuntimeInstanceState) -> RuntimeInstanceState:
    status = _decode_enum(
        _WIRE_TO_RUNTIME_STATUS, wire.status, "runtime status"
    )
    return RuntimeInstanceState(
        runtime_id=wire.runtime_id,
        backend=wire.backend,
        execution_id=(
            wire.execution_id if wire.HasField("execution_id") else None
        ),
        status=status,
        device_ids=tuple(wire.device_ids),
        container_id=(
            wire.container_id if wire.HasField("container_id") else None
        ),
        endpoint=wire.endpoint if wire.HasField("endpoint") else None,
        model_local_name=(
            wire.model_local_name if wire.HasField("model_local_name") else None
        ),
    )


def _model_entry_to_wire(entry: ModelInventoryEntry) -> pb.ModelInventoryEntry:
    wire = pb.ModelInventoryEntry(
        local_name=entry.local_name,
        status=_MODEL_AVAILABILITY_TO_WIRE[entry.status],
    )
    if entry.model_id is not None:
        wire.model_id = entry.model_id
    if entry.revision is not None:
        wire.revision = entry.revision
    if entry.size_bytes is not None:
        wire.size_bytes = entry.size_bytes
    return wire


def _model_entry_from_wire(wire: pb.ModelInventoryEntry) -> ModelInventoryEntry:
    status = _decode_enum(
        _WIRE_TO_MODEL_AVAILABILITY, wire.status, "model availability"
    )
    return ModelInventoryEntry(
        local_name=wire.local_name,
        model_id=wire.model_id if wire.HasField("model_id") else None,
        revision=wire.revision if wire.HasField("revision") else None,
        size_bytes=wire.size_bytes if wire.HasField("size_bytes") else None,
        status=status,
    )


# ---------------------------------------------------------------------------
# Timestamps (spec §31): wall-clock milliseconds, debugging aid only
# ---------------------------------------------------------------------------


def _timestamp_to_ms(reported_at: datetime | None) -> int | None:
    if reported_at is None:
        return None
    return int(reported_at.timestamp() * 1000)


def _timestamp_from_ms(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    # Integer arithmetic keeps the conversion exact at millisecond precision.
    return datetime.fromtimestamp(0, tz=UTC) + timedelta(milliseconds=ms)


# ---------------------------------------------------------------------------
# RPC message mapping (spec §29-30)
# ---------------------------------------------------------------------------


def register_request_to_wire(request: RegisterWorkerRequest) -> pb.RegisterWorkerRequest:
    wire = pb.RegisterWorkerRequest(
        protocol_version=request.protocol_version,
        worker_id=request.identity.worker_id,
        instance_id=request.instance_id,
        identity=identity_to_wire(request.identity),
        capability_revision=request.capability.capability_revision,
        capability=capability_to_wire(request.capability),
        initial_state=state_to_wire(request.initial_state),
    )
    # proto3 optional: absence (never a sentinel) encodes "no profiling" (§17).
    if request.profiling_endpoint is not None:
        wire.profiling_endpoint = request.profiling_endpoint
    return wire


def register_request_from_wire(wire: pb.RegisterWorkerRequest) -> RegisterWorkerRequest:
    if wire.protocol_version != CONTROL_PROTOCOL_VERSION:
        raise ControlProtocolError(
            f"unsupported control protocol version "
            f"{wire.protocol_version!r} (expected {CONTROL_PROTOCOL_VERSION!r})"
        )
    identity = identity_from_wire(wire.identity)
    if wire.worker_id != identity.worker_id:
        raise ControlProtocolError(
            f"registration worker_id mismatch: {wire.worker_id!r} vs "
            f"{identity.worker_id!r} in identity"
        )
    capability = capability_from_wire(wire.capability)
    if wire.capability_revision != capability.capability_revision:
        raise ControlProtocolError(
            f"registration capability_revision mismatch: "
            f"{wire.capability_revision!r} vs "
            f"{capability.capability_revision!r} in capability"
        )
    return RegisterWorkerRequest(
        protocol_version=wire.protocol_version,
        instance_id=wire.instance_id,
        identity=identity,
        capability=capability,
        initial_state=state_from_wire(wire.initial_state),
        profiling_endpoint=(
            wire.profiling_endpoint if wire.HasField("profiling_endpoint") else None
        ),
    )


def register_response_to_wire(
    response: RegisterWorkerResponse,
) -> pb.RegisterWorkerResponse:
    return pb.RegisterWorkerResponse(
        session_id=response.session_id,
        heartbeat_interval_ms=response.heartbeat_interval_ms,
        server_protocol_version=response.server_protocol_version,
    )


def register_response_from_wire(
    wire: pb.RegisterWorkerResponse,
) -> RegisterWorkerResponse:
    return RegisterWorkerResponse(
        session_id=wire.session_id,
        heartbeat_interval_ms=wire.heartbeat_interval_ms,
        server_protocol_version=wire.server_protocol_version,
    )


def heartbeat_request_to_wire(request: HeartbeatRequest) -> pb.HeartbeatRequest:
    wire = pb.HeartbeatRequest(
        worker_id=request.worker_id,
        instance_id=request.instance_id,
        session_id=request.session_id,
        sequence_number=request.sequence_number,
        capability_revision=request.capability_revision,
        state=state_to_wire(request.state),
    )
    reported_at_ms = _timestamp_to_ms(request.worker_reported_at)
    if reported_at_ms is not None:
        wire.worker_reported_at_ms = reported_at_ms
    return wire


def heartbeat_request_from_wire(wire: pb.HeartbeatRequest) -> HeartbeatRequest:
    return HeartbeatRequest(
        worker_id=wire.worker_id,
        instance_id=wire.instance_id,
        session_id=wire.session_id,
        sequence_number=wire.sequence_number,
        capability_revision=wire.capability_revision,
        state=state_from_wire(wire.state),
        worker_reported_at=(
            _timestamp_from_ms(wire.worker_reported_at_ms)
            if wire.HasField("worker_reported_at_ms")
            else None
        ),
    )


def _rejection_reason_to_wire(reason: RejectionReason | None) -> pb.RejectionReason:
    if reason is None:
        return pb.REJECTION_REASON_UNSPECIFIED
    return _REJECTION_REASON_TO_WIRE[reason]


def _rejection_reason_from_wire(
    wire_reason: pb.RejectionReason, *, accepted: bool, label: str
) -> RejectionReason | None:
    if accepted:
        # An accepted verdict carries no reason; ignore whatever the wire says.
        return None
    reason = _WIRE_TO_REJECTION_REASON.get(wire_reason)
    if reason is None:
        # Fail loudly (§47): a rejection without a usable reason leaves the
        # Worker Agent unable to choose a recovery path.
        raise ControlProtocolError(
            f"rejected {label} carries unknown rejection reason {wire_reason!r}"
        )
    return reason


def heartbeat_response_to_wire(response: HeartbeatResponse) -> pb.HeartbeatResponse:
    return pb.HeartbeatResponse(
        accepted=response.accepted,
        detail=response.detail,
        reason=_rejection_reason_to_wire(response.reason),
    )


def heartbeat_response_from_wire(wire: pb.HeartbeatResponse) -> HeartbeatResponse:
    return HeartbeatResponse(
        accepted=wire.accepted,
        detail=wire.detail,
        reason=_rejection_reason_from_wire(
            wire.reason, accepted=wire.accepted, label="heartbeat"
        ),
    )


def update_capability_request_to_wire(
    request: UpdateCapabilityRequest,
) -> pb.UpdateCapabilityRequest:
    return pb.UpdateCapabilityRequest(
        worker_id=request.worker_id,
        instance_id=request.instance_id,
        session_id=request.session_id,
        capability=capability_to_wire(request.capability),
        state=state_to_wire(request.state),
    )


def update_capability_request_from_wire(
    wire: pb.UpdateCapabilityRequest,
) -> UpdateCapabilityRequest:
    if not wire.HasField("state"):
        raise ControlProtocolError(
            "capability update must carry the state sampled atomically with "
            "the new capability (spec §16)"
        )
    return UpdateCapabilityRequest(
        worker_id=wire.worker_id,
        instance_id=wire.instance_id,
        session_id=wire.session_id,
        capability=capability_from_wire(wire.capability),
        state=state_from_wire(wire.state),
    )


def update_capability_response_to_wire(
    response: UpdateCapabilityResponse,
) -> pb.UpdateCapabilityResponse:
    return pb.UpdateCapabilityResponse(
        accepted=response.accepted,
        detail=response.detail,
        reason=_rejection_reason_to_wire(response.reason),
    )


def update_capability_response_from_wire(
    wire: pb.UpdateCapabilityResponse,
) -> UpdateCapabilityResponse:
    return UpdateCapabilityResponse(
        accepted=wire.accepted,
        detail=wire.detail,
        reason=_rejection_reason_from_wire(
            wire.reason, accepted=wire.accepted, label="capability update"
        ),
    )
