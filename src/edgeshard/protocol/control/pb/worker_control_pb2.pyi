from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DeviceKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DEVICE_KIND_UNSPECIFIED: _ClassVar[DeviceKind]
    DEVICE_KIND_CPU: _ClassVar[DeviceKind]
    DEVICE_KIND_GPU: _ClassVar[DeviceKind]
    DEVICE_KIND_NPU: _ClassVar[DeviceKind]
    DEVICE_KIND_OTHER: _ClassVar[DeviceKind]

class MemoryModel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MEMORY_MODEL_UNSPECIFIED: _ClassVar[MemoryModel]
    MEMORY_MODEL_DISCRETE: _ClassVar[MemoryModel]
    MEMORY_MODEL_SHARED: _ClassVar[MemoryModel]

class DeviceAvailability(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DEVICE_AVAILABILITY_UNSPECIFIED: _ClassVar[DeviceAvailability]
    DEVICE_AVAILABILITY_AVAILABLE: _ClassVar[DeviceAvailability]
    DEVICE_AVAILABILITY_UNAVAILABLE: _ClassVar[DeviceAvailability]
    DEVICE_AVAILABILITY_UNKNOWN: _ClassVar[DeviceAvailability]

class RuntimeStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RUNTIME_STATUS_UNSPECIFIED: _ClassVar[RuntimeStatus]
    RUNTIME_STATUS_CREATED: _ClassVar[RuntimeStatus]
    RUNTIME_STATUS_RUNNING: _ClassVar[RuntimeStatus]
    RUNTIME_STATUS_STOPPED: _ClassVar[RuntimeStatus]
    RUNTIME_STATUS_FAILED: _ClassVar[RuntimeStatus]
    RUNTIME_STATUS_UNKNOWN: _ClassVar[RuntimeStatus]

class ModelAvailability(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_AVAILABILITY_UNSPECIFIED: _ClassVar[ModelAvailability]
    MODEL_AVAILABILITY_READY: _ClassVar[ModelAvailability]
    MODEL_AVAILABILITY_INCOMPLETE: _ClassVar[ModelAvailability]
    MODEL_AVAILABILITY_INVALID: _ClassVar[ModelAvailability]
DEVICE_KIND_UNSPECIFIED: DeviceKind
DEVICE_KIND_CPU: DeviceKind
DEVICE_KIND_GPU: DeviceKind
DEVICE_KIND_NPU: DeviceKind
DEVICE_KIND_OTHER: DeviceKind
MEMORY_MODEL_UNSPECIFIED: MemoryModel
MEMORY_MODEL_DISCRETE: MemoryModel
MEMORY_MODEL_SHARED: MemoryModel
DEVICE_AVAILABILITY_UNSPECIFIED: DeviceAvailability
DEVICE_AVAILABILITY_AVAILABLE: DeviceAvailability
DEVICE_AVAILABILITY_UNAVAILABLE: DeviceAvailability
DEVICE_AVAILABILITY_UNKNOWN: DeviceAvailability
RUNTIME_STATUS_UNSPECIFIED: RuntimeStatus
RUNTIME_STATUS_CREATED: RuntimeStatus
RUNTIME_STATUS_RUNNING: RuntimeStatus
RUNTIME_STATUS_STOPPED: RuntimeStatus
RUNTIME_STATUS_FAILED: RuntimeStatus
RUNTIME_STATUS_UNKNOWN: RuntimeStatus
MODEL_AVAILABILITY_UNSPECIFIED: ModelAvailability
MODEL_AVAILABILITY_READY: ModelAvailability
MODEL_AVAILABILITY_INCOMPLETE: ModelAvailability
MODEL_AVAILABILITY_INVALID: ModelAvailability

class WorkerIdentity(_message.Message):
    __slots__ = ("worker_id", "hostname", "agent_version", "protocol_version")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    AGENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    hostname: str
    agent_version: str
    protocol_version: str
    def __init__(self, worker_id: _Optional[str] = ..., hostname: _Optional[str] = ..., agent_version: _Optional[str] = ..., protocol_version: _Optional[str] = ...) -> None: ...

class DeviceIdentity(_message.Message):
    __slots__ = ("device_id", "kind", "local_locator")
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    LOCAL_LOCATOR_FIELD_NUMBER: _ClassVar[int]
    device_id: str
    kind: DeviceKind
    local_locator: str
    def __init__(self, device_id: _Optional[str] = ..., kind: _Optional[_Union[DeviceKind, str]] = ..., local_locator: _Optional[str] = ...) -> None: ...

class MemoryPoolCapability(_message.Message):
    __slots__ = ("memory_pool_id", "model", "total_bytes")
    MEMORY_POOL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    memory_pool_id: str
    model: MemoryModel
    total_bytes: int
    def __init__(self, memory_pool_id: _Optional[str] = ..., model: _Optional[_Union[MemoryModel, str]] = ..., total_bytes: _Optional[int] = ...) -> None: ...

class OSInfo(_message.Message):
    __slots__ = ("name", "version", "kernel")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    KERNEL_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: str
    kernel: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., kernel: _Optional[str] = ...) -> None: ...

class ContainerRuntimeCapability(_message.Message):
    __slots__ = ("runtime", "version", "nvidia_runtime_available")
    RUNTIME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    NVIDIA_RUNTIME_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    runtime: str
    version: str
    nvidia_runtime_available: bool
    def __init__(self, runtime: _Optional[str] = ..., version: _Optional[str] = ..., nvidia_runtime_available: _Optional[bool] = ...) -> None: ...

class NetworkInterfaceCapability(_message.Message):
    __slots__ = ("interface_id", "name", "addresses", "mtu")
    INTERFACE_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ADDRESSES_FIELD_NUMBER: _ClassVar[int]
    MTU_FIELD_NUMBER: _ClassVar[int]
    interface_id: str
    name: str
    addresses: _containers.RepeatedScalarFieldContainer[str]
    mtu: int
    def __init__(self, interface_id: _Optional[str] = ..., name: _Optional[str] = ..., addresses: _Optional[_Iterable[str]] = ..., mtu: _Optional[int] = ...) -> None: ...

class RuntimePlatformCapability(_message.Message):
    __slots__ = ("backend", "platform", "image")
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    backend: str
    platform: str
    image: str
    def __init__(self, backend: _Optional[str] = ..., platform: _Optional[str] = ..., image: _Optional[str] = ...) -> None: ...

class DeviceCapability(_message.Message):
    __slots__ = ("identity", "vendor", "model", "compute_capability", "memory_pool_id", "supported_dtypes", "driver_version", "platform_tags")
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    VENDOR_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    MEMORY_POOL_ID_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_DTYPES_FIELD_NUMBER: _ClassVar[int]
    DRIVER_VERSION_FIELD_NUMBER: _ClassVar[int]
    PLATFORM_TAGS_FIELD_NUMBER: _ClassVar[int]
    identity: DeviceIdentity
    vendor: str
    model: str
    compute_capability: str
    memory_pool_id: str
    supported_dtypes: _containers.RepeatedScalarFieldContainer[str]
    driver_version: str
    platform_tags: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, identity: _Optional[_Union[DeviceIdentity, _Mapping]] = ..., vendor: _Optional[str] = ..., model: _Optional[str] = ..., compute_capability: _Optional[str] = ..., memory_pool_id: _Optional[str] = ..., supported_dtypes: _Optional[_Iterable[str]] = ..., driver_version: _Optional[str] = ..., platform_tags: _Optional[_Iterable[str]] = ...) -> None: ...

class WorkerCapability(_message.Message):
    __slots__ = ("architecture", "os", "container_runtime", "network_interfaces", "devices", "memory_pools", "runtime_platforms", "capability_revision")
    ARCHITECTURE_FIELD_NUMBER: _ClassVar[int]
    OS_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_RUNTIME_FIELD_NUMBER: _ClassVar[int]
    NETWORK_INTERFACES_FIELD_NUMBER: _ClassVar[int]
    DEVICES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_POOLS_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_PLATFORMS_FIELD_NUMBER: _ClassVar[int]
    CAPABILITY_REVISION_FIELD_NUMBER: _ClassVar[int]
    architecture: str
    os: OSInfo
    container_runtime: ContainerRuntimeCapability
    network_interfaces: _containers.RepeatedCompositeFieldContainer[NetworkInterfaceCapability]
    devices: _containers.RepeatedCompositeFieldContainer[DeviceCapability]
    memory_pools: _containers.RepeatedCompositeFieldContainer[MemoryPoolCapability]
    runtime_platforms: _containers.RepeatedCompositeFieldContainer[RuntimePlatformCapability]
    capability_revision: str
    def __init__(self, architecture: _Optional[str] = ..., os: _Optional[_Union[OSInfo, _Mapping]] = ..., container_runtime: _Optional[_Union[ContainerRuntimeCapability, _Mapping]] = ..., network_interfaces: _Optional[_Iterable[_Union[NetworkInterfaceCapability, _Mapping]]] = ..., devices: _Optional[_Iterable[_Union[DeviceCapability, _Mapping]]] = ..., memory_pools: _Optional[_Iterable[_Union[MemoryPoolCapability, _Mapping]]] = ..., runtime_platforms: _Optional[_Iterable[_Union[RuntimePlatformCapability, _Mapping]]] = ..., capability_revision: _Optional[str] = ...) -> None: ...

class DeviceState(_message.Message):
    __slots__ = ("device_id", "utilization", "temperature_c", "power_w", "availability", "running_runtime_ids")
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_C_FIELD_NUMBER: _ClassVar[int]
    POWER_W_FIELD_NUMBER: _ClassVar[int]
    AVAILABILITY_FIELD_NUMBER: _ClassVar[int]
    RUNNING_RUNTIME_IDS_FIELD_NUMBER: _ClassVar[int]
    device_id: str
    utilization: float
    temperature_c: float
    power_w: float
    availability: DeviceAvailability
    running_runtime_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, device_id: _Optional[str] = ..., utilization: _Optional[float] = ..., temperature_c: _Optional[float] = ..., power_w: _Optional[float] = ..., availability: _Optional[_Union[DeviceAvailability, str]] = ..., running_runtime_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class MemoryPoolState(_message.Message):
    __slots__ = ("memory_pool_id", "available_bytes")
    MEMORY_POOL_ID_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_BYTES_FIELD_NUMBER: _ClassVar[int]
    memory_pool_id: str
    available_bytes: int
    def __init__(self, memory_pool_id: _Optional[str] = ..., available_bytes: _Optional[int] = ...) -> None: ...

class RuntimeInstanceState(_message.Message):
    __slots__ = ("runtime_id", "backend", "execution_id", "status", "device_ids", "container_id", "endpoint", "model_local_name")
    RUNTIME_ID_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DEVICE_IDS_FIELD_NUMBER: _ClassVar[int]
    CONTAINER_ID_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    MODEL_LOCAL_NAME_FIELD_NUMBER: _ClassVar[int]
    runtime_id: str
    backend: str
    execution_id: str
    status: RuntimeStatus
    device_ids: _containers.RepeatedScalarFieldContainer[str]
    container_id: str
    endpoint: str
    model_local_name: str
    def __init__(self, runtime_id: _Optional[str] = ..., backend: _Optional[str] = ..., execution_id: _Optional[str] = ..., status: _Optional[_Union[RuntimeStatus, str]] = ..., device_ids: _Optional[_Iterable[str]] = ..., container_id: _Optional[str] = ..., endpoint: _Optional[str] = ..., model_local_name: _Optional[str] = ...) -> None: ...

class ModelInventoryEntry(_message.Message):
    __slots__ = ("local_name", "model_id", "revision", "size_bytes", "status")
    LOCAL_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    local_name: str
    model_id: str
    revision: str
    size_bytes: int
    status: ModelAvailability
    def __init__(self, local_name: _Optional[str] = ..., model_id: _Optional[str] = ..., revision: _Optional[str] = ..., size_bytes: _Optional[int] = ..., status: _Optional[_Union[ModelAvailability, str]] = ...) -> None: ...

class WorkerState(_message.Message):
    __slots__ = ("worker_id", "device_states", "memory_states", "runtime_instances", "models")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_STATES_FIELD_NUMBER: _ClassVar[int]
    MEMORY_STATES_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_INSTANCES_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    device_states: _containers.RepeatedCompositeFieldContainer[DeviceState]
    memory_states: _containers.RepeatedCompositeFieldContainer[MemoryPoolState]
    runtime_instances: _containers.RepeatedCompositeFieldContainer[RuntimeInstanceState]
    models: _containers.RepeatedCompositeFieldContainer[ModelInventoryEntry]
    def __init__(self, worker_id: _Optional[str] = ..., device_states: _Optional[_Iterable[_Union[DeviceState, _Mapping]]] = ..., memory_states: _Optional[_Iterable[_Union[MemoryPoolState, _Mapping]]] = ..., runtime_instances: _Optional[_Iterable[_Union[RuntimeInstanceState, _Mapping]]] = ..., models: _Optional[_Iterable[_Union[ModelInventoryEntry, _Mapping]]] = ...) -> None: ...

class RegisterWorkerRequest(_message.Message):
    __slots__ = ("protocol_version", "worker_id", "instance_id", "identity", "capability_revision", "capability", "initial_state")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    CAPABILITY_REVISION_FIELD_NUMBER: _ClassVar[int]
    CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    INITIAL_STATE_FIELD_NUMBER: _ClassVar[int]
    protocol_version: str
    worker_id: str
    instance_id: str
    identity: WorkerIdentity
    capability_revision: str
    capability: WorkerCapability
    initial_state: WorkerState
    def __init__(self, protocol_version: _Optional[str] = ..., worker_id: _Optional[str] = ..., instance_id: _Optional[str] = ..., identity: _Optional[_Union[WorkerIdentity, _Mapping]] = ..., capability_revision: _Optional[str] = ..., capability: _Optional[_Union[WorkerCapability, _Mapping]] = ..., initial_state: _Optional[_Union[WorkerState, _Mapping]] = ...) -> None: ...

class RegisterWorkerResponse(_message.Message):
    __slots__ = ("session_id", "heartbeat_interval_ms", "server_protocol_version")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_INTERVAL_MS_FIELD_NUMBER: _ClassVar[int]
    SERVER_PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    heartbeat_interval_ms: int
    server_protocol_version: str
    def __init__(self, session_id: _Optional[str] = ..., heartbeat_interval_ms: _Optional[int] = ..., server_protocol_version: _Optional[str] = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("worker_id", "instance_id", "session_id", "sequence_number", "worker_reported_at_ms", "capability_revision", "state")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    WORKER_REPORTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    CAPABILITY_REVISION_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    instance_id: str
    session_id: str
    sequence_number: int
    worker_reported_at_ms: int
    capability_revision: str
    state: WorkerState
    def __init__(self, worker_id: _Optional[str] = ..., instance_id: _Optional[str] = ..., session_id: _Optional[str] = ..., sequence_number: _Optional[int] = ..., worker_reported_at_ms: _Optional[int] = ..., capability_revision: _Optional[str] = ..., state: _Optional[_Union[WorkerState, _Mapping]] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ("accepted", "detail")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    detail: str
    def __init__(self, accepted: _Optional[bool] = ..., detail: _Optional[str] = ...) -> None: ...

class UpdateCapabilityRequest(_message.Message):
    __slots__ = ("worker_id", "instance_id", "session_id", "capability")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    instance_id: str
    session_id: str
    capability: WorkerCapability
    def __init__(self, worker_id: _Optional[str] = ..., instance_id: _Optional[str] = ..., session_id: _Optional[str] = ..., capability: _Optional[_Union[WorkerCapability, _Mapping]] = ...) -> None: ...

class UpdateCapabilityResponse(_message.Message):
    __slots__ = ("accepted", "detail")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    detail: str
    def __init__(self, accepted: _Optional[bool] = ..., detail: _Optional[str] = ...) -> None: ...
