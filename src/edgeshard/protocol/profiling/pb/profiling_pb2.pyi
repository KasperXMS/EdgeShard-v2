from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ProfilingRejectionReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROFILING_REJECTION_REASON_UNSPECIFIED: _ClassVar[ProfilingRejectionReason]
    PROFILING_REJECTION_REASON_UNKNOWN_SESSION: _ClassVar[ProfilingRejectionReason]
    PROFILING_REJECTION_REASON_STALE_SESSION: _ClassVar[ProfilingRejectionReason]
    PROFILING_REJECTION_REASON_SESSION_CLOSED: _ClassVar[ProfilingRejectionReason]
    PROFILING_REJECTION_REASON_SESSION_KIND_MISMATCH: _ClassVar[ProfilingRejectionReason]
    PROFILING_REJECTION_REASON_DEVICE_BUSY: _ClassVar[ProfilingRejectionReason]
    PROFILING_REJECTION_REASON_UNKNOWN_CASE: _ClassVar[ProfilingRejectionReason]
PROFILING_REJECTION_REASON_UNSPECIFIED: ProfilingRejectionReason
PROFILING_REJECTION_REASON_UNKNOWN_SESSION: ProfilingRejectionReason
PROFILING_REJECTION_REASON_STALE_SESSION: ProfilingRejectionReason
PROFILING_REJECTION_REASON_SESSION_CLOSED: ProfilingRejectionReason
PROFILING_REJECTION_REASON_SESSION_KIND_MISMATCH: ProfilingRejectionReason
PROFILING_REJECTION_REASON_DEVICE_BUSY: ProfilingRejectionReason
PROFILING_REJECTION_REASON_UNKNOWN_CASE: ProfilingRejectionReason

class PrepareProfilingSessionRequest(_message.Message):
    __slots__ = ("worker_id", "instance_id", "registration_session_id", "profiling_session_id", "session_request_payload", "network_facts_payload")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    REGISTRATION_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    PROFILING_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_REQUEST_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    NETWORK_FACTS_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    instance_id: str
    registration_session_id: str
    profiling_session_id: str
    session_request_payload: str
    network_facts_payload: str
    def __init__(self, worker_id: _Optional[str] = ..., instance_id: _Optional[str] = ..., registration_session_id: _Optional[str] = ..., profiling_session_id: _Optional[str] = ..., session_request_payload: _Optional[str] = ..., network_facts_payload: _Optional[str] = ...) -> None: ...

class PrepareProfilingSessionResponse(_message.Message):
    __slots__ = ("accepted", "detail", "reason", "session_facts_payload", "failure_payload")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    SESSION_FACTS_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    FAILURE_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    detail: str
    reason: ProfilingRejectionReason
    session_facts_payload: str
    failure_payload: str
    def __init__(self, accepted: _Optional[bool] = ..., detail: _Optional[str] = ..., reason: _Optional[_Union[ProfilingRejectionReason, str]] = ..., session_facts_payload: _Optional[str] = ..., failure_payload: _Optional[str] = ...) -> None: ...

class RunProfilingCaseRequest(_message.Message):
    __slots__ = ("worker_id", "instance_id", "registration_session_id", "profiling_session_id", "case_id", "case_payload")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    REGISTRATION_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    PROFILING_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CASE_ID_FIELD_NUMBER: _ClassVar[int]
    CASE_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    instance_id: str
    registration_session_id: str
    profiling_session_id: str
    case_id: str
    case_payload: str
    def __init__(self, worker_id: _Optional[str] = ..., instance_id: _Optional[str] = ..., registration_session_id: _Optional[str] = ..., profiling_session_id: _Optional[str] = ..., case_id: _Optional[str] = ..., case_payload: _Optional[str] = ...) -> None: ...

class RunProfilingCaseResponse(_message.Message):
    __slots__ = ("accepted", "detail", "reason", "outcome_payload")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    detail: str
    reason: ProfilingRejectionReason
    outcome_payload: str
    def __init__(self, accepted: _Optional[bool] = ..., detail: _Optional[str] = ..., reason: _Optional[_Union[ProfilingRejectionReason, str]] = ..., outcome_payload: _Optional[str] = ...) -> None: ...

class GetProfilingCaseRequest(_message.Message):
    __slots__ = ("worker_id", "instance_id", "registration_session_id", "profiling_session_id", "case_id")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    REGISTRATION_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    PROFILING_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CASE_ID_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    instance_id: str
    registration_session_id: str
    profiling_session_id: str
    case_id: str
    def __init__(self, worker_id: _Optional[str] = ..., instance_id: _Optional[str] = ..., registration_session_id: _Optional[str] = ..., profiling_session_id: _Optional[str] = ..., case_id: _Optional[str] = ...) -> None: ...

class GetProfilingCaseResponse(_message.Message):
    __slots__ = ("accepted", "detail", "reason", "case_state", "outcome_payload")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    CASE_STATE_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    detail: str
    reason: ProfilingRejectionReason
    case_state: str
    outcome_payload: str
    def __init__(self, accepted: _Optional[bool] = ..., detail: _Optional[str] = ..., reason: _Optional[_Union[ProfilingRejectionReason, str]] = ..., case_state: _Optional[str] = ..., outcome_payload: _Optional[str] = ...) -> None: ...

class CancelProfilingCaseRequest(_message.Message):
    __slots__ = ("worker_id", "instance_id", "registration_session_id", "profiling_session_id", "case_id")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    REGISTRATION_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    PROFILING_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CASE_ID_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    instance_id: str
    registration_session_id: str
    profiling_session_id: str
    case_id: str
    def __init__(self, worker_id: _Optional[str] = ..., instance_id: _Optional[str] = ..., registration_session_id: _Optional[str] = ..., profiling_session_id: _Optional[str] = ..., case_id: _Optional[str] = ...) -> None: ...

class CancelProfilingCaseResponse(_message.Message):
    __slots__ = ("accepted", "detail", "reason", "case_state")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    CASE_STATE_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    detail: str
    reason: ProfilingRejectionReason
    case_state: str
    def __init__(self, accepted: _Optional[bool] = ..., detail: _Optional[str] = ..., reason: _Optional[_Union[ProfilingRejectionReason, str]] = ..., case_state: _Optional[str] = ...) -> None: ...

class CloseProfilingSessionRequest(_message.Message):
    __slots__ = ("worker_id", "instance_id", "registration_session_id", "profiling_session_id")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    REGISTRATION_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    PROFILING_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    instance_id: str
    registration_session_id: str
    profiling_session_id: str
    def __init__(self, worker_id: _Optional[str] = ..., instance_id: _Optional[str] = ..., registration_session_id: _Optional[str] = ..., profiling_session_id: _Optional[str] = ...) -> None: ...

class CloseProfilingSessionResponse(_message.Message):
    __slots__ = ("accepted", "detail", "reason")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    detail: str
    reason: ProfilingRejectionReason
    def __init__(self, accepted: _Optional[bool] = ..., detail: _Optional[str] = ..., reason: _Optional[_Union[ProfilingRejectionReason, str]] = ...) -> None: ...

class StartExperimentRequest(_message.Message):
    __slots__ = ("request_payload",)
    REQUEST_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    request_payload: str
    def __init__(self, request_payload: _Optional[str] = ...) -> None: ...

class StartExperimentResponse(_message.Message):
    __slots__ = ("accepted", "detail", "experiment_id")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    detail: str
    experiment_id: str
    def __init__(self, accepted: _Optional[bool] = ..., detail: _Optional[str] = ..., experiment_id: _Optional[str] = ...) -> None: ...

class GetExperimentRequest(_message.Message):
    __slots__ = ("experiment_id",)
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    def __init__(self, experiment_id: _Optional[str] = ...) -> None: ...

class GetExperimentResponse(_message.Message):
    __slots__ = ("found", "status_payload")
    FOUND_FIELD_NUMBER: _ClassVar[int]
    STATUS_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    found: bool
    status_payload: str
    def __init__(self, found: _Optional[bool] = ..., status_payload: _Optional[str] = ...) -> None: ...

class CancelExperimentRequest(_message.Message):
    __slots__ = ("experiment_id",)
    EXPERIMENT_ID_FIELD_NUMBER: _ClassVar[int]
    experiment_id: str
    def __init__(self, experiment_id: _Optional[str] = ...) -> None: ...

class CancelExperimentResponse(_message.Message):
    __slots__ = ("accepted", "detail")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    detail: str
    def __init__(self, accepted: _Optional[bool] = ..., detail: _Optional[str] = ...) -> None: ...

class BuildProfileSnapshotRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class BuildProfileSnapshotResponse(_message.Message):
    __slots__ = ("accepted", "detail", "snapshot_payload")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    detail: str
    snapshot_payload: str
    def __init__(self, accepted: _Optional[bool] = ..., detail: _Optional[str] = ..., snapshot_payload: _Optional[str] = ...) -> None: ...
