from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Phase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PHASE_UNSPECIFIED: _ClassVar[Phase]
    PHASE_PREFILL: _ClassVar[Phase]
    PHASE_DECODE: _ClassVar[Phase]
PHASE_UNSPECIFIED: Phase
PHASE_PREFILL: Phase
PHASE_DECODE: Phase

class MessageHeader(_message.Message):
    __slots__ = ("protocol_version", "execution_id", "session_id", "request_id", "phase", "step", "source_stage", "target_stage")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    SOURCE_STAGE_FIELD_NUMBER: _ClassVar[int]
    TARGET_STAGE_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    execution_id: str
    session_id: str
    request_id: str
    phase: Phase
    step: int
    source_stage: int
    target_stage: int
    def __init__(self, protocol_version: _Optional[int] = ..., execution_id: _Optional[str] = ..., session_id: _Optional[str] = ..., request_id: _Optional[str] = ..., phase: _Optional[_Union[Phase, str]] = ..., step: _Optional[int] = ..., source_stage: _Optional[int] = ..., target_stage: _Optional[int] = ...) -> None: ...

class ExecutionContext(_message.Message):
    __slots__ = ("phase", "step", "batch_size", "sequence_lengths", "past_length")
    PHASE_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_LENGTHS_FIELD_NUMBER: _ClassVar[int]
    PAST_LENGTH_FIELD_NUMBER: _ClassVar[int]
    phase: Phase
    step: int
    batch_size: int
    sequence_lengths: _containers.RepeatedScalarFieldContainer[int]
    past_length: int
    def __init__(self, phase: _Optional[_Union[Phase, str]] = ..., step: _Optional[int] = ..., batch_size: _Optional[int] = ..., sequence_lengths: _Optional[_Iterable[int]] = ..., past_length: _Optional[int] = ...) -> None: ...

class TokenPayload(_message.Message):
    __slots__ = ("token_id",)
    TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    token_id: int
    def __init__(self, token_id: _Optional[int] = ...) -> None: ...

class HiddenStatePayload(_message.Message):
    __slots__ = ("tensor_key",)
    TENSOR_KEY_FIELD_NUMBER: _ClassVar[int]
    tensor_key: str
    def __init__(self, tensor_key: _Optional[str] = ...) -> None: ...

class LogitsPayload(_message.Message):
    __slots__ = ("tensor_key",)
    TENSOR_KEY_FIELD_NUMBER: _ClassVar[int]
    tensor_key: str
    def __init__(self, tensor_key: _Optional[str] = ...) -> None: ...

class ShardPayload(_message.Message):
    __slots__ = ("token", "hidden_states", "logits")
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_STATES_FIELD_NUMBER: _ClassVar[int]
    LOGITS_FIELD_NUMBER: _ClassVar[int]
    token: TokenPayload
    hidden_states: HiddenStatePayload
    logits: LogitsPayload
    def __init__(self, token: _Optional[_Union[TokenPayload, _Mapping]] = ..., hidden_states: _Optional[_Union[HiddenStatePayload, _Mapping]] = ..., logits: _Optional[_Union[LogitsPayload, _Mapping]] = ...) -> None: ...

class ForwardRequest(_message.Message):
    __slots__ = ("header", "context", "payload", "tensor_bundle")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    TENSOR_BUNDLE_FIELD_NUMBER: _ClassVar[int]
    header: MessageHeader
    context: ExecutionContext
    payload: ShardPayload
    tensor_bundle: bytes
    def __init__(self, header: _Optional[_Union[MessageHeader, _Mapping]] = ..., context: _Optional[_Union[ExecutionContext, _Mapping]] = ..., payload: _Optional[_Union[ShardPayload, _Mapping]] = ..., tensor_bundle: _Optional[bytes] = ...) -> None: ...

class ForwardReply(_message.Message):
    __slots__ = ("header", "context", "payload", "tensor_bundle")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    TENSOR_BUNDLE_FIELD_NUMBER: _ClassVar[int]
    header: MessageHeader
    context: ExecutionContext
    payload: ShardPayload
    tensor_bundle: bytes
    def __init__(self, header: _Optional[_Union[MessageHeader, _Mapping]] = ..., context: _Optional[_Union[ExecutionContext, _Mapping]] = ..., payload: _Optional[_Union[ShardPayload, _Mapping]] = ..., tensor_bundle: _Optional[bytes] = ...) -> None: ...

class RuntimeInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RuntimeInfoReply(_message.Message):
    __slots__ = ("model_id", "stage_index", "stage_count", "block_start", "block_end", "include_input_stage", "include_output_stage", "protocol_version")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    STAGE_INDEX_FIELD_NUMBER: _ClassVar[int]
    STAGE_COUNT_FIELD_NUMBER: _ClassVar[int]
    BLOCK_START_FIELD_NUMBER: _ClassVar[int]
    BLOCK_END_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_INPUT_STAGE_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_OUTPUT_STAGE_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    stage_index: int
    stage_count: int
    block_start: int
    block_end: int
    include_input_stage: bool
    include_output_stage: bool
    protocol_version: int
    def __init__(self, model_id: _Optional[str] = ..., stage_index: _Optional[int] = ..., stage_count: _Optional[int] = ..., block_start: _Optional[int] = ..., block_end: _Optional[int] = ..., include_input_stage: _Optional[bool] = ..., include_output_stage: _Optional[bool] = ..., protocol_version: _Optional[int] = ...) -> None: ...

class CreateSessionRequest(_message.Message):
    __slots__ = ("execution_id", "session_id")
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    execution_id: str
    session_id: str
    def __init__(self, execution_id: _Optional[str] = ..., session_id: _Optional[str] = ...) -> None: ...

class CreateSessionReply(_message.Message):
    __slots__ = ("ok", "detail")
    OK_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    detail: str
    def __init__(self, ok: _Optional[bool] = ..., detail: _Optional[str] = ...) -> None: ...

class CloseSessionRequest(_message.Message):
    __slots__ = ("execution_id", "session_id")
    EXECUTION_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    execution_id: str
    session_id: str
    def __init__(self, execution_id: _Optional[str] = ..., session_id: _Optional[str] = ...) -> None: ...

class CloseSessionReply(_message.Message):
    __slots__ = ("ok", "detail")
    OK_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    detail: str
    def __init__(self, ok: _Optional[bool] = ..., detail: _Optional[str] = ...) -> None: ...
