"""ShardRuntimeHandler unit tests: validation, rollback, forwarding (spec 18).

The handler is exercised with duck-typed stubs (mypy only checks src/):
the logic under test is orchestration, not computation.
"""

from __future__ import annotations

import copy
import uuid

import pytest
import torch

from edgeshard.inference.session import SessionError, ShardSession
from edgeshard.inference.state import (
    ExecutionContext,
    InferencePhase,
    LogitsOutput,
    ShardState,
)
from edgeshard.protocol.domain import (
    MASTER_STAGE,
    PROTOCOL_VERSION,
    HiddenStatePayload,
    LogitsPayload,
    ProtocolError,
    ShardMessage,
    ShardMessageHeader,
    TokenPayload,
)
from edgeshard.runtime.config import ShardRuntimeConfig
from edgeshard.runtime.shard_server import ShardRuntimeHandler

EXECUTION_ID = "exec-handler"
SESSION_ID = "s"

CONFIG_PAYLOAD: dict = {
    "runtime": {
        "backend": "edgeshard_shard",
        "runtime_id": "r",
        "execution_id": EXECUTION_ID,
    },
    "model": {"id": "tiny/llama", "path": "/tmp/tiny-llama"},
    "shard": {
        "start_block": 1,
        "end_block": 3,
        "include_input_stage": False,
        "include_output_stage": False,
    },
    "pipeline": {"stage_index": 1, "stage_count": 3, "next_endpoint": "127.0.0.1:1"},
    "server": {"listen_port": 9001},
}


def make_config(stage_index: int = 1, **shard_overrides: object) -> ShardRuntimeConfig:
    payload = copy.deepcopy(CONFIG_PAYLOAD)
    payload["shard"].update(shard_overrides)
    if stage_index == 0:
        payload["shard"]["start_block"] = 0
        payload["shard"]["end_block"] = 1
        payload["shard"]["include_input_stage"] = True
    payload["pipeline"]["stage_index"] = stage_index
    return ShardRuntimeConfig.model_validate(payload)


def make_final_config() -> ShardRuntimeConfig:
    payload = copy.deepcopy(CONFIG_PAYLOAD)
    payload["shard"]["start_block"] = 3
    payload["shard"]["end_block"] = 4
    payload["shard"]["include_output_stage"] = True
    payload["pipeline"] = {"stage_index": 2, "stage_count": 3}
    return ShardRuntimeConfig.model_validate(payload)


class StubSpec:
    def __init__(self, *, include_input_stage: bool, include_output_stage: bool) -> None:
        self.include_input_stage = include_input_stage
        self.include_output_stage = include_output_stage


class StubModule:
    """Duck-typed ShardModule recording session operations."""

    def __init__(self, *, include_input_stage: bool, include_output_stage: bool) -> None:
        self.shard_spec = StubSpec(
            include_input_stage=include_input_stage,
            include_output_stage=include_output_stage,
        )
        self.sessions: dict[str, ShardSession] = {}
        self.executed: list[str] = []

    def create_session(self, session_id: str) -> ShardSession:
        if session_id in self.sessions:
            raise SessionError(f"session already exists: {session_id!r}")
        session = ShardSession(session_id=session_id, kv_cache=object())
        self.sessions[session_id] = session
        return session

    def close_session(self, session_id: str) -> None:
        if session_id not in self.sessions:
            raise SessionError(f"no such session: {session_id!r}")
        del self.sessions[session_id]

    def session(self, session_id: str) -> ShardSession:
        if session_id not in self.sessions:
            raise SessionError(f"no such session: {session_id!r}")
        return self.sessions[session_id]

    def _output(self, phase: InferencePhase) -> ShardState | LogitsOutput:
        context = ExecutionContext(
            phase=phase,
            step=self.sessions[SESSION_ID].step + 1,
            batch_size=1,
            sequence_lengths=(1,),
            past_length=0,
            positions=None,
        )
        if self.shard_spec.include_output_stage:
            return LogitsOutput(logits=torch.zeros(1, 1, 8), context=context)
        return ShardState(hidden_states=torch.zeros(1, 1, 4), context=context)

    def prefill(self, session_id: str, **kwargs: object) -> ShardState | LogitsOutput:
        self.executed.append("prefill")
        self.sessions[session_id].step += 1
        return self._output(InferencePhase.PREFILL)

    def decode(self, session_id: str, **kwargs: object) -> ShardState | LogitsOutput:
        self.executed.append("decode")
        self.sessions[session_id].step += 1
        return self._output(InferencePhase.DECODE)


class StubDownstream:
    def __init__(self, *, fail_create: bool = False) -> None:
        self.fail_create = fail_create
        self.created: list[tuple[str, str]] = []
        self.closed: list[tuple[str, str]] = []
        self.forwarded: list[ShardMessage] = []

    async def create_session(self, execution_id: str, session_id: str) -> None:
        if self.fail_create:
            raise ProtocolError("downstream refused")
        self.created.append((execution_id, session_id))

    async def close_session(self, execution_id: str, session_id: str) -> None:
        self.closed.append((execution_id, session_id))

    async def prefill(self, message: ShardMessage) -> ShardMessage:
        self.forwarded.append(message)
        return _final_reply(message)

    async def decode(self, message: ShardMessage) -> ShardMessage:
        self.forwarded.append(message)
        return _final_reply(message)


def _final_reply(request: ShardMessage) -> ShardMessage:
    header = request.header
    return ShardMessage(
        header=ShardMessageHeader(
            protocol_version=PROTOCOL_VERSION,
            execution_id=header.execution_id,
            session_id=header.session_id,
            request_id=uuid.uuid4().hex,
            phase=header.phase,
            step=header.step,
            source_stage=2,
            target_stage=MASTER_STAGE,
        ),
        context=request.context,
        payload=LogitsPayload(logits=torch.ones(1, 1, 8)),
    )


def make_message(
    *,
    phase: InferencePhase,
    step: int,
    payload: TokenPayload | HiddenStatePayload,
    source_stage: int,
    target_stage: int,
    execution_id: str = EXECUTION_ID,
) -> ShardMessage:
    return ShardMessage(
        header=ShardMessageHeader(
            protocol_version=PROTOCOL_VERSION,
            execution_id=execution_id,
            session_id=SESSION_ID,
            request_id=uuid.uuid4().hex,
            phase=phase,
            step=step,
            source_stage=source_stage,
            target_stage=target_stage,
        ),
        context=ExecutionContext(
            phase=phase,
            step=step,
            batch_size=1,
            sequence_lengths=(1,),
            past_length=0,
            positions=None,
        ),
        payload=payload,
    )


def middle_handler() -> tuple[ShardRuntimeHandler, StubModule, StubDownstream]:
    module = StubModule(include_input_stage=False, include_output_stage=False)
    downstream = StubDownstream()
    handler = ShardRuntimeHandler(
        config=make_config(), module=module, downstream=downstream  # type: ignore[arg-type]
    )
    return handler, module, downstream


def input_handler() -> tuple[ShardRuntimeHandler, StubModule, StubDownstream]:
    module = StubModule(include_input_stage=True, include_output_stage=False)
    downstream = StubDownstream()
    handler = ShardRuntimeHandler(
        config=make_config(stage_index=0), module=module, downstream=downstream  # type: ignore[arg-type]
    )
    return handler, module, downstream


async def test_create_session_propagates_downstream() -> None:
    handler, module, downstream = middle_handler()
    await handler.create_session(EXECUTION_ID, SESSION_ID)
    assert SESSION_ID in module.sessions
    assert downstream.created == [(EXECUTION_ID, SESSION_ID)]


async def test_create_session_rolls_back_when_downstream_fails() -> None:
    module = StubModule(include_input_stage=False, include_output_stage=False)
    downstream = StubDownstream(fail_create=True)
    handler = ShardRuntimeHandler(
        config=make_config(), module=module, downstream=downstream  # type: ignore[arg-type]
    )
    with pytest.raises(ProtocolError, match="downstream refused"):
        await handler.create_session(EXECUTION_ID, SESSION_ID)
    assert SESSION_ID not in module.sessions


async def test_session_rpcs_reject_wrong_execution_id() -> None:
    handler, module, downstream = middle_handler()
    with pytest.raises(ProtocolError, match="execution ID"):
        await handler.create_session("exec-other", SESSION_ID)
    assert module.sessions == {}
    with pytest.raises(ProtocolError, match="execution ID"):
        await handler.close_session("exec-other", SESSION_ID)
    assert downstream.created == []


async def test_close_session_propagates_downstream() -> None:
    handler, module, downstream = middle_handler()
    await handler.create_session(EXECUTION_ID, SESSION_ID)
    await handler.close_session(EXECUTION_ID, SESSION_ID)
    assert module.sessions == {}
    assert downstream.closed == [(EXECUTION_ID, SESSION_ID)]


async def test_input_stage_requires_token_payload() -> None:
    handler, _, _ = input_handler()
    await handler.create_session(EXECUTION_ID, SESSION_ID)
    hidden = make_message(
        phase=InferencePhase.PREFILL,
        step=0,
        payload=HiddenStatePayload(hidden_states=torch.zeros(1, 1, 4)),
        source_stage=MASTER_STAGE,
        target_stage=0,
    )
    with pytest.raises(ProtocolError, match="token payload"):
        await handler.prefill(hidden)


async def test_input_stage_decode_requires_single_token() -> None:
    handler, _, _ = input_handler()
    await handler.create_session(EXECUTION_ID, SESSION_ID)
    prefill = make_message(
        phase=InferencePhase.PREFILL,
        step=0,
        payload=TokenPayload(token_ids=(1, 2)),
        source_stage=MASTER_STAGE,
        target_stage=0,
    )
    await handler.prefill(prefill)
    decode = make_message(
        phase=InferencePhase.DECODE,
        step=1,
        payload=TokenPayload(token_ids=(3, 4)),
        source_stage=MASTER_STAGE,
        target_stage=0,
    )
    with pytest.raises(ProtocolError, match="exactly one token"):
        await handler.decode(decode)


async def test_middle_stage_requires_hidden_states() -> None:
    handler, _, _ = middle_handler()
    await handler.create_session(EXECUTION_ID, SESSION_ID)
    tokens = make_message(
        phase=InferencePhase.PREFILL,
        step=0,
        payload=TokenPayload(token_ids=(1,)),
        source_stage=0,
        target_stage=1,
    )
    with pytest.raises(ProtocolError, match="hidden-state payload"):
        await handler.prefill(tokens)


async def test_middle_stage_forwards_and_returns_downstream_reply() -> None:
    handler, module, downstream = middle_handler()
    await handler.create_session(EXECUTION_ID, SESSION_ID)
    message = make_message(
        phase=InferencePhase.PREFILL,
        step=0,
        payload=HiddenStatePayload(hidden_states=torch.zeros(1, 2, 4)),
        source_stage=0,
        target_stage=1,
    )
    reply = await handler.prefill(message)

    assert module.executed == ["prefill"]
    (forwarded,) = downstream.forwarded
    assert forwarded.header.source_stage == 1
    assert forwarded.header.target_stage == 2
    assert forwarded.header.step == 0
    assert forwarded.header.execution_id == EXECUTION_ID
    assert isinstance(forwarded.payload, HiddenStatePayload)
    assert forwarded.payload.hidden_states.shape == (1, 1, 4)

    assert reply.header.source_stage == 2
    assert reply.header.target_stage == MASTER_STAGE
    assert isinstance(reply.payload, LogitsPayload)


async def test_final_stage_replies_with_logits() -> None:
    module = StubModule(include_input_stage=False, include_output_stage=True)
    handler = ShardRuntimeHandler(
        config=make_final_config(), module=module, downstream=None  # type: ignore[arg-type]
    )
    await handler.create_session(EXECUTION_ID, SESSION_ID)
    message = make_message(
        phase=InferencePhase.PREFILL,
        step=0,
        payload=HiddenStatePayload(hidden_states=torch.zeros(1, 2, 4)),
        source_stage=1,
        target_stage=2,
    )
    reply = await handler.prefill(message)
    assert reply.header.source_stage == 2
    assert reply.header.target_stage == MASTER_STAGE
    assert reply.header.step == 0
    assert isinstance(reply.payload, LogitsPayload)
    assert reply.payload.logits.shape == (1, 1, 8)
