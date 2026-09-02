"""EdgeShard shard runtime server (0F).

One process = one shard stage. The handler validates inbound messages
(spec 16.2), executes its blocks locally, and either replies (final stage)
or forwards downstream (spec 18): the Mock Master never relays hidden
states. ``CreateSession``/``CloseSession`` propagate through the pipeline
(spec 18.1); each runtime creates its own local KV state.

Computation is synchronous and blocks the event loop deliberately: Phase 0
is a correctness transport, not a performance commitment (spec 5.4).
"""

from __future__ import annotations

import asyncio
import time
import uuid

import grpc
import torch

from edgeshard.inference.shard import ShardModule
from edgeshard.inference.state import LogitsOutput, ShardState
from edgeshard.model.errors import EdgeShardError
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import ShardSpec
from edgeshard.protocol.domain import (
    MASTER_STAGE,
    PROTOCOL_VERSION,
    HiddenStatePayload,
    LogitsPayload,
    MessageExpectation,
    ProtocolError,
    ShardMessage,
    ShardMessageHeader,
    TokenPayload,
    validate_forward_message,
)
from edgeshard.protocol.grpc_client import ShardRuntimeClient
from edgeshard.protocol.grpc_server import start_shard_runtime_server
from edgeshard.runtime.config import ShardRuntimeConfig
from edgeshard.runtime.info import runtime_info_from_config, runtime_info_to_wire

DOWNSTREAM_READY_TIMEOUT_S = 300.0
"""Default time a stage waits for its downstream stage before serving."""

DOWNSTREAM_POLL_INTERVAL_S = 0.25


class RuntimeStartupError(EdgeShardError):
    """The shard runtime could not reach a serving state."""


class ShardRuntimeHandler:
    """Session/forward logic of one shard stage."""

    def __init__(
        self,
        *,
        config: ShardRuntimeConfig,
        module: ShardModule,
        downstream: ShardRuntimeClient | None,
    ) -> None:
        if config.is_final_stage and downstream is not None:
            raise ValueError("the final stage must not have a downstream client")
        if not config.is_final_stage and downstream is None:
            raise ValueError("non-final stages require a downstream client")
        self._config = config
        self._module = module
        self._downstream = downstream

    @property
    def stage_index(self) -> int:
        return self._config.pipeline.stage_index

    async def create_session(self, execution_id: str, session_id: str) -> None:
        self._check_execution_id(execution_id)
        self._module.create_session(session_id)
        if self._downstream is not None:
            try:
                await self._downstream.create_session(execution_id, session_id)
            except Exception:
                self._module.close_session(session_id)
                raise

    async def close_session(self, execution_id: str, session_id: str) -> None:
        self._check_execution_id(execution_id)
        self._module.close_session(session_id)
        if self._downstream is not None:
            await self._downstream.close_session(execution_id, session_id)

    async def prefill(self, message: ShardMessage) -> ShardMessage:
        return await self._execute(message, decode=False)

    async def decode(self, message: ShardMessage) -> ShardMessage:
        return await self._execute(message, decode=True)

    async def _execute(self, message: ShardMessage, *, decode: bool) -> ShardMessage:
        header = message.header
        session = self._module.session(header.session_id)
        validate_forward_message(
            message,
            MessageExpectation(
                execution_id=self._config.runtime.execution_id,
                stage_index=self.stage_index,
                next_step=session.step,
                session_open=True,
            ),
        )
        output = self._run_local(message, decode=decode)
        return await self._emit(message, output, decode=decode)

    def _run_local(self, message: ShardMessage, *, decode: bool) -> ShardState | LogitsOutput:
        payload = message.payload
        session_id = message.header.session_id
        if self._module.shard_spec.include_input_stage:
            if not isinstance(payload, TokenPayload):
                raise ProtocolError("input stage expects a token payload")
            if decode:
                if len(payload.token_ids) != 1:
                    raise ProtocolError("decode expects exactly one token")
                return self._module.decode(session_id, token_id=payload.token_ids[0])
            input_ids = torch.tensor([list(payload.token_ids)], dtype=torch.long)
            return self._module.prefill(
                session_id,
                input_ids=input_ids,
                logits_mode=message.header.logits_mode,
            )
        if not isinstance(payload, HiddenStatePayload):
            raise ProtocolError("non-input stage expects a hidden-state payload")
        if decode:
            return self._module.decode(session_id, hidden_states=payload.hidden_states)
        return self._module.prefill(
            session_id,
            hidden_states=payload.hidden_states,
            logits_mode=message.header.logits_mode,
        )

    async def _emit(
        self,
        message: ShardMessage,
        output: ShardState | LogitsOutput,
        *,
        decode: bool,
    ) -> ShardMessage:
        header = message.header
        if self._config.is_final_stage:
            if not isinstance(output, LogitsOutput):
                raise ProtocolError("final stage produced no logits")
            reply_header = ShardMessageHeader(
                protocol_version=PROTOCOL_VERSION,
                execution_id=header.execution_id,
                session_id=header.session_id,
                request_id=uuid.uuid4().hex,
                phase=header.phase,
                step=header.step,
                source_stage=self.stage_index,
                target_stage=MASTER_STAGE,
                logits_mode=header.logits_mode,
            )
            return ShardMessage(
                header=reply_header,
                context=output.context,
                payload=LogitsPayload(logits=output.logits),
            )
        if not isinstance(output, ShardState):
            raise ProtocolError("non-final stage must not produce logits")
        assert self._downstream is not None
        next_header = ShardMessageHeader(
            protocol_version=PROTOCOL_VERSION,
            execution_id=header.execution_id,
            session_id=header.session_id,
            request_id=uuid.uuid4().hex,
            phase=header.phase,
            step=header.step,
            source_stage=self.stage_index,
            target_stage=self.stage_index + 1,
            logits_mode=header.logits_mode,
        )
        next_message = ShardMessage(
            header=next_header,
            context=output.context,
            payload=HiddenStatePayload(hidden_states=output.hidden_states),
        )
        if decode:
            return await self._downstream.decode(next_message)
        return await self._downstream.prefill(next_message)

    def _check_execution_id(self, execution_id: str) -> None:
        expected = self._config.runtime.execution_id
        if execution_id != expected:
            raise ProtocolError(
                f"wrong execution ID: got {execution_id!r}, expected {expected!r}"
            )


def build_shard_module(config: ShardRuntimeConfig) -> ShardModule:
    """Build this process's shard from config without the full model weights."""
    source = ModelSource(path=config.model.path, model_id=config.model.id)
    spec = ShardSpec(
        model_id=config.model.id,
        blocks=config.shard.blocks(),
        include_input_stage=config.shard.include_input_stage,
        include_output_stage=config.shard.include_output_stage,
    )
    return ShardModule.build(
        source=source,
        shard=spec,
        dtype=config.inference.torch_dtype(),
        device=config.device.torch_device(),
    )


async def _wait_for_downstream_ready(
    downstream: ShardRuntimeClient,
    *,
    runtime_id: str,
    timeout_s: float,
    poll_interval_s: float = DOWNSTREAM_POLL_INTERVAL_S,
) -> None:
    """Poll the downstream stage's GetRuntimeInfo until it answers."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            await downstream.get_runtime_info()
            return
        except grpc.aio.AioRpcError:
            if time.monotonic() > deadline:
                raise RuntimeStartupError(
                    f"runtime {runtime_id!r}: downstream stage at "
                    f"{downstream.endpoint} not ready within {timeout_s}s"
                ) from None
            await asyncio.sleep(poll_interval_s)


class ShardRuntimeServer:
    """One running shard stage: module + handler + gRPC server."""

    def __init__(
        self,
        *,
        config: ShardRuntimeConfig,
        grpc_server: grpc.aio.Server,
        port: int,
        downstream: ShardRuntimeClient | None,
    ) -> None:
        self._config = config
        self._grpc_server = grpc_server
        self._port = port
        self._downstream = downstream

    @classmethod
    async def create(
        cls,
        config: ShardRuntimeConfig,
        *,
        downstream_ready_timeout_s: float = DOWNSTREAM_READY_TIMEOUT_S,
    ) -> ShardRuntimeServer:
        module = build_shard_module(config)
        downstream = (
            ShardRuntimeClient(config.pipeline.next_endpoint)
            if config.pipeline.next_endpoint
            else None
        )
        handler = ShardRuntimeHandler(config=config, module=module, downstream=downstream)
        if downstream is not None:
            # Readiness is transitive (spec 23): only the entry runtime is
            # reachable from the host, so a stage must not serve until its
            # whole downstream chain answers GetRuntimeInfo.
            await _wait_for_downstream_ready(
                downstream,
                runtime_id=config.runtime.runtime_id,
                timeout_s=downstream_ready_timeout_s,
            )
        info_wire = runtime_info_to_wire(runtime_info_from_config(config))
        grpc_server, port = await start_shard_runtime_server(
            handler,
            info_wire,
            host=config.server.listen_host,
            port=config.server.listen_port,
        )
        return cls(config=config, grpc_server=grpc_server, port=port, downstream=downstream)

    @property
    def config(self) -> ShardRuntimeConfig:
        return self._config

    @property
    def port(self) -> int:
        return self._port

    @property
    def endpoint(self) -> str:
        return f"{self._config.server.listen_host}:{self._port}"

    async def wait_for_termination(self) -> None:
        await self._grpc_server.wait_for_termination()

    async def stop(self) -> None:
        await self._grpc_server.stop(None)
        if self._downstream is not None:
            await self._downstream.close()
