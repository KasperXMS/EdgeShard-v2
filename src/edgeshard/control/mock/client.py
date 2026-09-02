"""Master-side clients for deployed runtimes (spec 22.1).

``RemotePipeline``/``RemoteGenerationDriver`` drive the EdgeShard shard
pipeline through its entry runtime: token hops go in (master -> stage 0),
final logits replies come out, and hidden states never pass through the
master (spec 18). Sampling stays outside the shard runtime, here in the
async gRPC twin of ``inference/generation.py``.

``VLLMClient`` issues plain OpenAI-compatible requests to an independent
vLLM runtime (spec 5.7): the master never reshapes vLLM into a shard
runtime (spec 20.2), and test requests stay deterministic greedy
(temperature 0) like the rest of Phase 0.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import httpx
import torch

from edgeshard.inference.state import (
    ExecutionContext,
    InferencePhase,
    LogitsMode,
    LogitsOutput,
)
from edgeshard.protocol.domain import (
    MASTER_STAGE,
    PROTOCOL_VERSION,
    LogitsPayload,
    ProtocolError,
    ShardMessage,
    ShardMessageHeader,
    TokenPayload,
)
from edgeshard.protocol.grpc_client import ShardRuntimeClient


@dataclass
class _SessionClock:
    """The master's view of one session's progress: next wire step, KV length."""

    step: int
    past_length: int


class RemotePipeline:
    """Pipeline facade over the entry runtime of a deployment.

    Mirrors ``LocalPipeline``'s session/prefill/decode surface, but every
    hop crosses the gRPC boundary. The master tracks each session's wire
    step and past length locally; the runtimes validate them (spec 16.2)
    and reject any drift.
    """

    def __init__(self, *, endpoint: str, execution_id: str) -> None:
        self._client = ShardRuntimeClient(endpoint)
        self._execution_id = execution_id
        self._clocks: dict[str, _SessionClock] = {}

    @property
    def execution_id(self) -> str:
        return self._execution_id

    @property
    def endpoint(self) -> str:
        return self._client.endpoint

    async def __aenter__(self) -> RemotePipeline:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.close()

    async def create_session(self, session_id: str) -> None:
        """Open the session on every stage (chained through the pipeline)."""
        await self._client.create_session(self._execution_id, session_id)

    async def close_session(self, session_id: str) -> None:
        await self._client.close_session(self._execution_id, session_id)
        self._clocks.pop(session_id, None)

    async def prefill(
        self,
        session_id: str,
        input_ids: torch.Tensor,
        *,
        logits_mode: LogitsMode = LogitsMode.FULL,
    ) -> LogitsOutput:
        """Send the prompt as the master -> stage-0 TokenPayload hop.

        ``logits_mode`` rides in the message header and is forwarded stage
        to stage; generation passes ``LAST_TOKEN`` so long-context prefills
        answer with a single-position projection instead of full
        ``[batch, seq, vocab]`` logits.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "Phase 0 is batch_size=1: expected input_ids of shape [1, L], "
                f"got {tuple(input_ids.shape)}"
            )
        token_ids = [int(token) for token in input_ids[0].tolist()]
        message = self._token_message(
            session_id,
            token_ids,
            phase=InferencePhase.PREFILL,
            step=0,
            past_length=0,
            logits_mode=logits_mode,
        )
        reply = await self._client.prefill(message)
        self._clocks[session_id] = _SessionClock(step=1, past_length=len(token_ids))
        return self._as_logits(reply)

    async def decode(self, session_id: str, token_id: int) -> LogitsOutput:
        """Send one sampled token; wire step/past length come from the clock."""
        clock = self._clocks.get(session_id)
        if clock is None:
            raise ValueError(f"session {session_id!r} must be prefilled before decode")
        message = self._token_message(
            session_id,
            [token_id],
            phase=InferencePhase.DECODE,
            step=clock.step,
            past_length=clock.past_length,
        )
        reply = await self._client.decode(message)
        clock.step += 1
        clock.past_length += 1
        return self._as_logits(reply)

    def _token_message(
        self,
        session_id: str,
        token_ids: list[int],
        *,
        phase: InferencePhase,
        step: int,
        past_length: int,
        logits_mode: LogitsMode = LogitsMode.FULL,
    ) -> ShardMessage:
        return ShardMessage(
            header=ShardMessageHeader(
                protocol_version=PROTOCOL_VERSION,
                execution_id=self._execution_id,
                session_id=session_id,
                request_id=uuid.uuid4().hex,
                phase=phase,
                step=step,
                source_stage=MASTER_STAGE,
                target_stage=0,
                logits_mode=logits_mode,
            ),
            context=ExecutionContext(
                phase=phase,
                step=step,
                batch_size=1,
                sequence_lengths=(past_length + len(token_ids),),
                past_length=past_length,
                positions=None,
            ),
            payload=TokenPayload(token_ids=tuple(token_ids)),
        )

    def _as_logits(self, reply: ShardMessage) -> LogitsOutput:
        if not isinstance(reply.payload, LogitsPayload):
            raise ProtocolError("entry runtime reply carries no logits payload")
        return LogitsOutput(logits=reply.payload.logits, context=reply.context)


class RemoteGenerationDriver:
    """Deterministic greedy generation over a deployed pipeline (spec 15.3)."""

    def __init__(self, pipeline: RemotePipeline) -> None:
        self._pipeline = pipeline

    async def generate(
        self, session_id: str, input_ids: torch.Tensor, *, max_new_tokens: int
    ) -> list[int]:
        """Prefill the prompt, then sample ``max_new_tokens`` greedy tokens."""
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        output = await self._pipeline.prefill(session_id, input_ids)
        generated: list[int] = []
        token = int(output.logits[0, -1].argmax())
        generated.append(token)
        for _ in range(max_new_tokens - 1):
            output = await self._pipeline.decode(session_id, token_id=token)
            token = int(output.logits[0, -1].argmax())
            generated.append(token)
        return generated[:max_new_tokens]


@dataclass(frozen=True)
class VLLMCompletion:
    """One completion from vLLM's OpenAI-compatible API."""

    text: str
    finish_reason: str | None


class VLLMClient:
    """OpenAI-compatible test requests to an independent vLLM runtime.

    The model name in requests is the value vLLM serves — the ``--model``
    argument, i.e. the container-side model path for master-deployed
    runtimes.
    """

    def __init__(self, endpoint: str, model: str, *, timeout_s: float = 60.0) -> None:
        self._http = httpx.AsyncClient(base_url=f"http://{endpoint}", timeout=timeout_s)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def __aenter__(self) -> VLLMClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    async def list_models(self) -> list[str]:
        """IDs of the models this runtime serves (``GET /v1/models``)."""
        response = await self._http.get("/v1/models")
        response.raise_for_status()
        payload = response.json()
        return [str(item["id"]) for item in payload["data"]]

    async def complete(
        self, prompt_token_ids: Sequence[int], *, max_tokens: int
    ) -> VLLMCompletion:
        """One deterministic greedy completion (``temperature=0``)."""
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        response = await self._http.post(
            "/v1/completions",
            json={
                "model": self._model,
                "prompt": [int(token) for token in prompt_token_ids],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        return VLLMCompletion(
            text=str(choice["text"]), finish_reason=choice.get("finish_reason")
        )
