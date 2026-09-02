"""The shard execution unit (spec 12).

A :class:`ShardModule` owns one contiguous block range of one model plus the
sessions that run on it. It knows nothing about Master, Worker, Docker, host
IPs, endpoints, or scheduling (spec 4.2): callers supply tensors and receive
canonical shard outputs.

Device and dtype are construction parameters only — all execution derives
devices/dtypes from the module and its inputs, so the same code serves the
CPU development tier and GPU containers.
"""

from __future__ import annotations

import torch

from edgeshard.inference.session import SessionError, ShardSession
from edgeshard.inference.state import (
    ExecutionContext,
    InferencePhase,
    LogitsMode,
    LogitsOutput,
    ShardState,
)
from edgeshard.model.adapters.base import ModelAdapter
from edgeshard.model.adapters.registry import default_registry, resolve_adapter_for_source
from edgeshard.model.layout import ModelLayout
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import ShardSpec
from edgeshard.model.weights.safetensors import SafetensorsWeightLoader

ShardResult = ShardState | LogitsOutput
"""Non-final shards emit hidden states; the final shard emits logits."""


class ShardModule:
    """One shard's weights, computation, and sessions.

    Sessions and their KV caches are owned here (spec 13): the cache object
    is created by the adapter and never leaves this module.
    """

    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        module: torch.nn.Module,
        layout: ModelLayout,
        shard: ShardSpec,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._adapter = adapter
        self._module = module
        self._layout = layout
        self._shard = shard
        self._device = device
        self._dtype = dtype
        self._sessions: dict[str, ShardSession] = {}

    @classmethod
    def build(
        cls,
        *,
        source: ModelSource,
        shard: ShardSpec,
        adapter: ModelAdapter | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> ShardModule:
        """Inspect, skeletonize, load, and place one shard without the full weights."""
        if adapter is None:
            adapter = resolve_adapter_for_source(source, default_registry())
        layout = adapter.inspect(source)
        shard.validate_bounds(layout.num_blocks)
        module = adapter.build_skeleton(source, shard)
        SafetensorsWeightLoader().load_shard(module, source, layout, shard)
        device_obj = torch.device(device)
        module.to(device=device_obj, dtype=dtype)
        adapter.restore_high_precision_buffers(module)
        module.eval()
        return cls(
            adapter=adapter,
            module=module,
            layout=layout,
            shard=shard,
            device=device_obj,
            dtype=dtype,
        )

    @property
    def layout(self) -> ModelLayout:
        return self._layout

    @property
    def shard_spec(self) -> ShardSpec:
        return self._shard

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def create_session(self, session_id: str) -> ShardSession:
        """Open a fresh generation session with its own native KV cache."""
        if session_id in self._sessions:
            raise SessionError(f"session already exists: {session_id!r}")
        session = ShardSession(session_id=session_id, kv_cache=self._adapter.new_cache())
        self._sessions[session_id] = session
        return session

    def close_session(self, session_id: str) -> None:
        """Release a session and its KV cache. Unknown sessions fail explicitly."""
        if session_id not in self._sessions:
            raise SessionError(f"no such session: {session_id!r}")
        del self._sessions[session_id]

    def session(self, session_id: str) -> ShardSession:
        if session_id not in self._sessions:
            raise SessionError(f"no such session: {session_id!r}")
        return self._sessions[session_id]

    def prefill(
        self,
        session_id: str,
        *,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        logits_mode: LogitsMode = LogitsMode.FULL,
    ) -> ShardResult:
        """Run the first step on a prompt.

        Input shards consume ``input_ids``; middle/last shards consume
        ``hidden_states`` from the preceding shard. Exactly one must match
        the shard's role. Only the output shard acts on ``logits_mode``;
        middle shards forward the request unchanged.
        """
        session = self.session(session_id)
        if session.step != 0:
            raise SessionError(
                f"prefill on session {session_id!r} after step {session.step}"
            )
        hidden = self._enter(session, input_ids=input_ids, hidden_states=hidden_states)
        return self._advance(session, hidden, InferencePhase.PREFILL, logits_mode)

    def decode(
        self,
        session_id: str,
        *,
        token_id: int | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> ShardResult:
        """Run one autoregressive step.

        Input shards consume the sampled ``token_id``; middle/last shards
        consume ``hidden_states`` from the preceding shard.
        """
        session = self.session(session_id)
        if session.step == 0:
            raise SessionError(f"decode before prefill on session {session_id!r}")
        if self._shard.include_input_stage:
            if token_id is None or hidden_states is not None:
                raise ValueError("input shard decode expects exactly one token_id")
            input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self._device)
            hidden = self._adapter.embed_tokens(self._module, input_ids)
        else:
            if hidden_states is None or token_id is not None:
                raise ValueError("middle/last shard decode expects exactly one hidden_states")
            hidden = hidden_states.to(self._device)
        return self._advance(session, hidden, InferencePhase.DECODE)

    def _enter(
        self,
        session: ShardSession,
        *,
        input_ids: torch.Tensor | None,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._shard.include_input_stage:
            if input_ids is None or hidden_states is not None:
                raise ValueError("input shard prefill expects exactly one input_ids")
            return self._adapter.embed_tokens(
                self._module, input_ids.to(self._device)
            )
        if hidden_states is None or input_ids is not None:
            raise ValueError("middle/last shard prefill expects exactly one hidden_states")
        return hidden_states.to(self._device)

    def _advance(
        self,
        session: ShardSession,
        hidden: torch.Tensor,
        phase: InferencePhase,
        logits_mode: LogitsMode = LogitsMode.FULL,
    ) -> ShardResult:
        q_len = int(hidden.shape[1])
        past_length = session.sequence_length
        if q_len == 1:
            positions = torch.tensor(
                [[past_length]], dtype=torch.long, device=self._device
            )
        else:
            positions = torch.arange(
                past_length, past_length + q_len, dtype=torch.long, device=self._device
            ).unsqueeze(0)

        with torch.no_grad():
            hidden = self._adapter.forward_blocks(
                self._module, hidden, positions, session.kv_cache
            )

        session.sequence_length += q_len
        session.step += 1
        context = ExecutionContext(
            phase=phase,
            step=session.step,
            batch_size=int(hidden.shape[0]),
            sequence_lengths=(session.sequence_length,),
            past_length=past_length,
            positions=positions,
        )
        if self._shard.include_output_stage:
            if logits_mode is LogitsMode.LAST_TOKEN and hidden.shape[1] > 1:
                # Generation consumes only the final position, so trim the
                # hidden states BEFORE the output stage (norm + LM head):
                # the head projects one position, not the whole context,
                # keeping prefill replies at [batch, 1, vocab] regardless
                # of prompt length. This is not a transport-layer slice.
                hidden = hidden[:, -1:, :]
            with torch.no_grad():
                logits = self._adapter.finalize(self._module, hidden)
            return LogitsOutput(logits=logits, context=context)
        return ShardState(hidden_states=hidden, context=context)
