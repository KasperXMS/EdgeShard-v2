"""Per-generation session state with locally owned KV cache (spec 12.3, 13).

The KV cache belongs to the runtime that owns the corresponding Transformer
blocks and never crosses shard boundaries: each session carries its own
native cache object, created by the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

from edgeshard.model.errors import EdgeShardError


class SessionError(EdgeShardError):
    """Session lifecycle violation (unknown, duplicate, or closed session)."""


@dataclass
class ShardSession:
    """Per-request generation state for one shard."""

    session_id: str
    kv_cache: object
    """Native cache object owned by this shard (spec 13); opaque by design."""

    sequence_length: int = 0
    """Number of tokens processed so far (per sequence; batch_size = 1)."""

    step: int = 0
    """Number of forward steps executed (prefill + decodes)."""
