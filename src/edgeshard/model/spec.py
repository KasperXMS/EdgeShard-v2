"""Shard specifications built on contiguous Transformer block ranges.

All block ranges use Python half-open interval semantics (spec 8.3):
``BlockRange(8, 20)`` means blocks 8 through 19. Inclusive ranges must never
be introduced elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True)
class BlockRange:
    """Half-open block index interval ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("invalid block range")

    def __len__(self) -> int:
        return self.end - self.start

    def __contains__(self, index: object) -> bool:
        return isinstance(index, int) and self.start <= index < self.end


class ShardSpec(BaseModel):
    """Static description of which model part a shard executes (spec 8.4).

    Deliberately contains no worker id, device id, container id, network
    endpoint, or scheduling information; unknown fields are rejected.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    blocks: BlockRange
    include_input_stage: bool = False
    include_output_stage: bool = False

    def validate_bounds(self, num_blocks: int) -> None:
        """Check that this shard's block range fits a model of ``num_blocks`` blocks."""
        if self.blocks.end > num_blocks:
            raise ValueError(
                f"shard block range [{self.blocks.start}, {self.blocks.end}) exceeds "
                f"model block count {num_blocks}"
            )
