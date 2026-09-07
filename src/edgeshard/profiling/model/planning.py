"""Pure model planning policy (spec §21-§22).

The torch-free planning facts shared by both sides of the control plane:
the Master-side strategy composes sparse TransformerLayer cases from the
representative positions (§47 step 6), and the Worker-side §22 sanity
check interprets the measurements taken at exactly those positions. One
module owns the policy so planner and checker can never drift apart.
"""

from __future__ import annotations

from edgeshard.profiling.domain.signature import LayerPosition

DEFAULT_PREFILL_SEQUENCE_LENGTHS = (128, 512, 2048)
"""Default v1 prefill workload dimensions (§21); batch_size = 1."""


def representative_layer_positions(
    num_layers: int,
) -> tuple[tuple[LayerPosition, int], ...]:
    """Positions to sample for the §22 check: first, middle, last.

    For a 32-layer stack this yields indices 0/16/31 (the spec example
    counts 1-based). Stacks smaller than three layers collapse onto the
    positions that exist; duplicates are removed, order preserved.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    candidates = [(LayerPosition.EARLY, 0)]
    if num_layers >= 3:
        candidates.append((LayerPosition.MIDDLE, num_layers // 2))
    if num_layers >= 2:
        candidates.append((LayerPosition.LATE, num_layers - 1))
    seen: set[int] = set()
    positions: list[tuple[LayerPosition, int]] = []
    for position, index in candidates:
        if index in seen:
            continue
        seen.add(index)
        positions.append((position, index))
    return tuple(positions)


__all__ = ["DEFAULT_PREFILL_SEQUENCE_LENGTHS", "representative_layer_positions"]
