"""Error hierarchy for the EdgeShard model layer.

Every error raised by model discovery, adaptation, and weight loading derives
from :class:`EdgeShardError` so callers can catch one base type. Unsupported
models must fail explicitly; silent guessing is forbidden (spec 9.4, rule 11).
"""

from __future__ import annotations


class EdgeShardError(Exception):
    """Base error for all EdgeShard failures."""


class ModelError(EdgeShardError):
    """Base error for the model layer: sources, layouts, adapters, weights."""


class ModelSourceError(ModelError):
    """The model source is missing, unreadable, or invalid."""


class UnsupportedArchitectureError(ModelError):
    """No adapter can handle this model architecture."""


class WeightError(ModelError):
    """Base error for checkpoint weight handling."""


class MissingWeightsError(WeightError):
    """Tensors required for a shard are absent from the checkpoint."""
