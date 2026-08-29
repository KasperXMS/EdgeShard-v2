"""Weight loader interface (spec 11.4)."""

from __future__ import annotations

from typing import Protocol

import torch.nn as nn

from edgeshard.model.layout import ModelLayout
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import ShardSpec


class WeightLoader(Protocol):
    """Materializes exactly one shard's weights into a skeleton module."""

    def load_shard(
        self,
        module: nn.Module,
        source: ModelSource,
        layout: ModelLayout,
        shard: ShardSpec,
    ) -> None:
        """Load the shard's tensors into ``module`` in place."""
        ...
