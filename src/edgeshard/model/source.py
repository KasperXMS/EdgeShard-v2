"""Model source description.

Phase 0 models always live in a locally materialized, read-only snapshot
directory (spec 21.1, 24). Runtime containers never download models
themselves; snapshots are prepared externally (e.g. via
``huggingface_hub.snapshot_download``) and mounted at ``/models:ro``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from edgeshard.model.errors import ModelSourceError


class ModelSource(BaseModel):
    """A locally materialized model snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    """Local snapshot directory (mounted read-only into containers)."""

    model_id: str | None = None
    """Hugging Face repository id for labeling, e.g. ``Qwen/Qwen2.5-7B``."""

    revision: str | None = None
    """Resolved immutable commit SHA, for reproducibility (spec 24)."""

    def ensure_local(self) -> Path:
        """Return the snapshot path, failing explicitly if it is not usable."""
        if not self.path.is_dir():
            raise ModelSourceError(f"model source path is not a directory: {self.path}")
        return self.path
