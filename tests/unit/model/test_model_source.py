"""ModelSource tests: local snapshot semantics (spec 21.1, 24)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from edgeshard.model.errors import ModelSourceError
from edgeshard.model.source import ModelSource


def test_defaults() -> None:
    source = ModelSource(path=Path("/models/tiny"))
    assert source.model_id is None
    assert source.revision is None


def test_ensure_local_returns_existing_directory(tmp_path: Path) -> None:
    source = ModelSource(path=tmp_path, model_id="tiny/qwen", revision="abc123")
    assert source.ensure_local() == tmp_path


def test_ensure_local_rejects_missing_directory(tmp_path: Path) -> None:
    source = ModelSource(path=tmp_path / "does-not-exist")
    with pytest.raises(ModelSourceError, match="not a directory"):
        source.ensure_local()


def test_ensure_local_rejects_file(tmp_path: Path) -> None:
    file_path = tmp_path / "config.json"
    file_path.write_text("{}")
    source = ModelSource(path=file_path)
    with pytest.raises(ModelSourceError, match="not a directory"):
        source.ensure_local()


def test_frozen_and_strict() -> None:
    source = ModelSource(path=Path("/models/tiny"))
    with pytest.raises(ValidationError):
        source.model_id = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ModelSource.model_validate({"path": "/models/tiny", "unknown": 1})
