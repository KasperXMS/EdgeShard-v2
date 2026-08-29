"""Error hierarchy tests: one catchable base type per layer."""

from __future__ import annotations

from edgeshard.model.errors import (
    EdgeShardError,
    MissingWeightsError,
    ModelError,
    ModelSourceError,
    UnsupportedArchitectureError,
    WeightError,
)


def test_hierarchy() -> None:
    assert issubclass(ModelError, EdgeShardError)
    assert issubclass(ModelSourceError, ModelError)
    assert issubclass(UnsupportedArchitectureError, ModelError)
    assert issubclass(WeightError, ModelError)
    assert issubclass(MissingWeightsError, WeightError)


def test_catchable_at_each_level() -> None:
    for error_cls in (ModelSourceError, UnsupportedArchitectureError, MissingWeightsError):
        try:
            raise error_cls("boom")
        except ModelError as exc:
            assert isinstance(exc, EdgeShardError)
