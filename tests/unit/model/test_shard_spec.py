"""ShardSpec tests: static shard description without deployment info (spec 8.4)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from edgeshard.model.spec import BlockRange, ShardSpec


def _spec(**overrides: object) -> ShardSpec:
    values: dict[str, object] = {"model_id": "tiny-llama", "blocks": BlockRange(0, 4)}
    values.update(overrides)
    return ShardSpec.model_validate(values)


def test_defaults() -> None:
    spec = _spec()
    assert spec.include_input_stage is False
    assert spec.include_output_stage is False


def test_nested_block_range_from_mapping() -> None:
    spec = ShardSpec.model_validate({"model_id": "m", "blocks": {"start": 2, "end": 8}})
    assert spec.blocks == BlockRange(2, 8)


def test_invalid_nested_block_range_rejected() -> None:
    with pytest.raises(ValidationError):
        ShardSpec.model_validate({"model_id": "m", "blocks": {"start": 4, "end": 4}})


@pytest.mark.parametrize("field", ["worker_id", "device_id", "container_id", "endpoint"])
def test_deployment_fields_forbidden(field: str) -> None:
    """ShardSpec must not accept scheduling/deployment information (spec 8.4)."""
    with pytest.raises(ValidationError):
        _spec(**{field: "x"})


def test_frozen() -> None:
    spec = _spec()
    with pytest.raises(ValidationError):
        spec.model_id = "other"  # type: ignore[misc]


def test_validate_bounds_accepts_exact_fit() -> None:
    _spec(blocks=BlockRange(2, 4)).validate_bounds(num_blocks=4)


def test_validate_bounds_rejects_overflow() -> None:
    spec = _spec(blocks=BlockRange(2, 6))
    with pytest.raises(ValueError, match="exceeds model block count"):
        spec.validate_bounds(num_blocks=4)
