"""BlockRange tests: half-open interval semantics (spec 8.3)."""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given
from hypothesis import strategies as st

from edgeshard.model.spec import BlockRange


def test_half_open_membership() -> None:
    blocks = BlockRange(8, 20)
    assert 8 in blocks
    assert 19 in blocks
    assert 20 not in blocks
    assert 7 not in blocks
    assert len(blocks) == 12


def test_non_int_membership() -> None:
    blocks = BlockRange(0, 4)
    assert 1.0 not in blocks  # type: ignore[comparison-overlap]
    assert "0" not in blocks  # type: ignore[comparison-overlap]


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (-1, 4),
        (0, 0),
        (4, 4),
        (5, 4),
    ],
)
def test_invalid_ranges_rejected(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="invalid block range"):
        BlockRange(start, end)


def test_frozen() -> None:
    blocks = BlockRange(0, 4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        blocks.start = 1  # type: ignore[misc]


def test_equality_and_hashing() -> None:
    assert BlockRange(0, 4) == BlockRange(0, 4)
    assert BlockRange(0, 4) != BlockRange(0, 5)
    assert hash(BlockRange(0, 4)) == hash(BlockRange(0, 4))
    assert {BlockRange(0, 4), BlockRange(0, 4)} == {BlockRange(0, 4)}


@given(
    start=st.integers(min_value=0, max_value=1000),
    length=st.integers(min_value=1, max_value=1000),
)
def test_valid_ranges_always_construct(start: int, length: int) -> None:
    blocks = BlockRange(start, start + length)
    assert len(blocks) == length
    assert start in blocks
    assert start + length not in blocks
