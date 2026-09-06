"""Canonical hashing guarantees (Phase 2 spec §7).

Every requirement of §7 has an explicit test here: identical semantics →
identical digests, mapping-order irrelevance, stable enum serialization,
volatile-field rejection, and pinned golden digests so the canonicalization
scheme can never drift silently (a drift would orphan every persisted
signature id).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from edgeshard.profiling.domain.hashing import (
    canonical_json,
    canonical_sha256,
    canonical_value,
    check_normalized_items,
    normalized_items,
)


@dataclass(frozen=True)
class _Sample:
    name: str
    values: tuple[int, ...]
    tags: tuple[str, ...] = ()


GOLDEN_TAGGED_MAP_DIGEST = "98c5b31f023deea29f3803610a7434294bb4046af1c6366a1e1e4135cf1ae1c4"


def test_canonical_json_sorts_mapping_keys() -> None:
    assert canonical_json(("tag", {"b": 1, "a": 2})) == '["tag",{"a":2,"b":1}]'


def test_golden_digest_is_pinned() -> None:
    """The scheme is frozen: this digest must never change (§7 stability)."""
    assert canonical_sha256(("tag", {"b": 1, "a": 2})) == GOLDEN_TAGGED_MAP_DIGEST


def test_mapping_order_does_not_affect_digest() -> None:
    first = canonical_sha256({"a": 1, "b": [1, 2], "c": {"x": "y"}})
    second = canonical_sha256({"c": {"x": "y"}, "b": [1, 2], "a": 1})
    assert first == second


def test_dataclass_expansion_is_fieldwise_and_stable() -> None:
    sample = _Sample(name="x", values=(1, 2, 3))
    assert canonical_value(sample) == {"name": "x", "values": [1, 2, 3], "tags": []}
    assert canonical_sha256(sample) == canonical_sha256(_Sample("x", (1, 2, 3)))
    assert canonical_sha256(sample) != canonical_sha256(_Sample("x", (1, 2, 4)))


def test_enum_serialization_is_stable() -> None:
    from edgeshard.profiling.domain.signature import OperatorKind

    assert canonical_value(OperatorKind.GEMM) == "gemm"
    assert canonical_sha256(("k", OperatorKind.GEMM)) == canonical_sha256(("k", "gemm"))


def test_timestamps_are_rejected() -> None:
    """§7.4: hashed identities must not embed volatile fields."""
    with pytest.raises(ValueError, match="timestamps"):
        canonical_sha256(("tag", datetime.now(UTC)))


def test_non_finite_floats_are_rejected() -> None:
    with pytest.raises(ValueError):
        canonical_sha256({"x": float("nan")})
    with pytest.raises(ValueError):
        canonical_sha256({"x": float("inf")})


def test_non_string_mapping_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="string"):
        canonical_json({1: "x"})


def test_order_insensitive_fields_are_sorted() -> None:
    first = canonical_sha256(
        _Sample("x", (1,), tags=("b", "a")), order_insensitive_fields={"tags"}
    )
    second = canonical_sha256(
        _Sample("x", (1,), tags=("a", "b")), order_insensitive_fields={"tags"}
    )
    assert first == second
    # Without the escape hatch, order remains significant.
    assert canonical_sha256(_Sample("x", (1,), tags=("b", "a"))) != canonical_sha256(
        _Sample("x", (1,), tags=("a", "b"))
    )


def test_normalized_items_sorts_and_validates() -> None:
    assert normalized_items({"b": 1, "a": 2}, "params") == (("a", 2), ("b", 1))


def test_normalized_items_rejects_empty_keys() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalized_items({"": 1}, "params")


def test_check_normalized_items_rejects_unsorted_or_duplicate() -> None:
    with pytest.raises(ValueError, match="sorted"):
        check_normalized_items([("b", 1), ("a", 2)], "params")
    with pytest.raises(ValueError, match="sorted"):
        check_normalized_items([("a", 1), ("a", 2)], "params")


def test_tuple_and_list_canonicalize_identically() -> None:
    """Domain fields are tuples; the canonical form is a JSON array."""
    assert canonical_json((1, (2, 3))) == "[1,[2,3]]"
