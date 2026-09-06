"""Canonical serialization and identity hashing (Phase 2 spec §7).

Every reusable profiling identity — model, transformer-layer, module,
operator, and network-pair signature ids, device performance class ids,
environment fingerprints, and profiling case ids — is the SHA-256 digest of
a canonical serialization. The mechanism deliberately mirrors the Phase 1
``capability_revision`` (``edgeshard.cluster.capability``): canonical JSON
with sorted keys and compact separators, dataclasses expanded field by
field, ``StrEnum`` members reduced to their values, tuples reduced to
arrays. The cluster utility is private to a frozen Phase 1 package that
profiling must not import, so the identical scheme is provided here — no
second, *incompatible* canonicalization is introduced (§7).

Guarantees required by §7:

1. semantically identical values hash identically;
2. mapping ordering never affects a hash (JSON ``sort_keys`` plus
   normalized item tuples for stored mappings);
3. enum serialization is stable (``StrEnum.value``);
4. hashed identities carry no volatile fields: timestamps are rejected
   outright so a signature can never embed one by accident;
5. every hash function has explicit unit tests
   (``tests/unit/profiling/test_hashing.py``).

``json.dumps(..., allow_nan=False)`` keeps digests canonical: NaN and
infinity have no JSON representation and would poison cross-language
comparisons, so they fail loudly instead.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Iterable, Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import StrEnum

type JsonScalar = bool | int | float | str | None
"""JSON scalar vocabulary for free-form metadata mappings (spec §8.3)."""

type StructuralValue = int | float | str | bool
"""Value vocabulary for structural parameter mappings (spec §6.2)."""


def check_normalized_items[T](
    items: Iterable[tuple[str, T]], field_name: str
) -> tuple[tuple[str, T], ...]:
    """Validate canonical item tuples: non-empty keys, sorted, unique.

    Mapping-shaped domain fields are stored as sorted ``(key, value)`` item
    tuples so frozen dataclasses stay hashable (signatures are deduplicated
    in sets, spec §20) and mapping order never leaks into identity (§7.2).
    """
    result = tuple(items)
    previous: str | None = None
    for key, _ in result:
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if previous is not None and key <= previous:
            raise ValueError(
                f"{field_name} items must be sorted by unique key "
                f"({previous!r} followed by {key!r})"
            )
        previous = key
    return result


def normalized_items[T](mapping: Mapping[str, T], field_name: str) -> tuple[tuple[str, T], ...]:
    """Normalize a mapping into canonical sorted item tuples.

    Use this when constructing domain objects that store a mapping-shaped
    field (structural parameters, metadata, software versions)::

        ModuleSignature(..., structural_parameters=normalized_items(params))
    """
    for key in mapping:
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
    return check_normalized_items(sorted(mapping.items()), field_name)


def _canonical_sort_key(item: object) -> str:
    return json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_value(
    value: object, *, order_insensitive_fields: Collection[str] = frozenset()
) -> object:
    """Convert a domain structure to plain JSON-serializable data.

    Dataclasses become mappings keyed by field name, tuples become arrays,
    ``StrEnum`` members become their values, mappings become dicts (JSON
    ``sort_keys`` makes their ordering irrelevant). Fields named in
    ``order_insensitive_fields`` are additionally sorted by their canonical
    encoding so enumeration order never leaks into the digest — the same
    escape hatch ``capability_revision`` uses for devices/pools/interfaces.

    ``datetime`` values are rejected: hashed identities must not embed
    volatile fields (§7.4).
    """
    if isinstance(value, datetime):
        raise ValueError(
            "timestamps must not be part of a canonical identity (§7.4): "
            f"got {value!r}"
        )
    if is_dataclass(value) and not isinstance(value, type):
        mapping: dict[str, object] = {}
        for field in fields(value):
            item = canonical_value(
                getattr(value, field.name),
                order_insensitive_fields=order_insensitive_fields,
            )
            if field.name in order_insensitive_fields and isinstance(item, list):
                item = sorted(item, key=_canonical_sort_key)
            mapping[field.name] = item
        return mapping
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return [
            canonical_value(item, order_insensitive_fields=order_insensitive_fields)
            for item in value
        ]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"canonical mapping keys must be strings, got {type(key).__name__}"
                )
            result[key] = canonical_value(
                item, order_insensitive_fields=order_insensitive_fields
            )
        return result
    return value


def canonical_json(
    value: object, *, order_insensitive_fields: Collection[str] = frozenset()
) -> str:
    """Canonical JSON text of a domain structure (sorted keys, compact)."""
    return json.dumps(
        canonical_value(value, order_insensitive_fields=order_insensitive_fields),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(
    value: object, *, order_insensitive_fields: Collection[str] = frozenset()
) -> str:
    """SHA-256 hex digest of the canonical JSON of a domain structure."""
    payload = canonical_json(value, order_insensitive_fields=order_insensitive_fields)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
