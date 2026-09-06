"""JSON payload codec for profiling domain objects (spec §41, §43-46).

Phase 2 crosses two serialization boundaries: the profiling wire protocol
(``protocol.profiling`` — spec §41 allows a simplification that fits the
existing control conventions) and the ProfileStore's payload columns
(§43-44). Both carry *domain DTOs*, never persistence-layer objects, and
both need the same strict, deterministic value encoding — this module is
that single codec.

Design:

* encoding is generic over frozen dataclasses: fields become object keys,
  ``StrEnum`` members become their values, tuples become arrays,
  ``datetime`` becomes ISO-8601 text, and every dataclass object carries a
  ``__type__`` tag so variant unions (``CaseSpec``, ``OperatorParameters``,
  the metric objects) decode without guessing;
* decoding is type-directed and strict: the declared field types drive
  reconstruction, unknown or missing keys fail loudly
  (:class:`PayloadCodecError`), and the domain ``__post_init__`` validation
  re-runs on every decode — a payload that would build an invalid domain
  object is rejected at the boundary (§47), never repaired;
* output text is deterministic (sorted keys, compact separators,
  ``allow_nan=False``) mirroring the §7 canonical-JSON discipline. Note the
  codec is *not* the identity hasher: ``canonical_value`` (hashing.py)
  rejects timestamps and adds no type tags, and stays the only source of
  canonical ids.

The numeric rule mirrors JSON semantics without silent narrowing: ``int``
fields accept only JSON integers (never ``bool``), ``float`` fields accept
integers and widen them, and non-finite floats are rejected by the encoder.
"""

from __future__ import annotations

import dataclasses
import json
import types
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any, TypeAliasType, Union, get_args, get_origin, get_type_hints

TYPE_TAG = "__type__"
"""Key carrying the dataclass name of an encoded object (variant dispatch)."""


class PayloadCodecError(ValueError):
    """A payload cannot be encoded to / decoded from the domain shape."""


def encode_payload(value: object) -> Any:
    """Domain value as JSON-compatible data (tagged objects, ISO datetimes)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value  # NaN/inf are rejected by encode_json's allow_nan=False
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        encoded: dict[str, Any] = {TYPE_TAG: type(value).__name__}
        for field in dataclasses.fields(value):
            encoded[field.name] = encode_payload(getattr(value, field.name))
        return encoded
    if isinstance(value, (tuple, list)):
        return [encode_payload(item) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PayloadCodecError(
                    f"mapping keys must be strings, got {type(key).__name__}"
                )
            result[key] = encode_payload(item)
        return result
    raise PayloadCodecError(f"cannot encode {type(value).__name__} as a JSON payload")


def encode_json(value: object) -> str:
    """Deterministic JSON text of a domain value (sorted keys, compact)."""
    try:
        return json.dumps(
            encode_payload(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except ValueError as exc:  # non-finite floats
        raise PayloadCodecError(f"payload is not JSON-serializable: {exc}") from exc


def decode_json[T](cls: type[T], text: str) -> T:
    """Parse JSON text and decode it into ``cls``, failing typed."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PayloadCodecError(f"malformed JSON payload: {exc}") from exc
    return decode_payload(cls, payload)


def decode_payload[T](cls: type[T], payload: object) -> T:
    """Decode JSON-compatible data into ``cls`` under strict type direction."""
    decoded = _decode(cls, payload, "$")
    if not isinstance(decoded, cls):
        raise PayloadCodecError(
            f"payload decoded to {type(decoded).__name__}, expected {cls.__name__}"
        )
    return decoded


# ---------------------------------------------------------------------------
# Type-directed decoding
# ---------------------------------------------------------------------------


def _unwrap(hint: Any) -> Any:
    """Resolve PEP-695 type aliases (``type JsonScalar = ...``) to their value."""
    while isinstance(hint, TypeAliasType):
        hint = hint.__value__
    return hint


def _decode(hint: Any, payload: object, path: str) -> Any:
    hint = _unwrap(hint)
    origin = get_origin(hint)
    if origin in (Union, types.UnionType):
        return _decode_union(hint, payload, path)
    if hint is Any:
        return payload
    if isinstance(hint, type):
        if dataclasses.is_dataclass(hint):
            return _decode_dataclass(hint, payload, path)
        if issubclass(hint, StrEnum):
            return _decode_enum(hint, payload, path)
        if hint is datetime:
            return _decode_datetime(payload, path)
        if hint is bool:
            if not isinstance(payload, bool):
                raise _type_error(hint, payload, path)
            return payload
        if hint is int:
            if isinstance(payload, bool) or not isinstance(payload, int):
                raise _type_error(hint, payload, path)
            return payload
        if hint is float:
            if isinstance(payload, bool) or not isinstance(payload, (int, float)):
                raise _type_error(hint, payload, path)
            return float(payload)
        if hint is str:
            if not isinstance(payload, str):
                raise _type_error(hint, payload, path)
            return payload
    if origin is tuple:
        return _decode_tuple(hint, payload, path)
    if origin is list:
        (item_hint,) = get_args(hint)
        if not isinstance(payload, list):
            raise _type_error(hint, payload, path)
        return [
            _decode(item_hint, item, f"{path}[{index}]")
            for index, item in enumerate(payload)
        ]
    if origin is dict:
        key_hint, value_hint = get_args(hint)
        if not isinstance(payload, dict):
            raise _type_error(hint, payload, path)
        return {
            _decode(key_hint, key, f"{path}.key"): _decode(
                value_hint, item, f"{path}.{key}"
            )
            for key, item in payload.items()
        }
    raise PayloadCodecError(f"{path}: unsupported type hint {hint!r}")


def _decode_union(hint: Any, payload: object, path: str) -> Any:
    args = tuple(_unwrap(arg) for arg in get_args(hint))
    if payload is None:
        if type(None) in args:
            return None
        raise PayloadCodecError(f"{path}: null is not a valid {hint!r}")
    candidates = tuple(arg for arg in args if arg is not type(None))
    dataclass_args = [
        arg for arg in candidates if isinstance(arg, type) and dataclasses.is_dataclass(arg)
    ]
    if dataclass_args and isinstance(payload, dict):
        # Variant union of tagged objects: the __type__ tag *commits* to one
        # variant — its decode errors surface, never fall through (a payload
        # tagged as a known variant is never silently reinterpreted).
        tag = payload.get(TYPE_TAG)
        for arg in dataclass_args:
            if arg.__name__ == tag:
                return _decode_dataclass(arg, payload, path)
        expected = ", ".join(sorted(arg.__name__ for arg in dataclass_args))
        raise PayloadCodecError(
            f"{path}: payload tag {tag!r} is not one of the variants ({expected})"
        )
    # No tagged variant applies: try each candidate with its strict decoder
    # in declaration order and keep the first that accepts the payload. In
    # the domain's unions the candidates are mutually exclusive by shape
    # (bool/int/float/str declare bool before int — an int subclass hazard —
    # and int before float, so integer samples never widen to floats), and
    # generic members like ``tuple[float, ...]`` only accept list payloads.
    for arg in candidates:
        try:
            return _decode(arg, payload, path)
        except PayloadCodecError:
            continue
    raise _type_error(hint, payload, path)


def _decode_dataclass(hint: type, payload: object, path: str) -> Any:
    if not isinstance(payload, dict):
        raise _type_error(hint, payload, path)
    tag = payload.get(TYPE_TAG)
    if tag != hint.__name__:
        raise PayloadCodecError(
            f"{path}: payload tag {tag!r} does not match {hint.__name__!r}"
        )
    try:
        hints = get_type_hints(hint)
    except NameError as exc:  # pragma: no cover - defensive
        raise PayloadCodecError(
            f"{path}: cannot resolve annotations of {hint.__name__}: {exc}"
        ) from exc
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(hint):
        if field.name not in payload:
            raise PayloadCodecError(f"{path}: missing field {field.name!r} for {hint.__name__}")
        kwargs[field.name] = _decode(
            hints[field.name], payload[field.name], f"{path}.{field.name}"
        )
    extra = set(payload) - {field.name for field in dataclasses.fields(hint)} - {TYPE_TAG}
    if extra:
        raise PayloadCodecError(
            f"{path}: unknown field(s) {sorted(extra)} for {hint.__name__}"
        )
    try:
        return hint(**kwargs)
    except (ValueError, TypeError) as exc:
        raise PayloadCodecError(f"{path}: invalid {hint.__name__}: {exc}") from exc


def _decode_enum(hint: type, payload: object, path: str) -> Any:
    if not isinstance(payload, str):
        raise _type_error(hint, payload, path)
    try:
        return hint(payload)
    except ValueError as exc:
        raise PayloadCodecError(f"{path}: {payload!r} is not a {hint.__name__}") from exc


def _decode_datetime(payload: object, path: str) -> datetime:
    if not isinstance(payload, str):
        raise _type_error(datetime, payload, path)
    try:
        return datetime.fromisoformat(payload)
    except ValueError as exc:
        raise PayloadCodecError(f"{path}: {payload!r} is not an ISO-8601 datetime") from exc


def _decode_tuple(hint: Any, payload: object, path: str) -> Any:
    if not isinstance(payload, (list, tuple)):
        raise _type_error(hint, payload, path)
    args = get_args(hint)
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(
            _decode(args[0], item, f"{path}[{index}]") for index, item in enumerate(payload)
        )
    if len(args) != len(payload):
        raise PayloadCodecError(
            f"{path}: expected {len(args)} tuple items, payload has {len(payload)}"
        )
    return tuple(
        _decode(arg, item, f"{path}[{index}]")
        for index, (arg, item) in enumerate(zip(args, payload, strict=True))
    )


def _type_error(hint: Any, payload: object, path: str) -> PayloadCodecError:
    return PayloadCodecError(
        f"{path}: expected {hint!r}, got {type(payload).__name__} ({payload!r})"
    )
