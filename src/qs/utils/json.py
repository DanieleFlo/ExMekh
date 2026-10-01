from __future__ import annotations

import json
import math
from typing import Any, overload

type JsonValue = (
    bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
)


def serialize_json(value: object) -> str:
    """Serialize compact JSON, preserving Unicode and rejecting nonfinite numbers."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


@overload
def to_json_value(value: dict[str, Any]) -> dict[str, JsonValue]: ...


@overload
def to_json_value(value: object) -> JsonValue: ...


def to_json_value(value: object) -> JsonValue:
    """Copy a value into JSON data, representing unsupported values as strings.

    Only ancestors count as cycles, so shared references remain ordinary data.
    """
    return _convert(value, set())


def _convert(value: object, ancestors: set[int]) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if not isinstance(value, (dict, list, tuple, set, frozenset)):
        return repr(value)
    identity = id(value)
    if identity in ancestors:
        return "<circular reference>"
    ancestors.add(identity)
    try:
        if isinstance(value, dict):
            return {str(key): _convert(item, ancestors) for key, item in value.items()}
        return [_convert(item, ancestors) for item in value]
    finally:
        ancestors.remove(identity)
