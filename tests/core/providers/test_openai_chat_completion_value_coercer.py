from collections import abc
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import NoneType
from typing import (
    Annotated,
    Any,
    Literal,
    NewType,
    NotRequired,
    Required,
    TypedDict,
    TypeVar,
)
from uuid import UUID

import pytest

from exmekh_lib.core.providers.openai_chat_completion_value_coercer import (
    OpenAIChatCompletionValueCoercer,
)


class OptionalFilters(TypedDict, total=False):
    category: Required[str]
    label: NotRequired[str]


class Mode(Enum):
    FAST = "fast"
    ACCURATE = "accurate"


@dataclass
class Location:
    city: str
    radius: int = 5


UserId = NewType("UserId", int)
Unrestricted = TypeVar("Unrestricted")


@pytest.mark.parametrize(
    ("value", "annotation", "expected"),
    [
        (3, float, 3.0),
        ("fast", Mode, Mode.FAST),
        ("2026-08-21", date, date(2026, 8, 21)),
        (
            "2026-08-21T10:30:00+00:00",
            datetime,
            datetime(2026, 8, 21, 10, 30, tzinfo=UTC),
        ),
        (
            "12345678-1234-5678-1234-567812345678",
            UUID,
            UUID("12345678-1234-5678-1234-567812345678"),
        ),
        ("1.25", Decimal, Decimal("1.25")),
        ("docs/readme.md", Path, Path("docs/readme.md")),
        ("raw", Any, "raw"),
        ({"raw": True}, object, {"raw": True}),
        ("generic", Unrestricted, "generic"),
        (4, Annotated[int, "positive"], 4),
        (None, NoneType, None),
        (True, bool, True),
        (4, int, 4),
        ("text", int | str, "text"),
        ("metric", Literal["metric", 1], "metric"),
        ([1, "two"], list, [1, "two"]),
        ([1, 2], abc.Sequence[int], [1, 2]),
        ([1, 2], tuple[int, ...], (1, 2)),
        ([1, "two"], tuple, (1, "two")),
        ([1, "two"], tuple[int, str], (1, "two")),
        (["a", "a"], set[str], {"a"}),
        (["a"], frozenset[str], frozenset({"a"})),
        ({"one": 1}, dict, {"one": 1}),
        ({"one": 1}, abc.Mapping[str, int], {"one": 1}),
        ({"category": "weather"}, OptionalFilters, {"category": "weather"}),
        ({"city": "Rome"}, Location, Location("Rome")),
        (7, UserId, 7),
    ],
)
def test_coerce_converts_json_values_for_python(
    value: object,
    annotation: object,
    expected: object,
) -> None:
    coercer = OpenAIChatCompletionValueCoercer()

    assert coercer.coerce(value, annotation, "value") == expected


@pytest.mark.parametrize(
    ("value", "annotation", "message"),
    [
        (1, bool, "boolean"),
        (True, int, "integer"),
        ("one", float, "number"),
        (1, str, "string"),
        (1, NoneType, "must be null"),
        ([], int | str, "does not match any allowed type"),
        (True, Literal[1], "allowed literal"),
        ("slow", Mode, "enum"),
        ("not-a-date", date, "invalid value"),
        ("not-a-uuid", UUID, "invalid value"),
        ("not-a-decimal", Decimal, "invalid value"),
        ("not-an-array", list[int], "must be an array"),
        ("not-an-object", dict[str, int], "must be an object"),
        ({"label": "x"}, OptionalFilters, "missing fields"),
        ({"category": "x", "extra": True}, OptionalFilters, "unknown fields"),
        ({"city": "Rome", "extra": True}, Location, "unknown fields"),
        ({}, Location, "is invalid"),
        ([1], tuple[int, str], "wrong array length"),
        ("value", complex, "Unsupported type"),
    ],
)
def test_coerce_rejects_values_that_violate_annotation(
    value: object,
    annotation: object,
    message: str,
) -> None:
    coercer = OpenAIChatCompletionValueCoercer()

    with pytest.raises(TypeError, match=message):
        coercer.coerce(value, annotation, "value")
