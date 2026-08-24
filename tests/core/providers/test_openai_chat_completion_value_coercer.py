from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import NewType, NotRequired, Required, TypedDict
from uuid import UUID

import pytest

from qs.core.providers.openai_chat_completion_value_coercer import (
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
        ([1, 2], tuple[int, ...], (1, 2)),
        (["a", "a"], set[str], {"a"}),
        (["a"], frozenset[str], frozenset({"a"})),
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
        ("slow", Mode, "enum"),
        ("not-a-date", date, "invalid value"),
        ({"label": "x"}, OptionalFilters, "missing fields"),
        ({"category": "x", "extra": True}, OptionalFilters, "unknown fields"),
        ({"city": "Rome", "extra": True}, Location, "unknown fields"),
        ([1], tuple[int, str], "wrong array length"),
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
