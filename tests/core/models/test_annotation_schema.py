from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    NewType,
    NotRequired,
    Required,
    TypedDict,
)
from uuid import UUID

import pytest

from exmekh_lib.core.models.annotation_schema import AnnotationSchemaBuilder


def test_schema_metadata_copies_do_not_share_nested_values():
    metadata = {"examples": [{"description": "Literal example"}]}
    annotation = Annotated[str, metadata]
    builder = AnnotationSchemaBuilder()
    first = builder.build(annotation)
    second = builder.build(annotation)

    first["examples"][0]["description"] = "Changed"

    assert second["examples"] == [{"description": "Literal example"}]
    assert metadata["examples"] == [{"description": "Literal example"}]


class OptionalFilters(TypedDict, total=False):
    category: Required[str]
    label: NotRequired[str]


class AllOptionalFilters(TypedDict, total=False):
    category: str


class Mode(Enum):
    FAST = "fast"
    ACCURATE = "accurate"


@dataclass
class Location:
    city: str
    radius: int = 5


@dataclass
class RecursiveNode:
    child: "RecursiveNode | None" = None


UserId = NewType("UserId", int)


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (Any, {}),
        (object, {}),
        (str, {"type": "string"}),
        (int, {"type": "integer"}),
        (float, {"type": "number"}),
        (bool, {"type": "boolean"}),
        (type(None), {"type": "null"}),
        (date, {"type": "string", "format": "date"}),
        (datetime, {"type": "string", "format": "date-time"}),
        (UUID, {"type": "string", "format": "uuid"}),
        (Decimal, {"type": "number"}),
        (Path, {"type": "string"}),
        (Literal["a", 1], {"enum": ["a", 1]}),
        (Mode, {"enum": ["fast", "accurate"], "type": "string"}),
        (list, {"type": "array", "items": {}}),
        (list[int], {"type": "array", "items": {"type": "integer"}}),
        (
            set[str],
            {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        ),
        (tuple, {"type": "array", "items": {}}),
        (tuple[int, ...], {"type": "array", "items": {"type": "integer"}}),
        (
            tuple[str, int],
            {
                "type": "array",
                "prefixItems": [{"type": "string"}, {"type": "integer"}],
                "minItems": 2,
                "maxItems": 2,
            },
        ),
        (dict, {"type": "object", "additionalProperties": {}}),
        (
            Mapping[str, bool],
            {"type": "object", "additionalProperties": {"type": "boolean"}},
        ),
        (
            OptionalFilters,
            {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "label": {"type": "string"},
                },
                "additionalProperties": False,
                "required": ["category"],
            },
        ),
        (
            Location,
            {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "radius": {"type": "integer"},
                },
                "additionalProperties": False,
                "required": ["city"],
            },
        ),
        (UserId, {"type": "integer"}),
        (
            Annotated[int, "Positive count", {"minimum": 1}],
            {"type": "integer", "description": "Positive count", "minimum": 1},
        ),
        (Annotated[int, object()], {"type": "integer"}),
        (
            AllOptionalFilters,
            {
                "type": "object",
                "properties": {"category": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        (Sequence[str], {"type": "array", "items": {"type": "string"}}),
    ],
)
def test_build_supports_python_type_rules(
    annotation: object,
    expected: dict,
) -> None:
    builder = AnnotationSchemaBuilder()

    assert builder.build(annotation) == expected


def test_build_rejects_non_string_mapping_keys() -> None:
    builder = AnnotationSchemaBuilder()

    with pytest.raises(TypeError, match="keys must be annotated as str"):
        builder.build(dict[int, str])


def test_build_rejects_unsupported_type() -> None:
    builder = AnnotationSchemaBuilder()

    with pytest.raises(TypeError, match="Unsupported tool parameter type"):
        builder.build(bytes)


def test_build_rejects_recursive_dataclass() -> None:
    builder = AnnotationSchemaBuilder()

    with pytest.raises(TypeError, match="Recursive"):
        builder.build(RecursiveNode)
