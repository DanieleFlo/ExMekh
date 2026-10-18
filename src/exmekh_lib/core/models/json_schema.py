"""Visit JSON Schema nodes without interpreting literal values as schemas."""

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

Schema = dict[str, Any]

_SCHEMA_MAPS = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
)
_SCHEMA_ARRAYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems", "items"})
_SCHEMA_VALUES = frozenset(
    {
        "items",
        "additionalProperties",
        "additionalItems",
        "unevaluatedProperties",
        "unevaluatedItems",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
    }
)


@dataclass(frozen=True, slots=True)
class SchemaDescription:
    path: str
    text: str


def schema_descriptions(schema: Schema) -> tuple[SchemaDescription, ...]:
    """Return descriptions with JSON Pointers relative to the parameter schema."""
    return tuple(
        SchemaDescription(path, node["description"])
        for path, node in iter_schema_nodes(schema)
        if isinstance(node.get("description"), str) and node["description"]
    )


def without_schema_descriptions(schema: Schema) -> Schema:
    """Copy a schema without documentation, preserving fields named description."""
    copied = deepcopy(schema)
    for _, node in iter_schema_nodes(copied):
        node.pop("description", None)
    return copied


def _pointer_part(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def iter_schema_nodes(schema: Schema, path: str = "") -> Iterator[tuple[str, Schema]]:
    """Yield mutable schema nodes and their JSON Pointers, skipping literal data."""
    yield path, schema
    for key, value in schema.items():
        child_path = f"{path}/{_pointer_part(key)}"
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            for name, child in value.items():
                if isinstance(child, dict):
                    yield from iter_schema_nodes(
                        child, f"{child_path}/{_pointer_part(name)}"
                    )
        elif key in _SCHEMA_ARRAYS and isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, dict):
                    yield from iter_schema_nodes(child, f"{child_path}/{index}")
        elif key in _SCHEMA_VALUES and isinstance(value, dict):
            yield from iter_schema_nodes(value, child_path)
