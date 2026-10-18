from copy import deepcopy

import pytest

from exmekh_lib.core.models.json_schema import (
    SchemaDescription,
    schema_descriptions,
    without_schema_descriptions,
)


def test_schema_documentation_preserves_literal_data_and_description_fields():
    schema = {
        "description": "Request documentation",
        "properties": {
            "description": {
                "type": "string",
                "description": "Field documentation",
                "enum": [{"description": "literal enum value"}],
                "default": {"description": "literal default"},
                "examples": [{"description": "literal example"}],
                "const": {"description": "literal constant"},
            },
        },
        "required": ["description"],
        "additionalProperties": False,
    }
    original = deepcopy(schema)

    assert schema_descriptions(schema) == (
        SchemaDescription("", "Request documentation"),
        SchemaDescription("/properties/description", "Field documentation"),
    )
    stripped = without_schema_descriptions(schema)
    assert "description" not in stripped
    expected = original["properties"]["description"].copy()
    expected.pop("description")
    assert stripped["properties"]["description"] == expected
    assert stripped["required"] == ["description"]
    assert stripped["additionalProperties"] is False
    assert schema == original


@pytest.mark.parametrize(
    "keyword",
    ["properties", "patternProperties", "$defs", "definitions", "dependentSchemas"],
)
def test_schema_maps_use_escaped_json_pointers(keyword):
    schema = {keyword: {"a~/b": {"description": "Nested documentation"}, "flag": True}}

    assert schema_descriptions(schema) == (
        SchemaDescription(f"/{keyword}/a~0~1b", "Nested documentation"),
    )
    assert without_schema_descriptions(schema) == {keyword: {"a~/b": {}, "flag": True}}


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "allOf", "prefixItems", "items"])
def test_schema_arrays_include_branch_positions(keyword):
    schema = {keyword: [False, {"description": "Branch documentation"}]}

    assert schema_descriptions(schema) == (
        SchemaDescription(f"/{keyword}/1", "Branch documentation"),
    )
    assert without_schema_descriptions(schema) == {keyword: [False, {}]}


@pytest.mark.parametrize(
    "keyword", ["items", "additionalProperties", "not", "if", "then", "else"]
)
def test_nested_schema_values_are_documented(keyword):
    schema = {keyword: {"description": "Nested documentation", "type": "integer"}}

    assert schema_descriptions(schema) == (
        SchemaDescription(f"/{keyword}", "Nested documentation"),
    )
    assert without_schema_descriptions(schema) == {keyword: {"type": "integer"}}


def test_empty_documentation_and_non_schema_extension_values_are_ignored():
    schema = {"description": "", "anyOf": None, "extension": {"description": "literal"}}
    assert schema_descriptions(schema) == ()
    assert without_schema_descriptions(schema) == {
        "anyOf": None,
        "extension": {"description": "literal"},
    }
