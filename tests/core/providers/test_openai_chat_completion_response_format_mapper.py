from typing import Annotated, cast

import pytest
from pydantic import BaseModel, Field

from qs.core.providers.openai_chat_completion_response_format_mapper import (
    OpenAIChatCompletionResponseFormatMapper,
)


class Details(BaseModel):
    label: str


class StructuredAnswer(BaseModel):
    value: int
    details: Annotated[Details | None, Field(description="Optional details")] = None


class AnswerWithDescribedReference(BaseModel):
    details: Annotated[Details, Field(description="Required details")]


class AnswerWithMappingAndDefault(BaseModel):
    labels: dict[str, int]
    count: int = 3


def test_map_builds_strict_nested_json_schema_by_default() -> None:
    mapped = OpenAIChatCompletionResponseFormatMapper.map(StructuredAnswer)

    assert mapped["type"] == "json_schema"
    assert mapped["json_schema"]["name"] == "StructuredAnswer"
    assert mapped["json_schema"]["strict"] is True

    schema = mapped["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["value", "details"]
    assert "default" not in schema["properties"]["details"]

    details_schema = schema["$defs"]["Details"]
    assert details_schema["additionalProperties"] is False
    assert details_schema["required"] == ["label"]


def test_map_preserves_pydantic_schema_in_non_strict_mode() -> None:
    mapped = OpenAIChatCompletionResponseFormatMapper.map(
        StructuredAnswer,
        strict=False,
    )

    assert mapped == {
        "type": "json_schema",
        "json_schema": {
            "name": "StructuredAnswer",
            "schema": StructuredAnswer.model_json_schema(),
            "strict": False,
        },
    }


@pytest.mark.parametrize("response_model", [object, StructuredAnswer(value=1)])
def test_map_rejects_non_pydantic_model_type(response_model) -> None:
    with pytest.raises(TypeError, match="Pydantic BaseModel type"):
        OpenAIChatCompletionResponseFormatMapper.map(
            cast(type[BaseModel], response_model)
        )


def test_map_rejects_non_boolean_strict_value() -> None:
    with pytest.raises(TypeError, match="Strict must be a boolean"):
        OpenAIChatCompletionResponseFormatMapper.map(
            StructuredAnswer,
            strict=cast(bool, 1),
        )


def test_map_inlines_reference_with_sibling_metadata_in_strict_mode() -> None:
    mapped = OpenAIChatCompletionResponseFormatMapper.map(AnswerWithDescribedReference)

    details = mapped["json_schema"]["schema"]["properties"]["details"]
    assert "$ref" not in details
    assert details["description"] == "Required details"
    assert details["type"] == "object"
    assert details["additionalProperties"] is False
    assert details["required"] == ["label"]


def test_map_preserves_mapping_schema_and_non_null_default_in_strict_mode() -> None:
    mapped = OpenAIChatCompletionResponseFormatMapper.map(AnswerWithMappingAndDefault)

    properties = mapped["json_schema"]["schema"]["properties"]
    assert properties["labels"] == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
        "title": "Labels",
    }
    assert properties["count"]["default"] == 3


def test_resolve_reference_decodes_json_pointer_escapes() -> None:
    resolved = OpenAIChatCompletionResponseFormatMapper._resolve_reference(
        {"$defs": {"path/to~value": {"type": "string"}}},
        "#/$defs/path~1to~0value",
    )

    assert resolved == {"type": "string"}


@pytest.mark.parametrize(
    ("root", "reference", "message"),
    [
        ({}, "external.json", "Unsupported JSON schema reference"),
        ({"$defs": {}}, "#/$defs/Missing", "Unresolvable JSON schema reference"),
        ({"$defs": {"Value": "text"}}, "#/$defs/Value", "not an object"),
    ],
)
def test_resolve_reference_rejects_invalid_targets(
    root: dict, reference: str, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        OpenAIChatCompletionResponseFormatMapper._resolve_reference(root, reference)
