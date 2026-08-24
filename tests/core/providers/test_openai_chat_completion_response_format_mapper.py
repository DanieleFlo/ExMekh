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
