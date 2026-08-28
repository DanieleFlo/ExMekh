from pydantic import BaseModel

from qs.core.providers.openai_response_format_mapper import OpenAIResponseFormatMapper


class StructuredAnswer(BaseModel):
    value: int


def test_map_wraps_json_schema_for_responses_text_format() -> None:
    mapped = OpenAIResponseFormatMapper.map(StructuredAnswer)

    assert mapped == {
        "format": {
            "type": "json_schema",
            "name": "StructuredAnswer",
            "schema": {
                "type": "object",
                "title": "StructuredAnswer",
                "properties": {"value": {"type": "integer", "title": "Value"}},
                "additionalProperties": False,
                "required": ["value"],
            },
            "strict": True,
        }
    }
