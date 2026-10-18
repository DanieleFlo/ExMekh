from copy import deepcopy

import pytest
from pydantic import BaseModel, Field

from exmekh_lib.core.providers.openai_json_schema_mapper import OpenAIJSONSchemaMapper


@pytest.mark.parametrize("keyword", ["default", "examples", "const", "enum", "x-data"])
def test_strict_mapping_preserves_literal_objects_in_schema_metadata(keyword):
    literal = {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "default": None,
    }
    value = [literal] if keyword in {"examples", "enum"} else literal

    class Answer(BaseModel):
        payload: dict[str, object] = Field(json_schema_extra={keyword: value})

    original = deepcopy(Answer.model_json_schema())
    schema = OpenAIJSONSchemaMapper.map(Answer)["schema"]

    assert schema["properties"]["payload"][keyword] == value
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["payload"]
    assert Answer.model_json_schema() == original
