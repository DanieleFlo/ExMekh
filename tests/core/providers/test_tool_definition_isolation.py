"""Exported provider schemas cannot mutate their tool definition."""

import pytest

from exmekh_lib.core.models.annotation_schema import (
    AnnotationSchemaBuilder,
)
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.providers.openai_function_tool_definition_mapper import (
    OpenAIFunctionToolDefinitionMapper,
)


@pytest.mark.parametrize("export", ["parameters", "openai"])
def test_nested_schema_exports_are_independent(export: str) -> None:
    def choose(values: list[str]) -> str:
        return values[0]

    mapper = OpenAIFunctionToolDefinitionMapper(AnnotationSchemaBuilder())
    definition = mapper.map(Tool(choose))
    parameters = (
        definition.get_parameters()
        if export == "parameters"
        else definition.to_openai()["parameters"]
    )
    parameters["properties"]["values"]["items"]["type"] = "integer"
    assert (
        definition.get_parameters()["properties"]["values"]["items"]["type"] == "string"
    )
    assert (
        definition.to_openai()["parameters"]["properties"]["values"]["items"]["type"]
        == "string"
    )
