from __future__ import annotations

from typing import Any

from openai.types.responses import ResponseFunctionToolCall

from exmekh_lib.config.provider_text import PROVIDER_TEXT
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.providers.openai_chat_completion_annotation_schema_builder import (
    OpenAIChatCompletionAnnotationSchemaBuilder,
)
from exmekh_lib.core.providers.openai_chat_completion_value_coercer import (
    OpenAIChatCompletionValueCoercer,
)
from exmekh_lib.core.providers.openai_function_tool_call import OpenAIFunctionToolCall
from exmekh_lib.core.providers.openai_function_tool_call_resolver import (
    OpenAIFunctionToolCallResolver,
)
from exmekh_lib.core.providers.openai_function_tool_definition_mapper import (
    OpenAIFunctionToolDefinition,
    OpenAIFunctionToolDefinitionMapper,
)


class OpenAIResponseToolMapper:
    def __init__(self) -> None:
        self._definition_mapper = OpenAIFunctionToolDefinitionMapper(
            OpenAIChatCompletionAnnotationSchemaBuilder()
        )
        self._call_resolver = OpenAIFunctionToolCallResolver(
            OpenAIChatCompletionValueCoercer()
        )

    @classmethod
    def map_tools(cls, tools: Tools) -> list[dict[str, Any]]:
        mapper = cls()
        definitions = mapper._definition_mapper.map_all(tools, strict=True)
        return [mapper._map_tool(definition) for definition in definitions]

    @staticmethod
    def _map_tool(definition: OpenAIFunctionToolDefinition) -> dict[str, Any]:
        mapped: dict[str, Any] = {
            "type": "function",
            "name": definition.get_name(),
            "parameters": definition.get_parameters(),
            "strict": True,  # OpenAI's response format requires strict validation of the parameters
        }
        description = definition.get_description()
        if description:
            mapped["description"] = description
        return mapped

    @classmethod
    def resolve_tool_call(
        cls,
        tool_call: ResponseFunctionToolCall,
        tools: Tools,
    ) -> tuple[Tool, dict[str, Any]]:
        if not isinstance(tool_call, ResponseFunctionToolCall):
            raise TypeError(
                PROVIDER_TEXT.tool_call_must_be_an_openai_response_function_tool
            )
        call = OpenAIFunctionToolCall(
            name=tool_call.name,
            arguments=tool_call.arguments,
        )
        return cls()._call_resolver.resolve(call, tools)
