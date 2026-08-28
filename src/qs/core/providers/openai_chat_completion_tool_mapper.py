from __future__ import annotations

from typing import Any

from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

from qs.core.models.tool import Tool
from qs.core.models.tools import Tools
from qs.core.providers.openai_chat_completion_annotation_schema_builder import (
    OpenAIChatCompletionAnnotationSchemaBuilder,
)
from qs.core.providers.openai_chat_completion_tool_call_resolver import (
    OpenAIChatCompletionToolCallResolver,
)
from qs.core.providers.openai_chat_completion_tool_schema_mapper import (
    OpenAIChatCompletionToolSchemaMapper,
)
from qs.core.providers.openai_chat_completion_value_coercer import (
    OpenAIChatCompletionValueCoercer,
)


class OpenAIChatCompletionToolMapper:
    def __init__(self) -> None:
        schema_builder = OpenAIChatCompletionAnnotationSchemaBuilder()
        coercer = OpenAIChatCompletionValueCoercer()

        self._tool_mapper = OpenAIChatCompletionToolSchemaMapper(schema_builder)
        self._call_resolver = OpenAIChatCompletionToolCallResolver(coercer)

    @classmethod
    def map_tool(cls, tool: Tool) -> ChatCompletionFunctionToolParam:
        return cls()._tool_mapper.map(tool)

    @classmethod
    def map_tools(
        cls,
        tools: Tools,
    ) -> list[ChatCompletionFunctionToolParam]:
        return cls()._tool_mapper.map_all(tools)

    @classmethod
    def find_tool(
        cls,
        tool_call: ChatCompletionMessageFunctionToolCall,
        tools: Tools,
    ) -> Tool:
        return cls()._call_resolver.find_tool(tool_call, tools)

    @classmethod
    def resolve_tool_call(
        cls,
        tool_call: ChatCompletionMessageFunctionToolCall,
        tools: Tools,
    ) -> tuple[Tool, dict[str, Any]]:
        return cls()._call_resolver.resolve(tool_call, tools)
