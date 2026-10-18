from __future__ import annotations

from typing import Any, ClassVar

from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

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
    OpenAIFunctionToolDefinitionMapper,
)


class OpenAIChatCompletionToolMapper:
    """Adapt Chat Completions calls with reusable, stateless mapping services."""

    _definition_mapper: ClassVar[OpenAIFunctionToolDefinitionMapper] = (
        OpenAIFunctionToolDefinitionMapper(
            OpenAIChatCompletionAnnotationSchemaBuilder()
        )
    )
    _call_resolver: ClassVar[OpenAIFunctionToolCallResolver] = (
        OpenAIFunctionToolCallResolver(OpenAIChatCompletionValueCoercer())
    )

    @classmethod
    def map_tool(cls, tool: Tool) -> ChatCompletionFunctionToolParam:
        definition = cls._definition_mapper.map(tool)
        return ChatCompletionFunctionToolParam(
            type="function",
            function=definition.to_openai(),
        )

    @classmethod
    def map_tools(
        cls,
        tools: Tools,
    ) -> list[ChatCompletionFunctionToolParam]:
        definitions = cls._definition_mapper.map_all(tools)
        return [
            ChatCompletionFunctionToolParam(
                type="function",
                function=definition.to_openai(),
            )
            for definition in definitions
        ]

    @classmethod
    def find_tool(
        cls,
        tool_call: ChatCompletionMessageFunctionToolCall,
        tools: Tools,
    ) -> Tool:
        call = cls._read_tool_call(tool_call)
        return cls._call_resolver.find_tool(call, tools)

    @classmethod
    def resolve_tool_call(
        cls,
        tool_call: ChatCompletionMessageFunctionToolCall,
        tools: Tools,
    ) -> tuple[Tool, dict[str, Any]]:
        call = cls._read_tool_call(tool_call)
        return cls._call_resolver.resolve(call, tools)

    @staticmethod
    def _read_tool_call(
        tool_call: ChatCompletionMessageFunctionToolCall,
    ) -> OpenAIFunctionToolCall:
        if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
            raise TypeError(
                PROVIDER_TEXT.tool_call_must_be_an_openai_function_tool_call
            )
        return OpenAIFunctionToolCall(
            name=tool_call.function.name,
            arguments=tool_call.function.arguments,
        )
