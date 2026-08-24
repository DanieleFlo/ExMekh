from __future__ import annotations

from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

from qs.core.models.assistant_message import FunctionToolCall
from qs.core.providers.openai_chat_completion_tool_call_data import (
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallData,
)


class OpenAIChatCompletionToolCallReader:
    def read(
        self,
        tool_call: OpenAIChatCompletionToolCall,
    ) -> OpenAIChatCompletionToolCallData:
        if isinstance(
            tool_call,
            ChatCompletionMessageFunctionToolCall,
        ):
            return OpenAIChatCompletionToolCallData(
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )

        return self._read_internal(tool_call)

    @staticmethod
    def _read_internal(
        tool_call: FunctionToolCall,
    ) -> OpenAIChatCompletionToolCallData:
        if not isinstance(tool_call, FunctionToolCall):
            raise TypeError("Expected a function tool call")

        name = tool_call.function.name
        arguments = tool_call.function.arguments

        if not isinstance(name, str):
            raise TypeError("Function tool call name must be a string")

        if not isinstance(arguments, str):
            raise TypeError("Function tool call arguments must be a string")

        return OpenAIChatCompletionToolCallData(
            name=name,
            arguments=arguments,
        )
