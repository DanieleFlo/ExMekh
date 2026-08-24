from __future__ import annotations

from dataclasses import dataclass

from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

from qs.core.models.assistant_message import FunctionToolCall

OpenAIChatCompletionToolCall = FunctionToolCall | ChatCompletionMessageFunctionToolCall


@dataclass(frozen=True)
class OpenAIChatCompletionToolCallData:
    name: str
    arguments: str
