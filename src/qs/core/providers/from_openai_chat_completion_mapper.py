from __future__ import annotations

from typing import ClassVar

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCallUnion,
)
from openai.types.chat.chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

from qs.core.models.assistant_message import (
    AssistantMessage,
    CustomCall,
    CustomToolCall,
    FunctionCall,
    FunctionToolCall,
    ToolCall,
)
from qs.core.models.model_thinking import ModelThinking
from qs.core.models.model_usage import ModelUsage


class FromOpenAIChatCompletionMapper:
    _THINKING_FIELDS: ClassVar[tuple[str, ...]] = (
        "reasoning",
        "reasoning_content",
        "thinking",
    )

    @staticmethod
    def extract_usage(response: ChatCompletion) -> ModelUsage:
        usage = response.usage

        if usage is None:
            return ModelUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            )

        return ModelUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    @classmethod
    def extract_thinking(
        cls,
        response: ChatCompletion,
    ) -> ModelThinking | None:
        for choice in response.choices:
            thinking = cls._thinking_from(choice.message)

            if thinking is not None:
                return ModelThinking(thinking)

        return None

    @classmethod
    def map_assistant_message(
        cls,
        message: ChatCompletionMessage,
    ) -> AssistantMessage:
        return AssistantMessage(
            content=message.content,
            tool_calls=cls.map_tool_calls(message.tool_calls),
        )

    @classmethod
    def _thinking_from(
        cls,
        message: ChatCompletionMessage,
    ) -> str | None:
        for field in cls._THINKING_FIELDS:
            value = getattr(message, field, None)

            if isinstance(value, str) and value.strip():
                return value

        return None

    @classmethod
    def map_tool_calls(
        cls,
        tool_calls: list[ChatCompletionMessageToolCallUnion] | None,
    ) -> list[ToolCall] | None:
        if tool_calls is None:
            return None

        return [cls._map_tool_call(tool_call) for tool_call in tool_calls]

    @classmethod
    def _map_tool_call(cls, tool_call: ChatCompletionMessageToolCallUnion) -> ToolCall:
        if isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
            return cls._map_function_tool_call(tool_call)

        if isinstance(tool_call, ChatCompletionMessageCustomToolCall):
            return cls._map_custom_tool_call(tool_call)

        raise TypeError(
            f"Unsupported OpenAI tool call type: {type(tool_call).__name__}"
        )

    @classmethod
    def _map_function_tool_call(
        cls, tool_call: ChatCompletionMessageFunctionToolCall
    ) -> FunctionToolCall:
        return FunctionToolCall(
            id=tool_call.id,
            function=FunctionCall(
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            ),
            type="function",
        )

    @classmethod
    def _map_custom_tool_call(
        cls, tool_call: ChatCompletionMessageCustomToolCall
    ) -> CustomToolCall:
        return CustomToolCall(
            id=tool_call.id,
            custom=CustomCall(
                name=tool_call.custom.name,
                input=tool_call.custom.input,
            ),
            type="custom",
        )
