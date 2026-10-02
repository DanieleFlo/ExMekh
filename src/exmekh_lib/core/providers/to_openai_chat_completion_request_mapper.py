from __future__ import annotations

from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.function_message import FunctionMessage
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.reasoning_effort import ReasoningEffort
from exmekh_lib.core.models.structured_assistant_message import (
    StructuredAssistantMessage,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.providers.openai_chat_completion_tool_mapper import (
    OpenAIChatCompletionToolMapper,
)
from exmekh_lib.core.providers.to_openai_chat_completion_mapper import (
    ToOpenAIChatCompletionMapper,
)
from exmekh_lib.interfaces.message import Message


class ToOpenAIChatCompletionRequestMapper:
    @classmethod
    def map_request(
        cls,
        request: ProviderRequest,
        *,
        model: str,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> dict[str, Any]:
        if not isinstance(request, ProviderRequest):
            raise TypeError("Expected a ProviderRequest instance")

        payload: dict[str, Any] = {
            "model": model,
            "messages": cls._map_messages(request.get_messages()),
            "stream": request.get_streaming(),
        }

        if request.get_streaming():
            payload["stream_options"] = {"include_usage": True}

        reasoning_effort = request.get_reasoning_effort_override() or reasoning_effort
        if reasoning_effort is not None:
            payload["reasoning_effort"] = cls._get_reasoning_effort_value(
                reasoning_effort
            )

        extra = request.get_extra()
        if extra is not None:
            payload["extra_body"] = extra

        tool_choice = request.get_tool_choice()
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        tools = request.get_tools()
        if tools is not None:
            mapped_tools = OpenAIChatCompletionToolMapper.map_tools(tools)
            if mapped_tools:
                payload["tools"] = mapped_tools

        return payload

    @classmethod
    def _get_reasoning_effort_value(cls, reasoning_effort: ReasoningEffort) -> str:
        if isinstance(reasoning_effort, str):
            return reasoning_effort
        if isinstance(reasoning_effort, ReasoningEffort):
            return reasoning_effort.value
        raise TypeError(
            "Reasoning effort must be a string or an instance of ReasoningEffort"
        )

    @classmethod
    def _map_messages(
        cls,
        messages: list[Message[object]],
    ) -> list[ChatCompletionMessageParam]:
        return [cls._map_message(message) for message in messages]

    @staticmethod
    def _map_message(message: Message[object]) -> ChatCompletionMessageParam:
        if isinstance(message, SystemMessage):
            return ToOpenAIChatCompletionMapper.map_system_message(message)
        if isinstance(message, UserMessage):
            return ToOpenAIChatCompletionMapper.map_user_message(message)
        if isinstance(message, AssistantMessage):
            return ToOpenAIChatCompletionMapper.map_assistant_message(message)
        if isinstance(message, StructuredAssistantMessage):
            return ToOpenAIChatCompletionMapper.map_structured_assistant_message(
                message
            )
        if isinstance(message, ToolMessage):
            return ToOpenAIChatCompletionMapper.map_tool_message(message)
        if isinstance(message, FunctionMessage):
            return ToOpenAIChatCompletionMapper.map_function_message(message)
        raise TypeError(f"Unsupported message type: {type(message)}")
