from __future__ import annotations

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion

from qs.config.local_provider import ReasoningEffort
from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.function_message import FunctionMessage
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool_message import ToolMessage
from qs.core.models.user_message import UserMessage
from qs.core.providers.from_openai_chat_completion_mapper import (
    FromOpenAIChatCompletionMapper,
)
from qs.core.providers.to_openai_chat_completion_mapper import (
    ToOpenAIChatCompletionMapper,
)
from qs.interfaces.message import Message


class ChatCompletionOpenAI:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._reasoning_effort = reasoning_effort

    def call(self, request: ProviderRequest) -> ProviderResponse:
        response = self._execute_request(request)
        return self._parse_response(response)

    def _execute_request(self, request: ProviderRequest) -> ChatCompletion:
        if self._reasoning_effort is None:
            return self._client.chat.completions.create(
                model=self._model,
                messages=self._map_request_to_provider_format(request),
                stream=False,
            )
        else:
            return self._client.chat.completions.create(
                model=self._model,
                messages=self._map_request_to_provider_format(request),
                stream=False,
                reasoning_effort=self._reasoning_effort.value,
            )

    def _parse_response(self, response: ChatCompletion) -> ProviderResponse:
        if not response.choices:
            raise RuntimeError("No choices returned from the provider")

        response_content = FromOpenAIChatCompletionMapper.map_assistant_message(
            response.choices[0].message
        )
        usage = FromOpenAIChatCompletionMapper.extract_usage(response)
        thinking = FromOpenAIChatCompletionMapper.extract_thinking(response)

        return ProviderResponse(
            response=response_content, thinking=thinking, usage=usage
        )

    def _map_request_to_provider_format(
        self, request: ProviderRequest
    ) -> list[ChatCompletionMessageParam]:
        history = request.get_request()
        messages = history.get_history()
        return [self._map_message_to_provider_format(message) for message in messages]

    def _map_message_to_provider_format(
        self, message: Message
    ) -> ChatCompletionMessageParam:
        if isinstance(message, SystemMessage):
            return ToOpenAIChatCompletionMapper.map_system_message(message)
        elif isinstance(message, UserMessage):
            return ToOpenAIChatCompletionMapper.map_user_message(message)
        elif isinstance(message, AssistantMessage):
            return ToOpenAIChatCompletionMapper.map_assistant_message(message)
        elif isinstance(message, ToolMessage):
            return ToOpenAIChatCompletionMapper.map_tool_message(message)
        elif isinstance(message, FunctionMessage):
            return ToOpenAIChatCompletionMapper.map_function_message(message)
        else:
            raise TypeError(f"Unsupported message type: {type(message)}")
