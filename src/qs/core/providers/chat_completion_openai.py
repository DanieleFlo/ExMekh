from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion, Choice

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
    def __init__(self, client: OpenAI, model: str, number_of_choices: int = 1) -> None:
        self._client = client
        self._model = model
        self._number_of_choices = number_of_choices

    def call(self, request: ProviderRequest) -> ProviderResponse:
        response = self._execute_request(request)
        return self._parse_response(response)

    def _execute_request(self, request: ProviderRequest) -> ChatCompletion:
        return self._client.chat.completions.create(
            model=self._model,
            messages=self._map_request_to_provider_format(request),
            n=self._number_of_choices,
            stream=False,
        )

    def _parse_response(self, response: ChatCompletion) -> ProviderResponse:
        if not response.choices:
            raise RuntimeError("No choices returned from the provider")

        response_content = self._extract_content(response.choices)
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

    def _extract_content(self, choices: Sequence[Choice]) -> Message:
        # For simplicity, return the content of the first choice
        return self._content_from(choices[0])

    @classmethod
    def _content_from(cls, choice: Choice) -> Message:
        # TODO: La risposta deve diventare un message. Di che tipi? AssistantMessage, ToolMessage, FunctionMessage
        if choice.message is None:
            raise RuntimeError("Choice message is None")
        # TODO
        return cls.resolve_message_type(choice.message)

    @classmethod
    def resolve_message_type(cls, message: ChatCompletionMessage) -> Message:
        mapped_tool_calls = FromOpenAIChatCompletionMapper.map_tool_calls(
            message.tool_calls
        )

        return AssistantMessage(content=message.content, tool_calls=mapped_tool_calls)
