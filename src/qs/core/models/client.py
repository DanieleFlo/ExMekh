from __future__ import annotations

from qs.core.models.history import MessageHistory
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.providers.provider_mapper import ProviderMapper
from qs.interfaces.message import Message
from qs.interfaces.provider import Provider
from qs.interfaces.provider_config import ProviderConfig


class Client:
    def __init__(self, config_provider: ProviderConfig) -> None:
        self.config_provider = config_provider
        self._history = MessageHistory()
        self._provider = ProviderMapper.provider(config_provider=config_provider)

        self.validate_provider()

    def validate_provider(self) -> None:
        if not isinstance(self._provider, Provider):
            raise TypeError(
                f"Invalid provider: {self._provider}. Expected a Provider instance."
            )

    def get_history(self) -> list[Message]:
        return self._history.get_history()

    def dump_history(self) -> list[dict[str, str]]:
        return self._history.dump_history()

    @staticmethod
    def _normalize_messages(messages: Message | list[Message]) -> list[Message]:
        if isinstance(messages, Message):
            return [messages]

        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("Expected a Message or a list of Messages.")

        if not all(isinstance(message, Message) for message in messages):
            raise ValueError("Expected a Message or a list of Messages.")

        return messages

    def call(self, messages: Message | list[Message]) -> ProviderResponse:
        request_messages = self._normalize_messages(messages)

        request = ProviderRequest(history=self._history, messages=messages)
        response = self._provider.call(request)

        self._history.add_exchange(
            request_messages=request_messages,
            response_message=response.response,
        )
        return response
