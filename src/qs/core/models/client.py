from __future__ import annotations

from qs.core.models.history import History
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.providers.provider_mapper import ProviderMapper
from qs.interfaces.message import Message
from qs.interfaces.provider import Provider
from qs.interfaces.provider_config import ProviderConfig


class Client:
    def __init__(self, config_provider: ProviderConfig) -> None:
        self.config_provider = config_provider
        self._history = History()
        self._provider = ProviderMapper.provider(config_provider=config_provider)

        self.validate_provider()

    def validate_provider(self) -> None:
        if not isinstance(self._provider, Provider):
            raise TypeError(
                f"Invalid provider: {self._provider}. Expected a Provider instance."
            )


    def call(self, messages: Message | list[Message]) -> ProviderResponse:
        if not isinstance(messages, Message) and not isinstance(messages, list) and not all(isinstance(message, Message) for message in messages):
            raise ValueError(f"Invalid messages: {messages}. Expected a Message or a list of Messages.")

        request = ProviderRequest(history=self._history, messages=messages)
        response = self._provider.call(request)
        self._history.add(response.response)
        return response
