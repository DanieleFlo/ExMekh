from __future__ import annotations
from typing import List

from qs.config.provider import SupportedProviders
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.providers.provider_mapper import ProviderMapper
from qs.core.models.history import History
from qs.interfaces.message import Message

class Client:
    def __init__(self, provider: SupportedProviders) -> None:
        self._provider = provider
        self._history = History()
        self.client = ProviderMapper.provider(provider)

        self.validate_provider()

    def validate_provider(self):
        if self._provider not in SupportedProviders:
            raise ValueError(f"Invalid provider: {self._provider}. Expected one of: {[p for p in SupportedProviders]}")


    def call(self, messages: Message | List[Message]) -> ProviderResponse:
        if not isinstance(messages, Message) and not isinstance(messages, list) and not all(isinstance(message, Message) for message in messages):
            raise ValueError(f"Invalid messages: {messages}. Expected a Message or a list of Messages.")

        request = ProviderRequest(history=self._history, messages=messages)
        response = self.client.call(request)
        self._history.add(response.response)
        return response
