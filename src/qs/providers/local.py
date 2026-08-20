from __future__ import annotations

import os

import openai
from openai import OpenAI

from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.providers.chat_completion_openai import ChatCompletionOpenAI
from qs.interfaces.provider import Provider


class LocalProvider(Provider):
    def __init__(
        self,
        model: str,
        api_key_env_var: str = "API_KEY",
        base_url_env_var: str = "BASE_URL",
    ) -> None:
        self._model = model
        self._api_key_env_var = api_key_env_var
        self._base_url_env_var = base_url_env_var

        self.client = self._build_client()
        self._check_provider()

    def _build_client(self) -> OpenAI:
        api_key = self._resolve_api_key()
        base_url = self._resolve_base_url()
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client

    def _resolve_api_key(self) -> str:
        api_key = os.getenv(self._api_key_env_var)
        if not api_key:
            # Local provider does not require an API key
            api_key = "local-api-key"
        return api_key

    def _resolve_base_url(self) -> str:
        base_url = os.getenv(self._base_url_env_var)
        if not base_url:
            raise ValueError(
                f"Environment variable '{self._base_url_env_var}' is not set."
            )
        return base_url

    def _check_provider(self) -> None:
        if not self._model:
            raise ValueError("Model name is not set.")

        available_models = self._get_available_models()

        if not available_models:
            raise ValueError("No models are available from the provider.")

        if self._model not in available_models:
            raise ValueError(
                f"Model '{self._model}' not found. "
                f"Available models: {', '.join(sorted(available_models))}"
            )

    def _get_available_models(self) -> set[str]:
        available_models: set[str] = set()

        model_list = self.client.models.list()
        for model in model_list.data:
            model_id = model.id
            available_models.add(model_id)

        return available_models

    def call(self, request: ProviderRequest) -> ProviderResponse:
        try:
            chat_completion = ChatCompletionOpenAI(
                client=self.client, model=self._model
            )
            response = chat_completion.call(request)
            return response
        except openai.APIError:
            raise RuntimeError("Local provider request failed")
