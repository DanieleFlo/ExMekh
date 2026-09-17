from __future__ import annotations

import os
from typing import Literal
from urllib.parse import urlparse

import openai
from openai import OpenAI
from pydantic import BaseModel

from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.reasoning_effort import ReasoningEffort
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.core.providers.chat_completion_openai import ChatCompletionOpenAI
from qs.core.providers.response_openai import ResponseOpenAI
from qs.interfaces.provider import Provider
from qs.interfaces.tokenizer import Tokenizer
from qs.tokenizers.qwen import QwenTokenizer


class LocalProvider(Provider):
    def __init__(
        self,
        model: str,
        api_key_env_var: str = "API_KEY",
        base_url_env_var: str = "BASE_URL",
        reasoning_effort: ReasoningEffort | None = None,
        api: Literal["chat_completions", "responses"] = "chat_completions",
        tokenizer: Tokenizer | None = None,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer if tokenizer is not None else QwenTokenizer()
        )
        self._model = model
        self._api_key_env_var = api_key_env_var
        self._base_url_env_var = base_url_env_var
        self._reasoning_effort = reasoning_effort
        self._api = api

        self._check_inputs()
        self.client = self._build_client()
        self._check_provider()

    def _check_inputs(self) -> None:
        if not isinstance(self._model, str):
            raise TypeError("Model must be a string")
        if not isinstance(self._api_key_env_var, str):
            raise TypeError("API key environment variable must be a string")
        if not isinstance(self._base_url_env_var, str):
            raise TypeError("Base URL environment variable must be a string")
        if self._reasoning_effort is not None and not isinstance(
            self._reasoning_effort, ReasoningEffort
        ):
            raise TypeError("Reasoning effort must be a ReasoningEffort or None")
        if self._api not in {"chat_completions", "responses"}:
            raise ValueError("API must be 'chat_completions' or 'responses'")

        if not self._model.strip():
            raise ValueError("Model must be a non-empty string")
        if not self._api_key_env_var.strip():
            raise ValueError("API key environment variable must be a non-empty string")
        if not self._base_url_env_var.strip():
            raise ValueError("Base URL environment variable must be a non-empty string")

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
        if not base_url or not base_url.strip():
            raise ValueError(
                f"Environment variable '{self._base_url_env_var}' is not set."
            )
        parsed_url = urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("Base URL must be a valid HTTP(S) URL")
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
        model_data = getattr(model_list, "data", None)
        if model_data is None:
            raise RuntimeError("Provider returned a malformed model catalog")
        if not isinstance(model_data, list):
            raise TypeError("Provider model catalog data must be a list")

        for model in model_data:
            model_id = getattr(model, "id", None)
            if not isinstance(model_id, str) or not model_id.strip():
                raise RuntimeError("Provider returned a malformed model identifier")
            available_models.add(model_id)

        return available_models

    def call(self, request: ProviderRequest) -> ProviderResponse:
        try:
            api_adapter = self._create_api_adapter()
            return api_adapter.call(request)
        except openai.APIError as error:
            raise RuntimeError("Local provider request failed") from error

    def _create_api_adapter(self) -> ChatCompletionOpenAI | ResponseOpenAI:
        if self._api == "chat_completions":
            return ChatCompletionOpenAI(
                client=self.client,
                model=self._model,
                reasoning_effort=self._reasoning_effort,
            )
        elif self._api == "responses":
            return ResponseOpenAI(
                client=self.client,
                model=self._model,
                reasoning_effort=self._reasoning_effort,
            )
        else:
            raise ValueError("API must be 'chat_completions' or 'responses'")

    def call_structured[ResponseModel: BaseModel](
        self,
        request: StructuredProviderRequest[ResponseModel],
    ) -> StructuredProviderResponse[ResponseModel]:
        try:
            api_adapter = self._create_api_adapter()
            return api_adapter.call_structured(request)
        except openai.APIError as error:
            raise RuntimeError("Local provider request failed") from error
