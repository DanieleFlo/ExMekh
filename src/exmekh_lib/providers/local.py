from __future__ import annotations

import os
from types import TracebackType
from typing import Literal, Self
from urllib.parse import urlparse

import openai
from openai import OpenAI
from pydantic import BaseModel

from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.config.provider_text import PROVIDER_TEXT
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.reasoning_effort import ReasoningEffort
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.providers.chat_completion_openai import ChatCompletionOpenAI
from exmekh_lib.core.providers.interactive_openai import InteractiveResponseOpenAI
from exmekh_lib.core.providers.response_openai import ResponseOpenAI
from exmekh_lib.interfaces.provider import Provider, ProviderRequestRejectedError
from exmekh_lib.interfaces.tokenizer import Tokenizer
from exmekh_lib.tokenizers.qwen import QwenTokenizer


class LocalProvider(Provider):
    def __init__(
        self,
        model: str,
        api_key_env_var: str = DEFAULT_PROVIDER_SETTINGS.api_key_env_var,
        base_url_env_var: str = DEFAULT_PROVIDER_SETTINGS.base_url_env_var,
        reasoning_effort: ReasoningEffort | None = None,
        api: Literal["chat_completions", "responses"] = (
            DEFAULT_PROVIDER_SETTINGS.local_provider_default_api
        ),
        tokenizer: Tokenizer | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer if tokenizer is not None else QwenTokenizer()
        )
        self._model = model
        self._api_key_env_var = api_key_env_var
        self._base_url_env_var = base_url_env_var
        self._api_key = api_key
        self._base_url = base_url
        self._reasoning_effort = reasoning_effort
        self._api = api

        self._check_inputs()
        self.client = self._build_client()
        try:
            self._check_provider()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release the connection pool owned by this provider."""
        self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _check_inputs(self) -> None:
        if not isinstance(self._model, str):
            raise TypeError(PROVIDER_TEXT.model_must_be_a_string)
        if not isinstance(self._api_key_env_var, str):
            raise TypeError(PROVIDER_TEXT.api_key_environment_variable_must_be_a_string)
        if not isinstance(self._base_url_env_var, str):
            raise TypeError(
                PROVIDER_TEXT.base_url_environment_variable_must_be_a_string
            )
        if self._api_key is not None and not isinstance(self._api_key, str):
            raise TypeError(PROVIDER_TEXT.api_key_must_be_a_string_or_none)
        if self._base_url is not None and not isinstance(self._base_url, str):
            raise TypeError(PROVIDER_TEXT.base_url_must_be_a_string_or_none)
        if self._reasoning_effort is not None and not isinstance(
            self._reasoning_effort, ReasoningEffort
        ):
            raise TypeError(
                PROVIDER_TEXT.reasoning_effort_must_be_a_reasoningeffort_or_none
            )
        if self._api not in {"chat_completions", "responses"}:
            raise ValueError(PROVIDER_TEXT.api_must_be_chat_completions_or_responses)

        if not self._model.strip():
            raise ValueError(PROVIDER_TEXT.model_must_be_a_non_empty_string)
        if not self._api_key_env_var.strip():
            raise ValueError(
                PROVIDER_TEXT.api_key_environment_variable_must_be_a_non_empty
            )
        if not self._base_url_env_var.strip():
            raise ValueError(
                PROVIDER_TEXT.base_url_environment_variable_must_be_a_non_empty
            )

    def _build_client(self) -> OpenAI:
        api_key = self._resolve_api_key()
        base_url = self._resolve_base_url()
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client

    def _resolve_api_key(self) -> str:
        api_key = (
            self._api_key
            if self._api_key is not None
            else os.getenv(self._api_key_env_var)
        )
        if not api_key:
            # Local provider does not require an API key
            api_key = DEFAULT_PROVIDER_SETTINGS.fallback_api_key
        return api_key

    def _resolve_base_url(self) -> str:
        base_url = (
            self._base_url
            if self._base_url is not None
            else os.getenv(self._base_url_env_var)
        )
        if not base_url or not base_url.strip():
            raise ValueError(
                PROVIDER_TEXT.environment_variable_is_not_set.format(
                    base_url_env_var=self._base_url_env_var
                )
            )
        parsed_url = urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError(PROVIDER_TEXT.base_url_must_be_a_valid_http_s_url)
        return base_url

    def _check_provider(self) -> None:
        if not self._model:
            raise ValueError(PROVIDER_TEXT.model_name_is_not_set)

        available_models = self._get_available_models()

        if not available_models:
            raise ValueError(PROVIDER_TEXT.no_models_are_available_from_the_provider)

        if self._model not in available_models:
            raise ValueError(
                PROVIDER_TEXT.model_not_found_available_models.format(
                    model=self._model, names=", ".join(sorted(available_models))
                )
            )

    def _get_available_models(self) -> set[str]:
        available_models: set[str] = set()

        model_list = self.client.models.list()
        model_data = getattr(model_list, "data", None)
        if model_data is None:
            raise RuntimeError(
                PROVIDER_TEXT.provider_returned_a_malformed_model_catalog
            )
        if not isinstance(model_data, list):
            raise TypeError(PROVIDER_TEXT.provider_model_catalog_data_must_be_a_list)

        for model in model_data:
            model_id = getattr(model, "id", None)
            if not isinstance(model_id, str) or not model_id.strip():
                raise RuntimeError(
                    PROVIDER_TEXT.provider_returned_a_malformed_model_identifier
                )
            available_models.add(model_id)

        return available_models

    def call(self, request: ProviderRequest) -> ProviderResponse:
        try:
            api_adapter = self._create_api_adapter()
            if self._has_interactive_tools(request):
                if self._api != "responses":
                    raise ProviderRequestRejectedError(
                        PROVIDER_TEXT.interactive_tools_require_the_ds4_responses_v2_api
                    )
                api_adapter = InteractiveResponseOpenAI(
                    client=self.client,
                    model=self._model,
                    reasoning_effort=self._reasoning_effort,
                )
            return api_adapter.call(request)
        except openai.BadRequestError as error:
            raise ProviderRequestRejectedError(
                self._rejection_message(error)
            ) from error
        except openai.APIError as error:
            raise RuntimeError(PROVIDER_TEXT.local_provider_request_failed) from error

    @staticmethod
    def _rejection_message(error: openai.BadRequestError) -> str:
        if isinstance(error.body, dict):
            detail = error.body.get("error", error.body)
            if isinstance(detail, dict) and isinstance(detail.get("message"), str):
                return detail["message"]
        return error.message

    def get_tools_text(self, tools: Tools | None) -> str:
        return self._create_api_adapter().get_tools_text(tools)

    @staticmethod
    def _has_interactive_tools(request: ProviderRequest) -> bool:
        tools = request.get_tools()
        scope = (request.get_extra() or {}).get("agentic", {})
        return (
            tools is not None
            and isinstance(scope, dict)
            and any(
                tool.interactive is not None
                and tool.name in scope.get("allowed_tools", [])
                for tool in tools.get_tools()
            )
        )

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
            raise ValueError(PROVIDER_TEXT.api_must_be_chat_completions_or_responses)

    def call_structured[ResponseModel: BaseModel](
        self,
        request: StructuredProviderRequest[ResponseModel],
    ) -> StructuredProviderResponse[ResponseModel]:
        if self._has_interactive_tools(request):
            raise ProviderRequestRejectedError(
                PROVIDER_TEXT.structured_responses_do_not_support_interactive_tool_continuations
            )
        try:
            api_adapter = self._create_api_adapter()
            return api_adapter.call_structured(request)
        except openai.BadRequestError as error:
            raise ProviderRequestRejectedError(
                self._rejection_message(error)
            ) from error
        except openai.APIError as error:
            raise RuntimeError(PROVIDER_TEXT.local_provider_request_failed) from error
