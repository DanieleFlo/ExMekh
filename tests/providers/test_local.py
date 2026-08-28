from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock, patch

import openai
import pytest
from pydantic import BaseModel

from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.reasoning_effort import (
    ReasoningEffort,
    ReasoningEffortLMH,
    ReasoningEffortLMHXM,
    ReasoningEffortLMX,
)
from qs.core.models.structured_assistant_message import StructuredAssistantMessage
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.core.providers.response_openai import ResponseOpenAI
from qs.providers.local import LocalProvider


class StructuredAnswer(BaseModel):
    value: int


def make_provider(
    model: str = "local-model",
    reasoning_effort: ReasoningEffort | None = None,
    api: Literal["chat_completions", "responses"] = "chat_completions",
) -> LocalProvider:
    with (
        patch.object(LocalProvider, "_build_client", return_value=MagicMock()),
        patch.object(LocalProvider, "_check_provider"),
    ):
        return LocalProvider(
            model=model,
            reasoning_effort=reasoning_effort,
            api=api,
        )


def test_init_builds_client_and_checks_provider() -> None:
    client = MagicMock()

    with (
        patch.object(LocalProvider, "_check_inputs") as check_inputs,
        patch.object(LocalProvider, "_build_client", return_value=client) as build,
        patch.object(LocalProvider, "_check_provider") as check_provider,
    ):
        provider = LocalProvider(
            model="local-model",
            api_key_env_var="CUSTOM_KEY",
            base_url_env_var="CUSTOM_URL",
        )

    assert provider._model == "local-model"
    assert provider._api_key_env_var == "CUSTOM_KEY"
    assert provider._base_url_env_var == "CUSTOM_URL"
    assert provider._reasoning_effort is None
    assert provider._api == "chat_completions"
    assert provider.client is client
    check_inputs.assert_called_once_with()
    build.assert_called_once_with()
    check_provider.assert_called_once_with()


@pytest.mark.parametrize(
    ("overrides", "error_type"),
    [
        ({"model": ""}, ValueError),
        ({"model": "   "}, ValueError),
        ({"model": None}, TypeError),
        ({"api_key_env_var": ""}, ValueError),
        ({"api_key_env_var": 7}, TypeError),
        ({"base_url_env_var": ""}, ValueError),
        ({"base_url_env_var": 7}, TypeError),
        ({"reasoning_effort": 7}, TypeError),
        ({"api": "unknown"}, ValueError),
    ],
)
def test_init_rejects_malformed_settings(overrides, error_type) -> None:
    values = {
        "model": "valid-model",
        "api_key_env_var": "VALID_KEY",
        "base_url_env_var": "VALID_URL",
        "reasoning_effort": ReasoningEffortLMH.LOW,
        "api": "chat_completions",
    }
    values.update(overrides)

    with pytest.raises(error_type):
        LocalProvider(**values)


@pytest.mark.parametrize(
    "effort",
    [
        ReasoningEffortLMX.XHIGH,
        ReasoningEffortLMH.HIGH,
        ReasoningEffortLMHXM.Max,
        "provider-specific-effort",
    ],
)
def test_init_accepts_supported_reasoning_efforts(effort: ReasoningEffort) -> None:
    provider = make_provider(reasoning_effort=effort)

    assert provider._reasoning_effort == effort


def test_build_client_uses_resolved_credentials() -> None:
    provider = make_provider()
    client = MagicMock()

    with (
        patch.object(provider, "_resolve_api_key", return_value="secret") as api_key,
        patch.object(
            provider, "_resolve_base_url", return_value="http://localhost:8000"
        ) as base_url,
        patch("qs.providers.local.OpenAI", return_value=client) as openai_client,
    ):
        result = provider._build_client()

    assert result is client
    api_key.assert_called_once_with()
    base_url.assert_called_once_with()
    openai_client.assert_called_once_with(
        api_key="secret", base_url="http://localhost:8000"
    )


def test_resolve_api_key_returns_configured_environment_value(monkeypatch) -> None:
    provider = make_provider()
    provider._api_key_env_var = "CUSTOM_KEY"
    monkeypatch.setenv("CUSTOM_KEY", "secret")

    assert provider._resolve_api_key() == "secret"


def test_resolve_api_key_returns_local_fallback_when_missing(monkeypatch) -> None:
    provider = make_provider()
    monkeypatch.delenv("API_KEY", raising=False)

    assert provider._resolve_api_key() == "local-api-key"


def test_resolve_base_url_returns_configured_environment_value(monkeypatch) -> None:
    provider = make_provider()
    provider._base_url_env_var = "CUSTOM_URL"
    monkeypatch.setenv("CUSTOM_URL", "http://localhost:8000")

    assert provider._resolve_base_url() == "http://localhost:8000"


def test_resolve_base_url_rejects_missing_environment_value(monkeypatch) -> None:
    provider = make_provider()
    provider._base_url_env_var = "MISSING_URL"
    monkeypatch.delenv("MISSING_URL", raising=False)

    with pytest.raises(
        ValueError, match="Environment variable 'MISSING_URL' is not set"
    ):
        provider._resolve_base_url()


@pytest.mark.parametrize("base_url", ["   ", "localhost:9000", "file:///tmp/socket"])
def test_resolve_base_url_rejects_malformed_or_unsafe_url(
    monkeypatch, base_url: str
) -> None:
    provider = make_provider()
    monkeypatch.setenv("BASE_URL", base_url)

    with pytest.raises(ValueError):
        provider._resolve_base_url()


def test_check_provider_rejects_empty_model() -> None:
    provider = make_provider()
    provider._model = ""

    with pytest.raises(ValueError, match="Model name is not set"):
        provider._check_provider()


def test_check_provider_rejects_empty_catalog() -> None:
    provider = make_provider()

    with (
        patch.object(provider, "_get_available_models", return_value=set()),
        pytest.raises(ValueError, match="No models are available from the provider"),
    ):
        provider._check_provider()


def test_check_provider_rejects_unknown_model_with_sorted_catalog() -> None:
    provider = make_provider(model="missing")

    with (
        patch.object(provider, "_get_available_models", return_value={"zeta", "alpha"}),
        pytest.raises(
            ValueError,
            match="Model 'missing' not found. Available models: alpha, zeta",
        ),
    ):
        provider._check_provider()


def test_check_provider_accepts_available_model() -> None:
    provider = make_provider(model="local-model")

    with patch.object(provider, "_get_available_models", return_value={"local-model"}):
        assert provider._check_provider() is None


def test_get_available_models_returns_unique_model_ids() -> None:
    provider = make_provider()
    provider.client.models.list.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(id="first"),
            SimpleNamespace(id="second"),
            SimpleNamespace(id="first"),
        ]
    )

    assert provider._get_available_models() == {"first", "second"}
    provider.client.models.list.assert_called_once_with()


@pytest.mark.parametrize(
    "catalog",
    [SimpleNamespace(data=None), SimpleNamespace(data=[SimpleNamespace(id=None)])],
)
def test_get_available_models_rejects_malformed_catalog(catalog) -> None:
    provider = make_provider()
    provider.client.models.list.return_value = catalog

    with pytest.raises(RuntimeError, match="malformed"):
        provider._get_available_models()


def test_get_available_models_rejects_catalog_with_wrong_collection_type() -> None:
    provider = make_provider()
    provider.client.models.list.return_value = SimpleNamespace(data=("model",))

    with pytest.raises(TypeError, match="catalog data must be a list"):
        provider._get_available_models()


def test_call_delegates_to_chat_completion() -> None:
    provider = make_provider(reasoning_effort=ReasoningEffortLMH.HIGH)
    request = MagicMock(spec=ProviderRequest)
    expected = ProviderResponse(AssistantMessage("answer"))
    completion = MagicMock()
    completion.call.return_value = expected

    with patch(
        "qs.providers.local.ChatCompletionOpenAI", return_value=completion
    ) as completion_class:
        result = provider.call(request)

    assert result is expected
    completion_class.assert_called_once_with(
        client=provider.client,
        model="local-model",
        reasoning_effort=ReasoningEffortLMH.HIGH,
    )
    completion.call.assert_called_once_with(request)


def test_call_delegates_to_responses_api() -> None:
    provider = make_provider(
        reasoning_effort=ReasoningEffortLMH.HIGH,
        api="responses",
    )
    request = ProviderRequest([AssistantMessage("question")])
    expected = ProviderResponse(AssistantMessage("answer"))
    completion = MagicMock(spec=ResponseOpenAI)
    completion.call.return_value = expected

    with patch(
        "qs.providers.local.ResponseOpenAI",
        return_value=completion,
    ) as completion_class:
        result = provider.call(request)

    assert result is expected
    completion_class.assert_called_once_with(
        client=provider.client,
        model="local-model",
        reasoning_effort=ReasoningEffortLMH.HIGH,
    )
    completion.call.assert_called_once_with(request)


def test_create_api_adapter_rejects_corrupted_api() -> None:
    provider = make_provider()
    provider._api = "unknown"

    with pytest.raises(ValueError, match="API must be"):
        provider._create_api_adapter()


def test_call_translates_openai_api_error_to_runtime_error() -> None:
    provider = make_provider()
    request = MagicMock(spec=ProviderRequest)
    completion = MagicMock()
    completion.call.side_effect = openai.APIError(
        "provider failed", MagicMock(), body=None
    )

    with (
        patch("qs.providers.local.ChatCompletionOpenAI", return_value=completion),
        pytest.raises(RuntimeError, match="Local provider request failed") as error,
    ):
        provider.call(request)

    assert error.value.__cause__ is completion.call.side_effect


def test_call_structured_delegates_to_chat_completion() -> None:
    provider = make_provider(reasoning_effort="high")
    request = MagicMock(spec=StructuredProviderRequest)
    expected = StructuredProviderResponse(
        StructuredAssistantMessage(StructuredAnswer(value=42)),
    )
    completion = MagicMock()
    completion.call_structured.return_value = expected

    with patch(
        "qs.providers.local.ChatCompletionOpenAI", return_value=completion
    ) as completion_class:
        result = provider.call_structured(request)

    assert result is expected
    completion_class.assert_called_once_with(
        client=provider.client,
        model="local-model",
        reasoning_effort="high",
    )
    completion.call_structured.assert_called_once_with(request)


def test_call_structured_delegates_to_responses_api() -> None:
    provider = make_provider(reasoning_effort="high", api="responses")
    request = StructuredProviderRequest(
        [AssistantMessage("question")],
        StructuredAnswer,
    )
    expected = StructuredProviderResponse(
        StructuredAssistantMessage(StructuredAnswer(value=42)),
    )
    completion = MagicMock(spec=ResponseOpenAI)
    completion.call_structured.return_value = expected

    with patch(
        "qs.providers.local.ResponseOpenAI",
        return_value=completion,
    ) as completion_class:
        result = provider.call_structured(request)

    assert result is expected
    completion_class.assert_called_once_with(
        client=provider.client,
        model="local-model",
        reasoning_effort="high",
    )
    completion.call_structured.assert_called_once_with(request)


def test_call_structured_translates_openai_api_error_to_runtime_error() -> None:
    provider = make_provider()
    request = MagicMock(spec=StructuredProviderRequest)
    completion = MagicMock()
    completion.call_structured.side_effect = openai.APIError(
        "provider failed", MagicMock(), body=None
    )

    with (
        patch("qs.providers.local.ChatCompletionOpenAI", return_value=completion),
        pytest.raises(RuntimeError, match="Local provider request failed") as error,
    ):
        provider.call_structured(request)

    assert error.value.__cause__ is completion.call_structured.side_effect
