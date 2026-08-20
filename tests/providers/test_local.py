from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import openai
import pytest

from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.providers.local import LocalProvider


def make_provider(model: str = "local-model") -> LocalProvider:
    provider = object.__new__(LocalProvider)
    provider._model = model
    provider._api_key_env_var = "API_KEY"
    provider._base_url_env_var = "BASE_URL"
    provider.client = MagicMock()
    return provider


def test_init_builds_client_and_checks_provider() -> None:
    client = MagicMock()

    with (
        patch.object(LocalProvider, "_build_client", return_value=client) as build,
        patch.object(LocalProvider, "_check_provider") as check,
    ):
        provider = LocalProvider(
            "local-model",
            api_key_env_var="CUSTOM_KEY",
            base_url_env_var="CUSTOM_URL",
        )

    assert provider._model == "local-model"
    assert provider._api_key_env_var == "CUSTOM_KEY"
    assert provider._base_url_env_var == "CUSTOM_URL"
    assert provider.client is client
    build.assert_called_once_with()
    check.assert_called_once_with()


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


def test_check_provider_rejects_empty_model() -> None:
    provider = make_provider(model="")

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


def test_call_delegates_to_chat_completion() -> None:
    provider = make_provider()
    request = MagicMock(spec=ProviderRequest)
    expected = ProviderResponse("answer")
    completion = MagicMock()
    completion.call.return_value = expected

    with patch(
        "qs.providers.local.ChatCompletionOpenAI", return_value=completion
    ) as completion_class:
        result = provider.call(request)

    assert result is expected
    completion_class.assert_called_once_with(
        client=provider.client, model="local-model"
    )
    completion.call.assert_called_once_with(request)


def test_call_translates_openai_api_error_to_runtime_error() -> None:
    provider = make_provider()
    request = MagicMock(spec=ProviderRequest)
    completion = MagicMock()
    completion.call.side_effect = openai.APIError(
        "provider failed", MagicMock(), body=None
    )

    with (
        patch("qs.providers.local.ChatCompletionOpenAI", return_value=completion),
        pytest.raises(RuntimeError, match="Local provider request failed"),
    ):
        provider.call(request)
