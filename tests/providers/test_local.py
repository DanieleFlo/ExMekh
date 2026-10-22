import re
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock, patch

import httpx
import httpx2
import openai
import pytest
from pydantic import BaseModel

from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.reasoning_effort import (
    ReasoningEffort,
    ReasoningEffortLMH,
    ReasoningEffortLMHXM,
    ReasoningEffortLMX,
)
from exmekh_lib.core.models.structured_assistant_message import (
    StructuredAssistantMessage,
)
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.providers.response_openai import ResponseOpenAI
from exmekh_lib.interfaces.provider import ProviderRequestRejectedError
from exmekh_lib.providers.local import LocalProvider


class StructuredAnswer(BaseModel):
    value: int


def test_injected_http_client_serves_chat_discovery_and_native_requests():
    requests = []

    def handler(request):
        requests.append(request.url.path)
        if request.url.path.endswith("/models"):
            return httpx2.Response(200, json={"data": [{"id": "chat-model"}]})
        return httpx2.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0] * 128}]}
        )

    http_client = httpx2.Client(transport=httpx2.MockTransport(handler))
    with LocalProvider(
        model="chat-model",
        base_url="http://native/v1",
        api_key="test",
        http_client=http_client,
    ) as provider:
        assert (
            len(provider.native_suite().embeddings(["text"], dimensions=128)["data"])
            == 1
        )
        assert not http_client.is_closed
    assert http_client.is_closed
    assert requests == ["/v1/models", "/v1/embeddings"]


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
            api_key="explicit-key",
            base_url="http://explicit.test/v1",
        )

    assert provider._model == "local-model"
    assert provider._api_key_env_var == "CUSTOM_KEY"
    assert provider._base_url_env_var == "CUSTOM_URL"
    assert provider._reasoning_effort is None
    assert provider._api == "chat_completions"
    assert provider._api_key == "explicit-key"
    assert provider._base_url == "http://explicit.test/v1"
    assert provider.client is client
    check_inputs.assert_called_once_with()
    build.assert_called_once_with()
    check_provider.assert_called_once_with()


def test_failed_provider_validation_closes_the_created_client():
    client = MagicMock()
    with (
        patch.object(LocalProvider, "_build_client", return_value=client),
        patch.object(
            LocalProvider, "_check_provider", side_effect=RuntimeError("offline")
        ),
        pytest.raises(RuntimeError, match="offline"),
    ):
        LocalProvider(model="local-model")

    client.close.assert_called_once_with()


def test_provider_context_closes_the_client_when_the_body_fails():
    provider = make_provider()

    with pytest.raises(ValueError, match="stop"):
        with provider as active:
            assert active is provider
            raise ValueError("stop")

    provider.client.close.assert_called_once_with()


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
        ({"api_key": 7}, TypeError),
        ({"base_url": 7}, TypeError),
        ({"reasoning_effort": 7}, TypeError),
        ({"api": "unknown"}, ValueError),
    ],
)
def test_init_rejects_malformed_settings(overrides, error_type) -> None:
    values = {
        "model": "valid-model",
        "api_key_env_var": "VALID_KEY",
        "base_url_env_var": "VALID_URL",
        "api_key": None,
        "base_url": None,
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
        patch(
            "exmekh_lib.providers.local.OpenAI", return_value=client
        ) as openai_client,
    ):
        result = provider._build_client()

    assert result is client
    api_key.assert_called_once_with()
    base_url.assert_called_once_with()
    openai_client.assert_called_once_with(
        api_key="secret", base_url="http://localhost:8000", http_client=None
    )


def test_resolve_api_key_returns_configured_environment_value(monkeypatch) -> None:
    provider = make_provider()
    provider._api_key_env_var = "CUSTOM_KEY"
    monkeypatch.setenv("CUSTOM_KEY", "secret")

    assert provider._resolve_api_key() == "secret"


def test_resolve_api_key_prefers_explicit_value(monkeypatch) -> None:
    provider = make_provider()
    provider._api_key = "explicit"
    monkeypatch.setenv(DEFAULT_PROVIDER_SETTINGS.api_key_env_var, "environment")

    assert provider._resolve_api_key() == "explicit"


def test_resolve_api_key_explicit_empty_value_uses_local_fallback(monkeypatch) -> None:
    provider = make_provider()
    provider._api_key = ""
    monkeypatch.setenv(DEFAULT_PROVIDER_SETTINGS.api_key_env_var, "environment")

    assert provider._resolve_api_key() == DEFAULT_PROVIDER_SETTINGS.fallback_api_key


def test_resolve_api_key_returns_local_fallback_when_missing(monkeypatch) -> None:
    provider = make_provider()
    monkeypatch.delenv(DEFAULT_PROVIDER_SETTINGS.api_key_env_var, raising=False)

    assert provider._resolve_api_key() == DEFAULT_PROVIDER_SETTINGS.fallback_api_key


def test_resolve_base_url_returns_configured_environment_value(monkeypatch) -> None:
    provider = make_provider()
    provider._base_url_env_var = "CUSTOM_URL"
    monkeypatch.setenv("CUSTOM_URL", "http://localhost:8000")

    assert provider._resolve_base_url() == "http://localhost:8000"


def test_resolve_base_url_prefers_explicit_value(monkeypatch) -> None:
    provider = make_provider()
    provider._base_url = "http://explicit.test/v1"
    monkeypatch.setenv(
        DEFAULT_PROVIDER_SETTINGS.base_url_env_var, "http://environment.test/v1"
    )

    assert provider._resolve_base_url() == "http://explicit.test/v1"


def test_resolve_base_url_explicit_empty_value_does_not_use_environment(
    monkeypatch,
) -> None:
    provider = make_provider()
    provider._base_url = ""
    monkeypatch.setenv(
        DEFAULT_PROVIDER_SETTINGS.base_url_env_var, "http://environment.test/v1"
    )

    with pytest.raises(
        ValueError,
        match=rf"{re.escape(DEFAULT_PROVIDER_SETTINGS.base_url_env_var)}.*not set",
    ):
        provider._resolve_base_url()


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
    monkeypatch.setenv(DEFAULT_PROVIDER_SETTINGS.base_url_env_var, base_url)

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
            SimpleNamespace(id="embed", ds4={"operations": ["embed"]}),
            SimpleNamespace(id="image", ds4={"operations": ["image_generate"]}),
            SimpleNamespace(id="second", ds4={"operations": ["chat"]}),
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
        "exmekh_lib.providers.local.ChatCompletionOpenAI", return_value=completion
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
        "exmekh_lib.providers.local.ResponseOpenAI",
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
        patch(
            "exmekh_lib.providers.local.ChatCompletionOpenAI", return_value=completion
        ),
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
        "exmekh_lib.providers.local.ChatCompletionOpenAI", return_value=completion
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
        "exmekh_lib.providers.local.ResponseOpenAI",
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
        patch(
            "exmekh_lib.providers.local.ChatCompletionOpenAI", return_value=completion
        ),
        pytest.raises(RuntimeError, match="Local provider request failed") as error,
    ):
        provider.call_structured(request)

    assert error.value.__cause__ is completion.call_structured.side_effect


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
@pytest.mark.parametrize("structured", [False, True])
def test_bad_request_preserves_server_message_and_marks_request_as_rejected(
    api, structured
):
    provider = make_provider(api=api)
    message = "Only text and images are supported. Convert documents before sending."
    response = httpx.Response(400, request=httpx.Request("POST", "http://localhost/v1"))
    failure = openai.BadRequestError(
        message, response=response, body={"error": {"message": message}}
    )
    adapter = MagicMock()
    method = adapter.call_structured if structured else adapter.call
    method.side_effect = failure
    name = "ChatCompletionOpenAI" if api == "chat_completions" else "ResponseOpenAI"
    with (
        patch(f"exmekh_lib.providers.local.{name}", return_value=adapter),
        pytest.raises(
            ProviderRequestRejectedError, match="Only text and images"
        ) as error,
    ):
        if structured:
            provider.call_structured(MagicMock(spec=StructuredProviderRequest))
        else:
            provider.call(MagicMock(spec=ProviderRequest))
    assert str(error.value) == message
    assert error.value.__cause__ is failure


@pytest.mark.parametrize(
    "body,expected",
    [
        ({"message": "Use text or images"}, "Use text or images"),
        ({"error": {"message": "Use text or images"}}, "Use text or images"),
        ({"message": 123}, "SDK fallback"),
        ({"error": "unsupported"}, "SDK fallback"),
        ("invalid response body", "SDK fallback"),
        (None, "SDK fallback"),
    ],
)
def test_rejected_request_uses_readable_server_message_or_sdk_fallback(body, expected):
    provider = make_provider()
    response = httpx.Response(400, request=httpx.Request("POST", "http://localhost/v1"))
    failure = openai.BadRequestError("SDK fallback", response=response, body=body)
    adapter = MagicMock()
    adapter.call.side_effect = failure
    with (
        patch("exmekh_lib.providers.local.ChatCompletionOpenAI", return_value=adapter),
        pytest.raises(ProviderRequestRejectedError) as error,
    ):
        provider.call(MagicMock(spec=ProviderRequest))
    assert str(error.value) == expected
    assert error.value.__cause__ is failure
