from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from qs.config.local_provider import LocalProviderConfig
from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.client import Client
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import UserMessage
from qs.core.providers.provider_mapper import ProviderMapper
from qs.interfaces.message import Message
from qs.interfaces.provider import Provider


def test_init_accepts_provider_returned_by_mapper() -> None:
    config = LocalProviderConfig(model="local-model")
    provider = MagicMock(spec=Provider)

    with patch.object(ProviderMapper, "provider", return_value=provider) as mapper:
        client = Client(config_provider=config)

    assert client.config_provider is config
    assert client._provider is provider
    mapper.assert_called_once_with(config_provider=config)


def test_init_rejects_non_provider_returned_by_mapper() -> None:
    config = LocalProviderConfig(model="local-model")

    with (
        patch.object(ProviderMapper, "provider", return_value=object()),
        pytest.raises(TypeError, match="Expected a Provider instance"),
    ):
        Client(config_provider=config)


@pytest.mark.parametrize(
    "malicious_input",
    [
        None,
        0,
        object(),
        "not-a-message",
        [],
        [SystemMessage("custom-rules"), object()],
    ],
)
def test_call_rejects_non_messages_without_contacting_provider(
    malicious_input,
) -> None:
    provider = MagicMock(spec=Provider)

    with patch.object(ProviderMapper, "provider", return_value=provider):
        client = Client(LocalProviderConfig(model="non-default-model"))

    with pytest.raises(ValueError, match="Expected a Message or a list of Messages"):
        client.call(malicious_input)

    provider.call.assert_not_called()


def test_call_sends_custom_messages_and_records_custom_response() -> None:
    provider = MagicMock(spec=Provider)
    response = ProviderResponse(AssistantMessage("non-default-answer"))
    provider.call.side_effect = lambda request: (request.get_request(), response)[1]
    messages = [SystemMessage("strict custom rules"), UserMessage("unusual request")]

    with patch.object(ProviderMapper, "provider", return_value=provider):
        client = Client(LocalProviderConfig(model="non-default-model"))

    result = client.call(messages)

    assert result is response
    assert client.get_history() == [*messages, response.response]
    assert client.dump_history() == [
        {"role": "system", "content": "strict custom rules"},
        {"role": "user", "content": "unusual request"},
        {"role": "assistant", "content": "non-default-answer"},
    ]
    provider.call.assert_called_once()


def test_call_normalizes_single_message_before_recording_response() -> None:
    provider = MagicMock(spec=Provider)
    message = SystemMessage("single-message-rules")
    response = ProviderResponse(AssistantMessage("single-message-answer"))
    provider.call.return_value = response

    with patch.object(ProviderMapper, "provider", return_value=provider):
        client = Client(LocalProviderConfig(model="single-message-model"))

    result = client.call(message)

    assert result is response
    assert client.get_history() == [message, response.response]


def test_call_provider_failure_leaves_history_unchanged() -> None:
    provider = MagicMock(spec=Provider)
    messages = [SystemMessage("stable-rules"), UserMessage("failing-request")]

    def fail_after_materializing_request(request):
        request.get_request()
        raise RuntimeError("simulated provider failure")

    provider.call.side_effect = fail_after_materializing_request

    with patch.object(ProviderMapper, "provider", return_value=provider):
        client = Client(LocalProviderConfig(model="failure-model"))

    with pytest.raises(RuntimeError, match="simulated provider failure"):
        client.call(messages)

    assert client.get_history() == []


def test_call_invalid_provider_response_leaves_history_unchanged() -> None:
    provider = MagicMock(spec=Provider)
    provider.call.return_value = ProviderResponse(cast(Message, object()))
    messages = [SystemMessage("stable-rules"), UserMessage("valid-request")]

    with patch.object(ProviderMapper, "provider", return_value=provider):
        client = Client(LocalProviderConfig(model="untrusted-response-model"))

    with pytest.raises(TypeError, match="list of Message instances"):
        client.call(messages)

    assert client.get_history() == []


def test_call_validation_does_not_evaluate_hostile_repr() -> None:
    class HostileRepresentation:
        def __repr__(self) -> str:
            raise RuntimeError("repr must not execute")

    provider = MagicMock(spec=Provider)
    with patch.object(ProviderMapper, "provider", return_value=provider):
        client = Client(LocalProviderConfig(model="repr-safe-model"))

    with pytest.raises(ValueError, match="Expected a Message"):
        client.call(cast(Message, HostileRepresentation()))

    provider.call.assert_not_called()
