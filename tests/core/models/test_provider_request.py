import pytest

from qs.core.models.history import MessageHistory
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import UserMessage


def test_get_request_adds_single_message_to_history() -> None:
    system_message = SystemMessage("rules")
    user_message = UserMessage("question")
    request = ProviderRequest(user_message, MessageHistory([system_message]))

    result = request.get_request()

    assert result.get_history() == [system_message, user_message]


def test_get_request_adds_message_list_to_history() -> None:
    system_message = SystemMessage("rules")
    messages = [UserMessage("first"), UserMessage("second")]
    request = ProviderRequest(messages, MessageHistory([system_message]))

    result = request.get_request()

    assert result.get_history() == [system_message, *messages]


def test_get_request_is_repeatable_without_mutating_source_history() -> None:
    system_message = SystemMessage("stable-rules")
    user_message = UserMessage("repeatable-question")
    source = MessageHistory([system_message])
    request = ProviderRequest(user_message, source)

    first = request.get_request()
    second = request.get_request()

    assert first.get_history() == [system_message, user_message]
    assert second.get_history() == first.get_history()
    assert source.get_history() == [system_message]


@pytest.mark.parametrize("messages", [[], ""])
def test_get_request_rejects_empty_messages(messages) -> None:
    request = ProviderRequest(messages, MessageHistory([SystemMessage("rules")]))

    with pytest.raises(ValueError, match="Messages cannot be empty"):
        request.get_request()


def test_get_request_rejects_non_message_list_items() -> None:
    request = ProviderRequest(
        [UserMessage("question"), object()],
        MessageHistory([SystemMessage("rules")]),
    )

    with pytest.raises(
        TypeError, match="All items in messages must be instances of Message"
    ):
        request.get_request()


def test_get_request_rejects_unsupported_type() -> None:
    request = ProviderRequest(42, MessageHistory([SystemMessage("rules")]))

    with pytest.raises(
        TypeError,
        match="Messages must be an instance of Message or a list of Message instances",
    ):
        request.get_request()
