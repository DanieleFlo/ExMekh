import pytest

from qs.core.models.history import History
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import UserMessage


def test_get_request_adds_single_message_to_history() -> None:
    system_message = SystemMessage("rules")
    user_message = UserMessage("question")
    request = ProviderRequest(user_message, History([system_message]))

    result = request.get_request()

    assert result.get_history() == [system_message, user_message]


def test_get_request_adds_message_list_to_history() -> None:
    system_message = SystemMessage("rules")
    messages = [UserMessage("first"), UserMessage("second")]
    request = ProviderRequest(messages, History([system_message]))

    result = request.get_request()

    assert result.get_history() == [system_message, *messages]


@pytest.mark.parametrize("messages", [[], ""])
def test_get_request_rejects_empty_messages(messages) -> None:
    request = ProviderRequest(messages, History([SystemMessage("rules")]))

    with pytest.raises(ValueError, match="Messages cannot be empty"):
        request.get_request()


def test_get_request_rejects_non_message_list_items() -> None:
    request = ProviderRequest(
        [UserMessage("question"), object()],
        History([SystemMessage("rules")]),
    )

    with pytest.raises(
        TypeError, match="All items in messages must be instances of Message"
    ):
        request.get_request()


def test_get_request_rejects_unsupported_type() -> None:
    request = ProviderRequest(42, History([SystemMessage("rules")]))

    with pytest.raises(
        TypeError,
        match="Messages must be an instance of Message or a list of Message instances",
    ):
        request.get_request()
