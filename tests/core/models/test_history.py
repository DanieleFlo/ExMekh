from dataclasses import dataclass
from unittest.mock import patch

import pytest

from qs.core.models.history import History
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import UserMessage


@dataclass
class NotAMessage:
    content: str


def test_init_creates_empty_history_by_default() -> None:
    history = History()

    assert history.get_history() == []


def test_init_preserves_supplied_message_list() -> None:
    messages = [SystemMessage("rules")]

    history = History(messages)

    assert history.get_history() is messages


def test_add_dispatches_single_message() -> None:
    history = History()
    message = SystemMessage("rules")
    expected = History([message])

    with patch.object(history, "_add_single_message", return_value=expected) as add:
        result = history.add(message)

    assert result is expected
    add.assert_called_once_with(message)


def test_add_dispatches_message_list() -> None:
    history = History()
    messages = [SystemMessage("rules"), UserMessage("question")]
    expected = History(messages)

    with patch.object(history, "_add_multiple_messages", return_value=expected) as add:
        result = history.add(messages)

    assert result is expected
    add.assert_called_once_with(messages)


@pytest.mark.parametrize("item", ["message", 7, [SystemMessage("rules"), object()]])
def test_add_rejects_unsupported_items(item) -> None:
    with pytest.raises(
        TypeError,
        match="Item must be an instance of Message or a list of Message instances",
    ):
        History().add(item)


def test_add_single_message_appends_validates_and_returns_history() -> None:
    history = History()
    message = SystemMessage("rules")

    with patch.object(history, "check_history") as check_history:
        result = history._add_single_message(message)

    assert history.get_history() == [message]
    assert result.get_history() is history.get_history()
    check_history.assert_called_once_with()


def test_add_multiple_messages_extends_validates_and_returns_history() -> None:
    system_message = SystemMessage("rules")
    user_message = UserMessage("question")
    history = History()

    with patch.object(history, "check_history") as check_history:
        result = history._add_multiple_messages([system_message, user_message])

    assert history.get_history() == [system_message, user_message]
    assert result.get_history() is history.get_history()
    check_history.assert_called_once_with()


def test_get_history_returns_underlying_message_list() -> None:
    messages = [SystemMessage("rules")]

    assert History(messages).get_history() is messages
