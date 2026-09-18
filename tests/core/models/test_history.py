from dataclasses import dataclass
from unittest.mock import patch

import pytest

from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.function_message import FunctionMessage
from qs.core.models.history import MessageHistory
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool_message import ToolMessage
from qs.core.models.user_message import UserMessage
from qs.interfaces.message import Message


@dataclass
class NotAMessage:
    content: str


class AlternativeSystemMessage(Message[str]):
    def get_message(self) -> str:
        return "alternative rules"

    def get_role_name(self) -> str:
        return "system"

    def message_dump(self) -> dict:
        return {"role": "system", "content": self.get_message()}


def test_init_creates_empty_history_by_default() -> None:
    history = MessageHistory()

    assert history.get_history() == []


def test_init_preserves_supplied_message_list() -> None:
    messages = [SystemMessage("rules")]

    history = MessageHistory(messages)

    assert history.get_history() == messages
    assert history.get_history() is not messages


def test_init_and_getter_isolate_history_from_external_mutation() -> None:
    messages = [SystemMessage("non-default-rules")]
    history = MessageHistory(messages)

    messages.append(UserMessage("injected through source list"))
    leaked_history = history.get_history()
    leaked_history.clear()

    assert history.get_history() == [messages[0]]


def test_add_dispatches_single_message() -> None:
    history = MessageHistory()
    message = SystemMessage("rules")
    expected = MessageHistory([message])

    with patch.object(history, "_add_single_message", return_value=expected) as add:
        result = history.add(message)

    assert result is expected
    add.assert_called_once_with(message)


def test_add_dispatches_message_list() -> None:
    history = MessageHistory()
    messages = [SystemMessage("rules"), UserMessage("question")]
    expected = MessageHistory(messages)

    with patch.object(history, "_add_multiple_messages", return_value=expected) as add:
        result = history.add(messages)

    assert result is expected
    add.assert_called_once_with(messages)


@pytest.mark.parametrize("item", ["message", 7, [], [SystemMessage("rules"), object()]])
def test_add_rejects_unsupported_items(item) -> None:
    with pytest.raises(
        TypeError,
        match="Item must be an instance of Message or a list of Message instances",
    ):
        MessageHistory().add(item)


def test_add_single_message_appends_validates_and_returns_history() -> None:
    history = MessageHistory()
    message = SystemMessage("rules")

    result = history._add_single_message(message)

    assert history.get_history() == [message]
    assert result.get_history() == history.get_history()


def test_add_multiple_messages_extends_validates_and_returns_history() -> None:
    system_message = SystemMessage("rules")
    user_message = UserMessage("question")
    history = MessageHistory()

    result = history._add_multiple_messages([system_message, user_message])

    assert history.get_history() == [system_message, user_message]
    assert result.get_history() == history.get_history()


def test_add_exchange_records_request_and_response_together() -> None:
    system_message = SystemMessage("exchange-rules")
    user_message = UserMessage("exchange-question")
    response_message = AssistantMessage("exchange-answer")
    history = MessageHistory()

    result = history.add([system_message, user_message])
    result = history.add(response_message)

    assert history.get_history() == [system_message, user_message, response_message]
    assert result.get_history() == history.get_history()


def test_invalid_add_leaves_existing_history_unchanged() -> None:
    system_message = SystemMessage("stable-rules")
    history = MessageHistory([system_message])

    with pytest.raises(TypeError, match="list of Message instances"):
        history.add([UserMessage("valid-looking"), object()])

    assert history.get_history() == [system_message]


def test_check_history_rejects_empty_or_corrupted_state() -> None:
    with pytest.raises(ValueError, match="History cannot be empty"):
        MessageHistory().check_history()

    corrupted = MessageHistory()
    corrupted._history = [object()]
    with pytest.raises(TypeError, match="Item must be an instance of Message"):
        corrupted.check_history()


def test_add_rejects_non_system_first_message_atomically() -> None:
    history = MessageHistory()

    with pytest.raises(TypeError, match="First item in history"):
        history.add(UserMessage("attempted injection"))

    assert history.get_history() == []


def test_history_accepts_system_role_without_concrete_system_message() -> None:
    message = AlternativeSystemMessage()
    history = MessageHistory()

    history.add(message)

    assert history.get_history() == [message]


def test_check_history_accepts_all_supported_transitions(caplog) -> None:
    history = MessageHistory(
        [
            SystemMessage("rules"),
            UserMessage("question"),
            AssistantMessage("tool request"),
            ToolMessage("tool result", "call-id"),
            ToolMessage("parallel tool result", "parallel-call-id"),
            AssistantMessage("function request"),
            FunctionMessage("function result", "calculate"),
            AssistantMessage("answer"),
            UserMessage("follow-up"),
        ]
    )

    history.check_history()

    assert caplog.messages == []


def test_check_history_warns_for_invalid_transition_and_tolerates_unknown_role(
    caplog,
) -> None:
    unknown = UserMessage("role tampering")
    unknown._role = "intruder"
    history = MessageHistory(
        [
            SystemMessage("rules"),
            AssistantMessage("out of order"),
            unknown,
            UserMessage("accepted after unknown role"),
        ]
    )

    history.check_history()

    assert caplog.messages == [
        "Wrong message sequence: system -> assistant",
        "Wrong message sequence: assistant -> intruder",
    ]


def test_dump_history_serializes_each_message_in_order() -> None:
    history = MessageHistory(
        [SystemMessage("custom rules"), UserMessage("custom question")]
    )

    assert history.dump_history() == [
        {"role": "system", "content": "custom rules"},
        {"role": "user", "content": "custom question"},
    ]


def test_replace_from_compacts_history_and_validates_result() -> None:
    opening = AssistantMessage("open skill")
    history = MessageHistory(
        [
            SystemMessage("rules"),
            UserMessage("question"),
            opening,
            ToolMessage("skill instructions", "skill-call"),
            AssistantMessage("exit skill"),
        ]
    )
    result = ToolMessage("skill result", "skill-call", "skill")

    history.replace_from(3, [result])

    assert history.get_history() == [
        history.get_history()[0],
        history.get_history()[1],
        opening,
        result,
    ]


@pytest.mark.parametrize("start_index", [-1, 2])
def test_replace_from_rejects_out_of_bounds_index(start_index: int) -> None:
    history = MessageHistory([SystemMessage("rules")])

    with pytest.raises(IndexError, match="outside the history"):
        history.replace_from(start_index, [])


@pytest.mark.parametrize("start_index", [True, "0"])
def test_replace_from_rejects_non_integer_index(start_index: object) -> None:
    history = MessageHistory([SystemMessage("rules")])

    with pytest.raises(TypeError, match="start_index must be an integer"):
        history.replace_from(start_index, [])
