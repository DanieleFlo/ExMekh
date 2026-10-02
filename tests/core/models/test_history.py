from dataclasses import dataclass
from unittest.mock import patch

import pytest

from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.function_message import FunctionMessage
from exmekh_lib.core.models.history import MessageHistory
from exmekh_lib.core.models.model_thinking import ModelThinking
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import (
    ErrorUserMessage,
    TextsUserMessage,
    UserMessage,
)
from exmekh_lib.interfaces.message import Message


@dataclass
class NotAMessage:
    content: str


def test_compaction_discards_delivered_skill_input():
    content = TextsUserMessage(["first"])
    history = MessageHistory(
        [
            SystemMessage("rules"),
            UserMessage("start"),
            AssistantMessage("skill"),
            UserMessage(content, name="author"),
            ErrorUserMessage("application feedback"),
            UserMessage("second"),
        ]
    )
    replacement = ToolMessage("skill result", "opening", "skill")

    history.replace_from(2, [replacement])
    content.texts.append("later mutation")
    assert [message.get_message() for message in history.get_history()] == [
        "rules",
        "start",
        "skill result",
    ]


@pytest.mark.parametrize("start_index", [-1, 10, True, "2"])
def test_compaction_rejects_invalid_index_without_changing_history(start_index):
    history = MessageHistory([SystemMessage("rules"), UserMessage("keep")])
    before = history.get_history()

    with pytest.raises((TypeError, IndexError)):
        history.replace_from(start_index, [])

    assert history.get_history() == before


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


def test_steering_messages_after_tools_and_each_other_are_valid(caplog) -> None:
    history = MessageHistory(
        [
            SystemMessage("rules"),
            UserMessage("initial"),
            AssistantMessage(
                None, tool_calls=[ToolCall("origin", FunctionCall("work", {}))]
            ),
            ToolMessage("running", "origin", "work"),
            UserMessage("first correction"),
            UserMessage("second correction"),
        ]
    )
    history.check_history()
    assert "Wrong message sequence" not in caplog.text


def test_init_preserves_supplied_message_list() -> None:
    messages = [SystemMessage("rules")]

    history = MessageHistory(messages)

    assert history.get_history() == messages
    assert history.get_history() is not messages


def test_latest_tool_batch_preserves_only_its_recorded_sibling_replies() -> None:
    previous = ToolCall("previous", FunctionCall("work", {}))
    first = ToolCall("first", FunctionCall("work", {}))
    second = ToolCall("second", FunctionCall("skill", {}))
    previous_reply = ToolMessage("old", "previous", "work")
    sibling_reply = ToolMessage("done", "first", "work")
    history = MessageHistory(
        [
            SystemMessage("rules"),
            UserMessage("start"),
            ProviderResponse(AssistantMessage(None, tool_calls=[previous])),
            previous_reply,
            UserMessage("correction"),
            ProviderResponse(AssistantMessage(None, tool_calls=[first, second])),
            sibling_reply,
        ]
    )

    batch = history.get_latest_tool_batch()
    assert batch.opening_index == 5
    assert batch.preceding_responses == (sibling_reply,)
    assert history.get_tool_continuation_messages() == history.get_history()
    returned = history.get_tool_continuation_messages()
    returned.clear()
    assert history.get_latest_tool_batch() == batch


@pytest.mark.parametrize(
    "messages",
    [[], [SystemMessage("rules")], [SystemMessage("rules"), AssistantMessage("done")]],
)
def test_tool_continuation_rejects_histories_without_a_final_tool_reply(messages):
    history = MessageHistory(messages)
    with pytest.raises(TypeError, match="last message must be a tool response"):
        history.get_tool_continuation_messages()


@pytest.mark.parametrize(
    "assistant",
    [None, AssistantMessage("done"), AssistantMessage("done", tool_calls=[])],
)
def test_latest_tool_batch_rejects_histories_without_a_generating_call(assistant):
    history = MessageHistory([SystemMessage("rules"), UserMessage("start")])
    if assistant is not None:
        history.add(assistant)
    with pytest.raises(ValueError, match="no assistant message|no tool calls"):
        history.get_latest_tool_batch()


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


@pytest.mark.parametrize("read_count", [1, 2])
def test_harness_reads_preserve_the_return_batch_and_provider_metadata(read_count):
    original = ToolCall("opening", FunctionCall("skill", {}))
    sibling = ToolCall("sibling", FunctionCall("work", {}))
    thinking = ModelThinking("opening reasoning")
    opening = ProviderResponse(
        AssistantMessage("open", name="agent", tool_calls=[sibling, original]),
        thinking=thinking,
    )
    sibling_reply = ToolMessage("work done", "sibling")
    return_reply = ToolMessage("skill done", "opening")
    history = MessageHistory(
        [
            SystemMessage("rules"),
            UserMessage("start"),
            opening,
            sibling_reply,
            return_reply,
        ]
    )
    for index in range(read_count):
        call = ToolCall(f"read-{index}", FunctionCall("read_tool_output", {}))
        with history.record_tool_call(call, batch_call_id="opening"):
            history.add(ToolMessage("final output", call.id))

    messages = history.get_history()
    updated = messages[2]
    assert isinstance(updated, ProviderResponse)
    assert updated.thinking is thinking
    assert updated.get_name() == "agent" and updated.get_message() == "open"
    assert [call.id for call in updated.get_tool_calls()] == [
        "sibling",
        "opening",
        *(f"read-{index}" for index in range(read_count)),
    ]
    assert messages[3] is sibling_reply and messages[-1] is return_reply
    assert opening.get_tool_calls() == [sibling, original]
    assert len(messages) == 5 + read_count


def test_recorded_tool_call_restores_return_reply_when_execution_raises():
    opening = ToolCall("opening", FunctionCall("skill", {}))
    reply = ToolMessage("skill done", "opening")
    history = MessageHistory(
        [SystemMessage("rules"), AssistantMessage(None, tool_calls=[opening]), reply]
    )
    read = ToolCall("read", FunctionCall("read_tool_output", {}))
    with pytest.raises(RuntimeError, match="failed read"):
        with history.record_tool_call(read, batch_call_id="opening"):
            history.add(ToolMessage("read error", "read"))
            raise RuntimeError("failed read")
    assert history.get_history()[-1] is reply


def test_recorded_tool_call_rejects_an_unrelated_batch_without_mutating_history():
    history = MessageHistory(
        [SystemMessage("rules"), ToolMessage("different return", "other")]
    )
    before = history.get_history()
    call = ToolCall("read", FunctionCall("read_tool_output", {}))
    with pytest.raises(ValueError, match="does not belong"):
        with history.record_tool_call(call, batch_call_id="opening"):
            pytest.fail("Unrelated batch should be rejected before execution")
    assert history.get_history() == before


@pytest.mark.parametrize(
    "messages", [[], [SystemMessage("rules"), UserMessage("start")]]
)
def test_replacing_latest_tool_calls_requires_an_assistant_message(messages):
    history = MessageHistory(messages)

    with pytest.raises(TypeError, match="last message must be an assistant"):
        history.replace_latest_tool_calls([ToolCall("call", FunctionCall("work", {}))])

    assert history.get_history() == messages


@pytest.mark.parametrize(
    ("messages", "error", "message"),
    [
        ([], TypeError, "last message must be a tool response"),
        (
            [SystemMessage("rules"), UserMessage("start")],
            TypeError,
            "last message must be a tool response",
        ),
        (
            [SystemMessage("rules"), ToolMessage("done", "opening")],
            ValueError,
            "no assistant message",
        ),
        (
            [
                SystemMessage("rules"),
                AssistantMessage(None),
                ToolMessage("done", "opening"),
            ],
            ValueError,
            "does not contain the batch call",
        ),
    ],
)
def test_recorded_tool_call_rejects_malformed_batches_atomically(
    messages, error, message
):
    history = MessageHistory(messages)
    call = ToolCall("read", FunctionCall("read_tool_output", {}))

    with pytest.raises(error, match=message):
        with history.record_tool_call(call, batch_call_id="opening"):
            pytest.fail("Malformed batches must be rejected before execution")

    assert history.get_history() == messages


def test_automatic_tool_call_preserves_the_latest_text_response_and_metadata():
    response = AssistantMessage("work is ready", name="agent")
    response.set_metadata("provider", {"response_id": "original"})
    history = MessageHistory([SystemMessage("rules"), UserMessage("start"), response])
    call = ToolCall("read", FunctionCall("read_tool_output", {}))
    with history.record_tool_call(call):
        history.add(ToolMessage("completed output", "read"))

    messages = history.get_history()
    assert [message.get_role_name() for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    updated = messages[2]
    assert updated.get_message() == "work is ready"
    assert updated.get_name() == "agent"
    assert updated.get_metadata() == {"provider": {"response_id": "original"}}
    assert updated.get_tool_calls() == [call]
    assert response.get_tool_calls() is None
