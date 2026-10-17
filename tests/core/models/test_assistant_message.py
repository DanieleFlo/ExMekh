import json
from decimal import Decimal
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.providers.to_openai_chat_completion_mapper import (
    ToOpenAIChatCompletionMapper,
)
from exmekh_lib.core.providers.to_openai_response_mapper import ToOpenAIResponseMapper


def test_init_uses_assistant_defaults() -> None:
    message = AssistantMessage("answer")

    assert message.get_role_name() == "assistant"
    assert message.get_name() == ""
    assert message.get_tool_calls() is None


def test_get_message_returns_content() -> None:
    message = AssistantMessage("answer")

    assert message.get_message() == "answer"


def test_get_name_returns_custom_name() -> None:
    message = AssistantMessage("answer", name="agent")

    assert message.get_name() == "agent"


def test_message_dump_returns_provider_fields() -> None:
    message = AssistantMessage(
        "answer",
        name="agent",
        tool_calls=[
            ToolCall(
                id="call_123",
                function=FunctionCall(name="search", arguments={}),
            )
        ],
    )
    dumped = message.message_dump()

    assert {"role": dumped["role"], "content": dumped["content"]} == {
        "role": "assistant",
        "content": "answer",
    }
    assert json.loads(dumped["tool_calls"]) == [
        {
            "id": "call_123",
            "function": {"name": "search", "arguments": {}},
            "type": "function",
        }
    ]


def test_message_dump_normalizes_missing_content() -> None:
    assert AssistantMessage(None).message_dump() == {
        "role": "assistant",
        "content": "",
        "tool_calls": "null",
    }


def test_init_rejects_non_string_content() -> None:
    with pytest.raises(TypeError, match="Assistant content must be a string"):
        AssistantMessage(cast(str, object()))


@pytest.mark.parametrize("tool_calls", [object(), [object()]])
def test_init_rejects_invalid_tool_calls(tool_calls: object) -> None:
    with pytest.raises(TypeError, match="list of ToolCall objects"):
        AssistantMessage(None, tool_calls=cast(list, tool_calls))


def test_tool_call_exposes_function_data() -> None:
    arguments = {"query": "docs"}
    tool_call = ToolCall(
        id="call-123",
        function=FunctionCall(name="search", arguments=arguments),
    )

    assert tool_call.get_name() == "search"
    assert tool_call.get_id() == "call-123"
    assert tool_call.get_fun_name() == "search"
    assert tool_call.get_type() == "function"
    assert tool_call.get_arguments() is arguments


def test_metadata_updates_do_not_mutate_a_message_copied_with_new_calls() -> None:
    message = AssistantMessage("answer")
    message.set_metadata("frontend", {"turn": "original"})
    copied = message.with_tool_calls([])
    copied.set_metadata("frontend", {"turn": "copy"})
    assert copied.get_metadata()["frontend"] == {"turn": "copy"}
    assert message.get_metadata()["frontend"] == {"turn": "original"}


@pytest.mark.parametrize("value", ["bad\ud800path", "bad\udfffpath", "é😊\r\n\x00"])
def test_tool_call_json_retains_rejected_unicode_in_history_and_provider_arguments(
    value,
):
    arguments = {"path": value, "nested": [{value: value}]}
    call = ToolCall("call", FunctionCall("new-file", arguments))
    message = AssistantMessage(None, tool_calls=[call])
    exported = json.loads(message.message_dump()["tool_calls"].encode("utf-8"))
    assert exported[0]["function"]["arguments"] == arguments
    assert json.loads(call.arguments_json().encode("utf-8")) == arguments
    response = ToOpenAIResponseMapper.map_assistant_message(message)
    chat = ToOpenAIChatCompletionMapper.map_assistant_message(message)
    assert json.loads(response[0]["arguments"].encode("utf-8")) == arguments
    assert (
        json.loads(chat["tool_calls"][0]["function"]["arguments"].encode("utf-8"))
        == arguments
    )


def test_tool_call_json_preserves_typed_values_alongside_rejected_unicode():
    identifier = UUID(int=1)
    arguments = {
        "lookup": {identifier: Decimal("1.5")},
        "path": Path("notes.txt"),
        "rejected": {"bad\ud800key": "bad\ud800value"},
    }
    call = ToolCall("call", FunctionCall("tool", arguments))
    expected = {
        "lookup": {str(identifier): "1.5"},
        "path": "notes.txt",
        "rejected": {"bad\ud800key": "bad\ud800value"},
    }
    assert json.loads(call.arguments_json()) == expected
    assert (
        json.loads(ToolCall.dump_many([call]))[0]["function"]["arguments"] == expected
    )
