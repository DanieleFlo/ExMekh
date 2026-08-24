import json
from typing import cast

import pytest

from qs.core.models.assistant_message import (
    AssistantMessage,
    CustomCall,
    CustomToolCall,
    FunctionCall,
    FunctionToolCall,
)


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


def test_get_tool_calls_returns_tool_calls() -> None:
    tool_calls = [
        FunctionToolCall(
            id="call_123",
            function=FunctionCall(name="search", arguments="{}"),
        ),
        CustomToolCall(
            id="call_456",
            custom=CustomCall(name="execute_python", input="print(2 + 2)"),
        ),
    ]
    message = AssistantMessage(None, tool_calls=tool_calls)

    assert message.get_tool_calls() is tool_calls
    assert tool_calls[0].function.name == "search"
    assert tool_calls[1].custom.name == "execute_python"


def test_message_dump_returns_provider_fields() -> None:
    message = AssistantMessage(
        "answer",
        name="agent",
        tool_calls=[
            FunctionToolCall(
                id="call_123",
                function=FunctionCall(name="search", arguments="{}"),
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
            "function": {"name": "search", "arguments": "{}"},
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
