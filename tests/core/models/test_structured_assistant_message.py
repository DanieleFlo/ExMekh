import json
from typing import cast

import pytest
from pydantic import BaseModel

from qs.core.models.assistant_message import (
    CustomCall,
    CustomToolCall,
    FunctionCall,
    FunctionToolCall,
)
from qs.core.models.structured_assistant_message import StructuredAssistantMessage


class Answer(BaseModel):
    value: int


def test_init_preserves_structured_content_and_defaults() -> None:
    content = Answer(value=42)
    message = StructuredAssistantMessage(content)

    assert message.get_message() is content
    assert message.get_role_name() == "assistant"
    assert message.get_name() == ""
    assert message.get_tool_calls() is None


def test_message_dump_serializes_content_as_json() -> None:
    message = StructuredAssistantMessage(Answer(value=42), name="agent")

    assert message.message_dump() == {
        "role": "assistant",
        "content": '{"value":42}',
        "tool_calls": "null",
    }


def test_init_rejects_non_pydantic_content() -> None:
    with pytest.raises(TypeError, match="Pydantic BaseModel"):
        StructuredAssistantMessage(cast(Answer, object()))


@pytest.mark.parametrize("tool_calls", [object(), [object()]])
def test_init_rejects_invalid_tool_calls(tool_calls: object) -> None:
    with pytest.raises(TypeError, match="list of ToolCall objects"):
        StructuredAssistantMessage(Answer(value=42), tool_calls=cast(list, tool_calls))


def test_get_message_rejects_corrupted_content() -> None:
    message = StructuredAssistantMessage(Answer(value=42))
    message._content = cast(Answer, object())

    with pytest.raises(TypeError, match="content must be a subclass of BaseModel"):
        message.get_message()


def test_message_dump_serializes_function_and_custom_tool_calls() -> None:
    message = StructuredAssistantMessage(
        Answer(value=42),
        tool_calls=[
            FunctionToolCall(
                id="function-call",
                function=FunctionCall(name="search", arguments='{"query":"docs"}'),
            ),
            CustomToolCall(
                id="custom-call",
                custom=CustomCall(name="execute", input="print(42)"),
            ),
        ],
    )

    assert json.loads(message.message_dump()["tool_calls"]) == [
        {
            "id": "function-call",
            "function": {"name": "search", "arguments": '{"query":"docs"}'},
            "type": "function",
        },
        {
            "id": "custom-call",
            "custom": {"name": "execute", "input": "print(42)"},
            "type": "custom",
        },
    ]
