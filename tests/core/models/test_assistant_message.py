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
            type="function",
        ),
        CustomToolCall(
            id="call_456",
            custom=CustomCall(name="execute_python", input="print(2 + 2)"),
            type="custom",
        ),
    ]
    message = AssistantMessage(None, tool_calls=tool_calls)

    assert message.get_tool_calls() is tool_calls


def test_message_dump_returns_provider_fields() -> None:
    message = AssistantMessage(
        "answer",
        name="agent",
        tool_calls=[],
    )

    assert message.message_dump() == {"role": "assistant", "content": "answer"}
