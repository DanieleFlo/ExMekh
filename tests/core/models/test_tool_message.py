from qs.core.models.tool_message import ToolMessage


def test_init_uses_tool_defaults() -> None:
    message = ToolMessage("result", "call_123")

    assert message.get_role_name() == "tool"
    assert message.get_name() == ""
    assert message.get_tool_call_id() == "call_123"


def test_get_message_returns_content() -> None:
    message = ToolMessage("result", "call_123")

    assert message.get_message() == "result"


def test_get_role_name_returns_custom_role() -> None:
    message = ToolMessage("result", "call_123", role="custom")

    assert message.get_role_name() == "custom"


def test_get_name_returns_custom_name() -> None:
    message = ToolMessage("result", "call_123", name="lookup")

    assert message.get_name() == "lookup"


def test_message_dump_returns_provider_fields() -> None:
    message = ToolMessage("result", "call_123", name="lookup")

    assert message.message_dump() == {"role": "tool", "content": "result"}
