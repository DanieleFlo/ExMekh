from typing import cast

import pytest

from exmekh_lib.core.models.tool_message import ToolMessage


def test_init_uses_tool_defaults() -> None:
    message = ToolMessage("result", "call_123")

    assert message.get_role_name() == "tool"
    assert message.get_name() == ""
    assert message.get_tool_call_id() == "call_123"


def test_get_message_returns_content() -> None:
    message = ToolMessage("result", "call_123")

    assert message.get_message() == "result"


def test_get_name_returns_custom_name() -> None:
    message = ToolMessage("result", "call_123", name="lookup")

    assert message.get_name() == "lookup"


def test_message_dump_returns_provider_fields() -> None:
    message = ToolMessage("result", "call_123", name="lookup")

    assert message.message_dump() == {"role": "tool", "content": "result"}


@pytest.mark.parametrize(
    ("content", "tool_call_id", "name", "message"),
    [
        (object(), "call_123", "", "content must be a string"),
        ("result", object(), "", "call ID must be a string"),
        ("result", "call_123", object(), "name must be a string"),
    ],
)
def test_init_rejects_invalid_fields(
    content: object, tool_call_id: object, name: object, message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        ToolMessage(
            cast(str, content),
            cast(str, tool_call_id),
            name=cast(str, name),
        )
