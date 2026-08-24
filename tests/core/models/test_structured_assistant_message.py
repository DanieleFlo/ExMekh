from typing import cast

import pytest
from pydantic import BaseModel

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
