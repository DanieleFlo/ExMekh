from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from pydantic import BaseModel

from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.function_message import FunctionMessage
from qs.core.models.structured_assistant_message import StructuredAssistantMessage
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool_message import ToolMessage
from qs.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    ImagesUserMessage,
    TextsUserMessage,
    UserMessage,
)
from qs.core.providers.to_openai_response_mapper import ToOpenAIResponseMapper
from qs.interfaces.message import Message


class StructuredAnswer(BaseModel):
    value: int


@pytest.mark.parametrize(
    ("message", "mapper_name"),
    [
        (SystemMessage("rules"), "map_system_message"),
        (UserMessage("question"), "map_user_message"),
        (AssistantMessage("answer"), "map_assistant_message"),
        (
            StructuredAssistantMessage(StructuredAnswer(value=42)),
            "map_structured_assistant_message",
        ),
        (ToolMessage("result", "call_123"), "map_tool_message"),
        (FunctionMessage("result", "legacy"), "map_function_message"),
    ],
)
def test_map_message_dispatches_supported_messages(message, mapper_name) -> None:
    mapped = [{"mapped": True}]
    with patch.object(
        ToOpenAIResponseMapper,
        mapper_name,
        return_value=mapped,
    ) as mapper:
        result = ToOpenAIResponseMapper.map_message(message)

    assert result is mapped
    mapper.assert_called_once_with(message)


def test_map_message_rejects_unknown_message() -> None:
    with pytest.raises(TypeError, match="Unsupported message type"):
        ToOpenAIResponseMapper.map_message(MagicMock(spec=Message))


def test_map_structured_assistant_rejects_non_model_content() -> None:
    message = MagicMock(spec=StructuredAssistantMessage)
    message.get_message.return_value = object()

    with pytest.raises(TypeError, match="Pydantic model"):
        ToOpenAIResponseMapper.map_structured_assistant_message(message)


def test_map_structured_assistant_serializes_content() -> None:
    message = StructuredAssistantMessage(StructuredAnswer(value=42))

    assert ToOpenAIResponseMapper.map_structured_assistant_message(message) == [
        {"role": "assistant", "content": '{"value":42}'}
    ]


def test_map_user_message_maps_text_parts() -> None:
    message = UserMessage(TextsUserMessage(["first", "second"]))

    assert ToOpenAIResponseMapper.map_user_message(message) == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "first"},
                {"type": "input_text", "text": "second"},
            ],
        }
    ]


def test_map_user_message_maps_images() -> None:
    image = Image.new("RGB", (1, 1), "red")
    message = UserMessage(ImagesUserMessage([image]))

    content = ToOpenAIResponseMapper.map_user_message(message)[0]["content"]

    assert content[0]["type"] == "input_image"
    assert content[0]["detail"] == "auto"
    assert content[0]["image_url"].startswith("data:image/png;base64,")


def test_map_user_message_maps_files() -> None:
    document = DocumentUserMessage(
        b"content",
        "notes",
        "txt",
        "text/plain",
    )
    message = UserMessage(FilesUserMessage([document]))

    assert ToOpenAIResponseMapper.map_user_message(message) == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "filename": "notes_0.txt",
                    "file_data": "data:text/plain;base64,Y29udGVudA==",
                }
            ],
        }
    ]


def test_map_user_message_rejects_unknown_content_part() -> None:
    class InvalidMessage(UserMessage):
        def iter_content_parts(self):
            yield object()

    with pytest.raises(TypeError, match="Unsupported user content type"):
        ToOpenAIResponseMapper.map_user_message(InvalidMessage([]))
