import pytest
from PIL import Image

from qs.capabilities.tools.read_tool_output.tool_output_reader import ToolOutputReader
from qs.core.cache.session_cache import SessionCache
from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.cache.user_message_codec import (
    deserialize_user_message,
    serialize_user_message,
)
from qs.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    ImagesUserMessage,
    TextsUserMessage,
    UserMessage,
)


def test_cached_message_roundtrip_preserves_order_pixels_and_documents(tmp_path):
    image = Image.new("RGBA", (2, 3), (10, 20, 30, 40))
    document = DocumentUserMessage(b"file contents", "notes", "txt", "text/plain")
    message = UserMessage(
        [
            "before",
            ImagesUserMessage([image]),
            TextsUserMessage(["between", "texts"]),
            FilesUserMessage([document]),
            "after",
        ],
        name="author",
    )
    payload = serialize_user_message(message)
    image.putpixel((0, 0), (0, 0, 0, 0))
    document.file = b"changed"
    with SessionCache(tmp_path) as session:
        cache = session.node("owner")
        record = ToolCallRecord(
            "steering_message",
            "guidance",
            {},
            message.message_dump()["content"],
            source="UserMessage",
            user_message=payload,
        )
        cache.store(record)
        restored = deserialize_user_message(cache.get("guidance").user_message)
        parts = list(restored.iter_content_parts())
        assert restored.get_name() == "author"
        assert parts[0] == "before"
        assert parts[1].getpixel((0, 0)) == (10, 20, 30, 40)
        assert parts[2:4] == ["between", "texts"]
        assert parts[4].file == b"file contents"
        assert parts[4].name == "notes"
        assert parts[4].extension == "txt"
        assert parts[4].mime_type == "text/plain"
        assert parts[5] == "after"
        read = ToolOutputReader(lambda: cache).read("guidance")
        assert read.attachments is not None
        second_read = ToolOutputReader(lambda: cache).read("guidance")
        restored_image = next(
            p
            for p in read.attachments.iter_content_parts()
            if isinstance(p, Image.Image)
        )
        restored_image.putpixel((0, 0), (0, 0, 0, 0))
        second_image = next(
            p
            for p in second_read.attachments.iter_content_parts()
            if isinstance(p, Image.Image)
        )
        assert second_image.getpixel((0, 0)) == (10, 20, 30, 40)
        foreign = ToolOutputReader(lambda: session.node("other")).read("guidance")
        assert "<found>false</found>" in foreign.content
        assert foreign.attachments is None


def test_unknown_cached_content_type_is_rejected():
    with pytest.raises(ValueError, match="Unsupported cached user content"):
        deserialize_user_message({"name": "", "parts": [{"type": "unknown"}]})


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"name": 1, "parts": []},
        {"name": "", "parts": None},
        {"name": "", "parts": [None]},
        {"name": "", "parts": [{}]},
        {"name": "", "parts": [{"type": "text", "text": None}]},
        {"name": "", "parts": [{"type": "image", "data": "missing prefix"}]},
        {"name": "", "parts": [{"type": "image", "data": "data:image/png;base64,"}]},
        {"name": "", "parts": [{"type": "file", "data": "invalid base64"}]},
    ],
)
def test_malformed_cached_messages_raise_recoverable_read_errors(payload):
    with pytest.raises(ValueError):
        deserialize_user_message(payload)
