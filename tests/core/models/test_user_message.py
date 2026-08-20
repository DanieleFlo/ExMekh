import hashlib

import pytest
from PIL import Image

from qs.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    ImagesUserMessage,
    TextsUserMessage,
    UserMessage,
)


@pytest.mark.parametrize(
    "content",
    ["question", TextsUserMessage(), ImagesUserMessage(), FilesUserMessage()],
)
def test_init_and_get_message_preserve_supported_content(content) -> None:
    message = UserMessage(content)

    assert message.get_message() is content


def test_get_role_name_always_returns_user() -> None:
    message = UserMessage("question")

    assert message.get_role_name() == "user"


def test_get_name_returns_custom_name() -> None:
    message = UserMessage("question", name="customer")

    assert message.get_name() == "customer"


def test_message_dump_returns_provider_fields() -> None:
    message = UserMessage("question", name="customer")

    assert message.message_dump() == {"role": "user", "content": "question"}


def test_message_dump_normalizes_injected_none_content() -> None:
    message = UserMessage("temporary")
    message._content = None

    assert message.message_dump() == {"role": "user", "content": ""}


def test_structured_content_collections_are_independent() -> None:
    first = TextsUserMessage()
    second = TextsUserMessage()

    first.texts.append("injected")

    assert second.texts == []


def test_message_dump_joins_non_default_text_parts() -> None:
    message = UserMessage(TextsUserMessage(["first line", "second\nline"]))

    assert message.message_dump() == {
        "role": "user",
        "content": "first line\nsecond\nline",
    }


def test_message_dump_hashes_image_pixels_with_sha256() -> None:
    image = Image.new("RGB", (2, 1), color=(17, 34, 51))
    expected_hash = hashlib.sha256(image.tobytes()).hexdigest()

    result = UserMessage(ImagesUserMessage([image])).message_dump()

    assert result == {"role": "user", "content": f"Image hash: {expected_hash}"}


def test_message_dump_hashes_document_bytes_without_opening_filename() -> None:
    file_bytes = b"\x00hostile-looking-content\xff"
    document = DocumentUserMessage(
        file_bytes,
        "../../not-opened\nreport",
        "json",
        mime_type="application/custom+json",
    )
    expected_hash = hashlib.sha256(file_bytes).hexdigest()

    result = UserMessage(FilesUserMessage([document])).message_dump()

    assert result["content"] == (
        "Document: ../../not-opened\nreport, extension: json, "
        "mime_type: application/custom+json, "
        f"Hash: {expected_hash}"
    )


def test_document_user_message_uses_default_mime_type() -> None:
    document = DocumentUserMessage(b"data", "report", "pdf")

    assert document.file == b"data"
    assert document.name == "report"
    assert document.extension == "pdf"
    assert document.mime_type == "application/pdf"


def test_document_user_message_preserves_custom_mime_type() -> None:
    document = DocumentUserMessage(
        b"name,value", "records", "csv", mime_type="text/csv"
    )

    assert document.mime_type == "text/csv"
