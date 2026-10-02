import hashlib
from typing import cast

import pytest
from PIL import Image

from exmekh_lib.core.models.user_message import (
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


@pytest.mark.parametrize(
    ("content", "name", "message"),
    [
        (object(), "", "User content must be"),
        ("question", object(), "User name must be a string"),
    ],
)
def test_init_rejects_invalid_fields(
    content: object, name: object, message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        UserMessage(cast(str, content), name=cast(str, name))


@pytest.mark.parametrize(
    "content",
    [
        [],
        ["first", "second"],
        [
            "question",
            TextsUserMessage(["extra"]),
            ImagesUserMessage(),
            FilesUserMessage(),
        ],
    ],
)
def test_user_message_accepts_one_ordered_list_of_content_parts(content):
    message = UserMessage(content, name="customer")

    assert message.get_message() is content
    assert message.get_role_name() == "user"
    assert message.get_name() == "customer"


@pytest.mark.parametrize(
    "content", [[object()], ["text", 42], [["nested"]], ("tuple",)]
)
def test_user_message_rejects_invalid_or_nested_content_parts(content):
    with pytest.raises(TypeError, match="User content"):
        UserMessage(content)


def test_mixed_message_dump_preserves_text_image_and_file_order():
    first = Image.new("RGB", (2, 3), "red")
    second = Image.new("RGB", (3, 2), "blue")
    document = DocumentUserMessage(b"attachment", "notes", "txt", "text/plain")
    content = [
        "before",
        ImagesUserMessage([first]),
        TextsUserMessage(["middle", "next\nline"]),
        FilesUserMessage([document]),
        ImagesUserMessage([second]),
        "after",
    ]

    assert UserMessage(content).message_dump() == {
        "role": "user",
        "content": "\n".join(
            [
                "before",
                f"Image hash: {hashlib.sha256(first.tobytes()).hexdigest()}",
                "middle",
                "next\nline",
                "Document: notes, extension: txt, mime_type: text/plain, "
                f"Hash: {hashlib.sha256(document.file).hexdigest()}",
                f"Image hash: {hashlib.sha256(second.tobytes()).hexdigest()}",
                "after",
            ]
        ),
    }
    assert UserMessage([]).message_dump() == {"role": "user", "content": ""}


def test_copy_of_mixed_message_keeps_independent_text_pixels_and_documents():
    image = Image.new("RGB", (2, 3), "red")
    content = [
        "prompt",
        TextsUserMessage(["more text"]),
        ImagesUserMessage([image]),
        FilesUserMessage([DocumentUserMessage(b"original", "notes", "txt")]),
    ]
    copied = UserMessage(content).copy().get_message()
    copied[0] = "modified"
    copied[1].texts.append("changed")
    copied[2].images[0].putpixel((0, 0), (0, 0, 255))
    copied[3].documents[0].file = b"changed"

    assert content[0] == "prompt"
    assert content[1].texts == ["more text"]
    assert image.getpixel((0, 0)) == (255, 0, 0)
    assert content[3].documents[0].file == b"original"


def test_content_iterator_expands_containers_without_copying_values():
    image = Image.new("RGB", (1, 1), "red")
    document = DocumentUserMessage(b"notes", "report", "txt")
    content = [
        "before",
        TextsUserMessage(["middle", ""]),
        ImagesUserMessage([image]),
        FilesUserMessage([document]),
        TextsUserMessage(),
        "after",
    ]
    message = UserMessage(content)

    assert list(message.iter_content_parts()) == [
        "before",
        "middle",
        "",
        image,
        document,
        "after",
    ]
    assert message.get_message() is content


def test_empty_containers_do_not_add_text_to_a_mixed_dump():
    message = UserMessage(
        ["before", TextsUserMessage(), ImagesUserMessage(), FilesUserMessage(), "after"]
    )

    assert message.message_dump()["content"] == "before\nafter"


@pytest.mark.parametrize(
    "content",
    [
        "text",
        TextsUserMessage(["first", "second"]),
        ImagesUserMessage(),
        FilesUserMessage(),
    ],
)
def test_content_iterator_accepts_single_content_values(content):
    message = UserMessage(content)
    expected = (
        [content]
        if isinstance(content, str)
        else (content.texts if isinstance(content, TextsUserMessage) else [])
    )

    assert list(message.iter_content_parts()) == expected


def test_content_iterator_rejects_invalid_parts_added_after_construction():
    content = ["valid"]
    message = UserMessage(content)
    content.append(object())

    with pytest.raises(TypeError, match="Unsupported user content type: object"):
        list(message.iter_content_parts())
