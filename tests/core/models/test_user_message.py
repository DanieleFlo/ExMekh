import pytest

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
