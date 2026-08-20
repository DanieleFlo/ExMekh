from __future__ import annotations

from typing import Literal

from PIL import Image

from qs.interfaces.message import Message


class UserMessage(Message):
    def __init__(
        self,
        content: str | TextsUserMessage | ImagesUserMessage | FilesUserMessage,
        name: str = "",
    ) -> None:
        self._content = content
        self._role = "user"
        self._name = name

    def get_message(
        self,
    ) -> str | TextsUserMessage | ImagesUserMessage | FilesUserMessage:
        return self._content

    def get_role_name(self) -> str:
        return self._role

    def get_name(self) -> str:
        return self._name

    def message_dump(self) -> dict:
        return {"role": self._role, "content": self._content}


class TextsUserMessage:
    texts: list[str]


class ImagesUserMessage:
    images: list[Image.Image]


class FilesUserMessage:
    documents: list[DocumentUserMessage]


class DocumentUserMessage:
    def __init__(
        self,
        file: bytes,
        name: str,
        extension: Literal["pdf", "txt", "csv", "json", "xml", "docx", "xlsx", "pptx"],
        mime_type: str = "application/pdf",
    ) -> None:
        self.file = file
        self.name = name
        self.extension = extension
        self.mime_type = mime_type
