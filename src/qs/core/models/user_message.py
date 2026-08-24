from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
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
        self._check_inputs()

    def _check_inputs(self):
        if not isinstance(self._name, str):
            raise TypeError("User name must be a string")
        if not isinstance(
            self._content,
            (str, TextsUserMessage, ImagesUserMessage, FilesUserMessage),
        ):
            raise TypeError(
                "User content must be a string or an instance of TextsUserMessage, "
                "ImagesUserMessage, or FilesUserMessage"
            )

    def get_message(
        self,
    ) -> str | TextsUserMessage | ImagesUserMessage | FilesUserMessage:
        return self._content

    def get_role_name(self) -> str:
        return self._role

    def get_name(self) -> str:
        return self._name

    def message_dump(self) -> dict[str, str]:
        content = self._content
        if content is None:
            content = ""
        if isinstance(content, TextsUserMessage):
            content = "\n".join(content.texts)
        if isinstance(content, ImagesUserMessage):
            content = "\n".join(
                [f"Image hash: {self._get_hash(image)}" for image in content.images]
            )
        if isinstance(content, FilesUserMessage):
            content = "\n".join(
                [
                    f"Document: {document.name}, extension: {document.extension}, "
                    f"mime_type: {document.mime_type}, "
                    f"Hash: {self._get_hash(document.file)}"
                    for document in content.documents
                ]
            )
        return {"role": self._role, "content": content}

    def _get_hash(self, data: bytes | Image.Image) -> str:
        raw_data = data if isinstance(data, bytes) else data.tobytes()
        return hashlib.sha256(raw_data).hexdigest()


@dataclass
class TextsUserMessage:
    texts: list[str] = field(default_factory=list)


@dataclass
class ImagesUserMessage:
    images: list[Image.Image] = field(default_factory=list)


@dataclass
class FilesUserMessage:
    documents: list[DocumentUserMessage] = field(default_factory=list)


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
