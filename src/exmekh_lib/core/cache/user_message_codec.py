from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image

from exmekh_lib.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    ImagesUserMessage,
    UserMessage,
    UserMessagePart,
)
from exmekh_lib.utils.image import pil_image_to_data_url


def serialize_user_message(message: UserMessage) -> dict[str, Any]:
    """Keep ordered text and attachment bytes in the node's JSON cache."""
    parts: list[dict[str, Any]] = []
    for part in message.iter_content_parts():
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
        elif isinstance(part, Image.Image):
            parts.append({"type": "image", "data": pil_image_to_data_url(part)})
        else:
            parts.append(
                {
                    "type": "file",
                    "data": base64.b64encode(part.file).decode("ascii"),
                    "name": part.name,
                    "extension": part.extension,
                    "mime_type": part.mime_type,
                }
            )
    return {"name": message.get_name(), "parts": parts}


def deserialize_user_message(data: dict[str, Any]) -> UserMessage:
    """Restore fresh attachments, reporting malformed cache data as a read error."""
    try:
        name = _text_field(data, "name")
        raw_parts = data["parts"]
        if not isinstance(raw_parts, list):
            raise ValueError("Cached user message parts must be a list")
        return UserMessage([_deserialize_part(part) for part in raw_parts], name=name)
    except (KeyError, TypeError, OSError) as error:
        raise ValueError(f"Invalid cached user message: {error}") from error


def _deserialize_part(part: dict[str, Any]) -> UserMessagePart:
    match part["type"]:
        case "text":
            return _text_field(part, "text")
        case "image":
            prefix, separator, data = _text_field(part, "data").partition(",")
            if not separator or prefix != "data:image/png;base64":
                raise ValueError("Cached images must be PNG data URLs")
            with Image.open(BytesIO(base64.b64decode(data, validate=True))) as image:
                return ImagesUserMessage([image.copy()])
        case "file":
            return FilesUserMessage(
                [
                    DocumentUserMessage(
                        file=base64.b64decode(_text_field(part, "data"), validate=True),
                        name=_text_field(part, "name"),
                        extension=part["extension"],
                        mime_type=_text_field(part, "mime_type"),
                    )
                ]
            )
        case unknown:
            raise ValueError(f"Unsupported cached user content type: {unknown}")


def _text_field(data: dict[str, Any], name: str) -> str:
    value = data[name]
    if not isinstance(value, str):
        raise ValueError(f"Cached user message {name} must be a string")
    return value
