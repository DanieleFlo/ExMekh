from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image

from exmekh_lib.config.cache_text import CACHE_TEXT
from exmekh_lib.core.models.interactive import InterruptedMessage
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
    return {
        "name": message.get_name(),
        "parts": parts,
        **(
            {"interactive": message.record}
            if isinstance(message, InterruptedMessage)
            else {}
        ),
    }


def deserialize_user_message(data: dict[str, Any]) -> UserMessage:
    """Restore fresh attachments, reporting malformed cache data as a read error."""
    try:
        if "interactive" in data:
            record = data["interactive"]
            if not isinstance(record, dict):
                raise ValueError(CACHE_TEXT.cached_interactive_record_must_be_an_object)
            return InterruptedMessage(record)
        name = _text_field(data, "name")
        raw_parts = data["parts"]
        if not isinstance(raw_parts, list):
            raise ValueError(CACHE_TEXT.cached_user_message_parts_must_be_a_list)
        return UserMessage([_deserialize_part(part) for part in raw_parts], name=name)
    except (KeyError, TypeError, OSError) as error:
        raise ValueError(
            CACHE_TEXT.invalid_cached_user_message.format(error=error)
        ) from error


def _deserialize_part(part: dict[str, Any]) -> UserMessagePart:
    match part["type"]:
        case "text":
            return _text_field(part, "text")
        case "image":
            prefix, separator, data = _text_field(part, "data").partition(",")
            if not separator or prefix != "data:image/png;base64":
                raise ValueError(CACHE_TEXT.cached_images_must_be_png_data_urls)
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
            raise ValueError(
                CACHE_TEXT.unsupported_cached_user_content_type.format(unknown=unknown)
            )


def _text_field(data: dict[str, Any], name: str) -> str:
    value = data[name]
    if not isinstance(value, str):
        raise ValueError(
            CACHE_TEXT.cached_user_message_must_be_a_string.format(name=name)
        )
    return value
