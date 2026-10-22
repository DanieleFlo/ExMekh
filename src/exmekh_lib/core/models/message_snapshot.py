"""Lossless tool attachments for exported history, separate from compact dumps."""

from typing import Any

from exmekh_lib.core.cache.user_message_codec import serialize_user_message
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.interfaces.message import Message


def snapshot_message(message: Message[object]) -> dict[str, Any]:
    snapshot: dict[str, Any] = dict(message.message_dump())
    if (
        isinstance(message, ToolMessage)
        and (attachments := message.get_attachments()) is not None
    ):
        snapshot["attachments"] = serialize_user_message(attachments)
    return snapshot
