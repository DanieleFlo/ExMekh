from __future__ import annotations

from typing import Any

from PIL import Image
from pydantic import BaseModel
from pydantic_core import to_json

from qs.core.models.assistant_message import AssistantMessage, ToolCall
from qs.core.models.function_message import FunctionMessage
from qs.core.models.structured_assistant_message import StructuredAssistantMessage
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool_message import ToolMessage
from qs.core.models.user_message import (
    DocumentUserMessage,
    UserMessage,
)
from qs.interfaces.message import Message
from qs.utils.file import convert_bytes_to_base64
from qs.utils.image import pil_image_to_data_url


class ToOpenAIResponseMapper:
    @staticmethod
    def map_system_message(message: SystemMessage) -> list[dict[str, Any]]:
        return [
            {
                "role": message.get_role_name(),
                "content": message.get_message(),
            }
        ]

    @classmethod
    def map_user_message(cls, message: UserMessage) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": cls._map_user_content(message),
            }
        ]

    @classmethod
    def map_assistant_message(
        cls,
        message: AssistantMessage,
    ) -> list[dict[str, Any]]:
        return cls._map_assistant_content(
            message.get_message(),
            message.get_tool_calls(),
        )

    @classmethod
    def map_structured_assistant_message(
        cls,
        message: StructuredAssistantMessage[BaseModel],
    ) -> list[dict[str, Any]]:
        content = message.get_message()
        if not isinstance(content, BaseModel):
            raise TypeError("Structured assistant content must be a Pydantic model")
        return cls._map_assistant_content(
            content.model_dump_json(),
            message.get_tool_calls(),
        )

    @staticmethod
    def map_tool_message(message: ToolMessage) -> list[dict[str, Any]]:
        return [
            {
                "type": "function_call_output",
                "call_id": message.get_tool_call_id(),
                "output": message.get_message(),
            }
        ]

    @staticmethod
    def map_function_message(message: FunctionMessage) -> list[dict[str, Any]]:
        raise TypeError("Responses API does not support legacy function messages")

    @classmethod
    def map_message(cls, message: Message[object]) -> list[dict[str, Any]]:
        if isinstance(message, SystemMessage):
            return cls.map_system_message(message)
        if isinstance(message, UserMessage):
            return cls.map_user_message(message)
        if isinstance(message, AssistantMessage):
            return cls.map_assistant_message(message)
        if isinstance(message, StructuredAssistantMessage):
            return cls.map_structured_assistant_message(message)
        if isinstance(message, ToolMessage):
            return cls.map_tool_message(message)
        if isinstance(message, FunctionMessage):
            return cls.map_function_message(message)
        raise TypeError(f"Unsupported message type: {type(message)}")

    @classmethod
    def _map_assistant_content(
        cls,
        content: str | None,
        tool_calls: list[ToolCall] | None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if content is not None:
            items.append({"role": "assistant", "content": content})
        if tool_calls is not None:
            items.extend(cls._map_tool_call(tool_call) for tool_call in tool_calls)
        return items

    @staticmethod
    def _map_tool_call(tool_call: ToolCall) -> dict[str, Any]:
        return {
            "type": "function_call",
            "call_id": tool_call.get_id(),
            "name": tool_call.get_name(),
            "arguments": to_json(tool_call.get_arguments()).decode(),
        }

    @classmethod
    def _map_user_content(cls, message: UserMessage) -> str | list[dict[str, Any]]:
        content = message.get_message()
        if isinstance(content, str):
            return content
        parts: list[dict[str, Any]] = []
        document_index = 0
        for part in message.iter_content_parts():
            if isinstance(part, str):
                parts.append({"type": "input_text", "text": part})
            elif isinstance(part, Image.Image):
                parts.append(
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": pil_image_to_data_url(part),
                    }
                )
            elif isinstance(part, DocumentUserMessage):
                parts.append(cls._map_file(part, document_index))
                document_index += 1
            else:
                raise TypeError(f"Unsupported user content type: {type(part).__name__}")
        return parts

    @staticmethod
    def _map_file(document: DocumentUserMessage, index: int) -> dict[str, Any]:
        return {
            "type": "input_file",
            "filename": f"{document.name}_{index}.{document.extension}",
            "file_data": (
                f"data:{document.mime_type};base64,"
                f"{convert_bytes_to_base64(document.file)}"
            ),
        }
