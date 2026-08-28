from __future__ import annotations

from pydantic import BaseModel

from qs.core.models.assistant_message import ToolCall
from qs.interfaces.message import Message


class StructuredAssistantMessage[MessageContent: BaseModel](Message[MessageContent]):
    def __init__(
        self,
        content: MessageContent,
        name: str = "",
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        self._content = content
        self._role = "assistant"
        self._name = name
        self._tool_calls = tool_calls
        self._check_inputs()

    def _check_inputs(self):
        if not issubclass(type(self._content), BaseModel):
            raise TypeError("Structured assistant content must be a Pydantic BaseModel")
        if self._tool_calls is not None:
            if not isinstance(self._tool_calls, list):
                raise TypeError(
                    "Structured assistant tool_calls must be a list of ToolCall objects"
                )
            for tool_call in self._tool_calls:
                if not isinstance(tool_call, ToolCall):
                    raise TypeError(
                        "Structured assistant tool_calls must be a list of ToolCall objects"
                    )

    def get_message(self) -> MessageContent:
        if not issubclass(type(self._content), BaseModel):
            raise TypeError(
                f"content must be a subclass of BaseModel, got {type(self._content)!r}"
            )
        return self._content

    def get_role_name(self) -> str:
        return self._role

    def get_name(self) -> str:
        return self._name

    def get_tool_calls(self) -> list[ToolCall] | None:
        return self._tool_calls

    def message_dump(self) -> dict[str, str]:
        _tool_calls = self._tool_calls
        if _tool_calls is None:
            tool_calls_content = "null"
        else:
            tool_calls_content = ToolCall.dump_many(_tool_calls)
        return {
            "role": self._role,
            "content": self._content.model_dump_json(),
            "tool_calls": tool_calls_content,
        }
