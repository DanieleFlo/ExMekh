from __future__ import annotations

from typing import Literal, TypedDict

from qs.interfaces.message import Message


class AssistantMessage(Message):
    def __init__(
        self,
        content: str | None,
        name: str = "",
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        self._content = content
        self._role: str = "assistant"
        self._name = name

        # function_call is deprecated; use tool_calls
        self._tool_calls = tool_calls

    def get_message(self) -> str | None:
        return self._content

    def get_role_name(self) -> str:
        return self._role

    def get_name(self) -> str:
        return self._name

    def get_tool_calls(self) -> list[ToolCall] | None:
        return self._tool_calls

    def message_dump(self) -> dict[str, str]:
        content = self._content
        if content is None:
            content = ""
        return {
            "role": self._role,
            "content": content,
        }


class FunctionCall(TypedDict):
    name: str
    arguments: str


class FunctionToolCall(TypedDict):
    id: str
    function: FunctionCall
    type: Literal["function"]


class CustomCall(TypedDict):
    name: str
    input: str


class CustomToolCall(TypedDict):
    id: str
    custom: CustomCall
    type: Literal["custom"]


ToolCall = FunctionToolCall | CustomToolCall
