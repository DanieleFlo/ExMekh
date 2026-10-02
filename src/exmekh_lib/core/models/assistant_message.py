from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Literal, Self

from pydantic_core import to_json

from exmekh_lib.interfaces.message import Message


class AssistantMessage(Message[str | None]):
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
        self._check_inputs()

    def _check_inputs(self):
        if self._content is not None and not isinstance(self._content, str):
            raise TypeError("Assistant content must be a string")
        if self._tool_calls is not None:
            if not isinstance(self._tool_calls, list):
                raise TypeError(
                    "Assistant tool_calls must be a list of ToolCall objects"
                )
            for tool_call in self._tool_calls:
                if not isinstance(tool_call, ToolCall):
                    raise TypeError(
                        "Assistant tool_calls must be a list of ToolCall objects"
                    )

    def get_message(self) -> str | None:
        return self._content

    def get_role_name(self) -> str:
        return self._role

    def get_name(self) -> str:
        return self._name

    def get_tool_calls(self) -> list[ToolCall] | None:
        return self._tool_calls

    def with_tool_calls(self, tool_calls: Sequence[ToolCall]) -> Self:
        """Copy the message with new calls, preserving provider metadata."""
        message = copy(self)
        message._tool_calls = list(tool_calls)
        message._check_inputs()
        return message

    def message_dump(self) -> dict[str, str]:
        content = self._content
        if content is None:
            content = ""
        _tool_calls = self._tool_calls
        if _tool_calls is None:
            tool_calls_content = "null"
        else:
            tool_calls_content = ToolCall.dump_many(_tool_calls)
        return {
            "role": self._role,
            "content": content,
            "tool_calls": tool_calls_content,
        }


@dataclass(frozen=True, slots=True)
class FunctionCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    function: FunctionCall
    type: Literal["function"] = field(default="function", init=False)

    def get_name(self) -> str:
        return self.function.name

    def get_id(self) -> str:
        return self.id

    def get_fun_name(self) -> str | None:
        return self.function.name

    def get_type(self) -> str:
        return self.type

    def get_arguments(self) -> dict[str, Any]:
        return self.function.arguments

    @staticmethod
    def dump_many(tool_calls: Sequence[ToolCall]) -> str:
        return to_json(tool_calls, indent=2).decode()
