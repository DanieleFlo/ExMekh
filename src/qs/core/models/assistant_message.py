from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

from qs.interfaces.message import Message


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
                raise TypeError("Assistant tool_calls must be a list of ToolCall objects")
            for tool_call in self._tool_calls:
                if not isinstance(tool_call, (FunctionToolCall, CustomToolCall)):
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

    def message_dump(self) -> dict[str, str]:
        content = self._content
        if content is None:
            content = ""
        _tool_calls = self._tool_calls
        if _tool_calls is None:
            tool_calls_content = "null"
        else:
            tool_calls_content = self._dump_tool_calls(_tool_calls)
        return {
            "role": self._role,
            "content": content,
            "tool_calls": tool_calls_content,
        }

    def _dump_tool_calls(self, tool_calls_content: list[ToolCall]) -> str:
        return json.dumps(
            [asdict(tool_call) for tool_call in tool_calls_content],
            indent=2,
            ensure_ascii=False,
        )


@dataclass(frozen=True, slots=True)
class FunctionCall:
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class FunctionToolCall:
    id: str
    function: FunctionCall
    type: Literal["function"] = field(default="function", init=False)


@dataclass(frozen=True, slots=True)
class CustomCall:
    name: str
    input: str


@dataclass(frozen=True, slots=True)
class CustomToolCall:
    id: str
    custom: CustomCall
    type: Literal["custom"] = field(default="custom", init=False)


ToolCall = FunctionToolCall | CustomToolCall
