from __future__ import annotations

import json
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Literal, Self

from pydantic_core import to_jsonable_python

from exmekh_lib.config.model_text import MODEL_TEXT
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
            raise TypeError(MODEL_TEXT.assistant_content_must_be_a_string)
        if self._tool_calls is not None:
            if not isinstance(self._tool_calls, list):
                raise TypeError(
                    MODEL_TEXT.assistant_tool_calls_must_be_a_list_of_toolcall
                )
            for tool_call in self._tool_calls:
                if not isinstance(tool_call, ToolCall):
                    raise TypeError(
                        MODEL_TEXT.assistant_tool_calls_must_be_a_list_of_toolcall
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

    def arguments_json(self) -> str:
        """Keep rejected arguments recoverable when sending the next provider turn."""
        return self._serialize_json(self.get_arguments())

    @staticmethod
    def dump_many(tool_calls: Sequence[ToolCall]) -> str:
        records = [
            {
                "id": call.get_id(),
                "function": {
                    "name": call.get_name(),
                    "arguments": call.get_arguments(),
                },
                "type": call.get_type(),
            }
            for call in tool_calls
        ]
        return ToolCall._serialize_json(records, indent=2)

    @staticmethod
    def _serialize_json(value: object, *, indent: int | None = None) -> str:
        # Preserve typed values while escaping invalid Unicode instead of losing history.
        return json.dumps(
            ToolCall._json_value(value),
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
        )

    @staticmethod
    def _json_value(value: object) -> Any:
        if isinstance(value, dict):
            converted = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    # Keep the provider's formatting for typed mapping keys.
                    key = next(iter(to_jsonable_python({key: None})))
                converted[key] = ToolCall._json_value(item)
            return converted
        if isinstance(value, (list, tuple, set, frozenset)):
            return [ToolCall._json_value(item) for item in value]
        return to_jsonable_python(value)
