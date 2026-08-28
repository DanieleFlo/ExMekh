from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any, Literal

from qs.core.models.assistant_message import ToolCall


class Tool:
    def __init__(
        self,
        tool: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.tool = tool
        self._tool_type: Literal["function"] = "function"
        self._check_callable_tool()
        self.name = self._resolve_tool_name(tool, name)
        self.description = description or inspect.getdoc(tool) or ""
        self._check_inputs()

    def _resolve_tool_name(self, tool: Callable[..., Any], name: str | None) -> str:
        resolved_name = getattr(tool, "__name__", "") if name is None else name
        if not self.is_valid_tool_name(resolved_name):
            raise ValueError(
                "Tool name must contain 1 to 64 letters, digits, underscores, or hyphens"
            )
        return resolved_name

    def _check_callable_tool(self) -> None:
        if not callable(self.tool):
            raise TypeError("Tool must be callable")

    def _check_inputs(self):
        if not isinstance(self.description, str):
            raise TypeError("Tool description must be a string")

    def is_valid_tool_name(self, name: str) -> bool:
        _TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
        return bool(_TOOL_NAME_PATTERN.fullmatch(name))

    def execute(self, **arguments: Any) -> Any:
        return self.tool(**arguments)


class CalledTool:
    def __init__(self, tool: Tool, tool_call: ToolCall) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("tool must be a Tool instance")

        self._tool = tool
        self._tool_call = tool_call
        self._validate_matching()

    @classmethod
    def from_tool_call(cls, tool: Tool, tool_call: ToolCall) -> CalledTool:
        if not isinstance(tool_call, ToolCall):
            raise TypeError("tool_call must be a ToolCall instance")

        return cls(tool, tool_call)

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def call_id(self) -> str:
        return self._tool_call.id

    def execute(self) -> Any:
        return self._tool.execute(**self._tool_call.function.arguments)

    def _validate_matching(self) -> None:
        call_name = self._tool_call.get_name()
        if call_name != self._tool.name:
            raise ValueError(
                f"Tool call name '{call_name}' does not match "
                f"tool name '{self._tool.name}'"
            )
