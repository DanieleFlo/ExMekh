from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any, Literal


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
        if not isinstance(self.name, str):
            raise TypeError("Tool name must be a string")
        if not isinstance(self.description, str):
            raise TypeError("Tool description must be a string")

    def is_valid_tool_name(self, name: str) -> bool:
        _TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
        return bool(_TOOL_NAME_PATTERN.fullmatch(name))
