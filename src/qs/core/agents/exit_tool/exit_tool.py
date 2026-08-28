from __future__ import annotations

from pathlib import Path
from typing import Any

from qs.core.agents.exit_tool.data import ExitToolData
from qs.core.agents.exit_tool.decorator import ExitToolFunction
from qs.core.models.assistant_message import ToolCall
from qs.core.models.tool import Tool


class ExitTool(Tool):
    def __init__(
        self,
        function: ExitToolFunction,
        *,
        name: str,
        description: str,
        exit_skill_name: str,
        folder_path: Path,
    ) -> None:
        self.exit_skill_name = exit_skill_name
        self.folder_path = folder_path
        super().__init__(function, name=name, description=description)

    def get_exit_skill_name(self) -> str:
        return self.exit_skill_name

    def execute(self, *args: Any, **kwargs: Any) -> dict[str, str | list[str]]:
        result = self.tool(*args, **kwargs)
        if not isinstance(result, ExitToolData):
            raise TypeError("exit tool must return ExitToolData")

        return result.dump()


class CalledExitTool:
    def __init__(self, exit_tool: ExitTool, tool_call: ToolCall) -> None:
        if not isinstance(exit_tool, ExitTool):
            raise TypeError("exit_tool must be an ExitTool")
        if not isinstance(tool_call, ToolCall):
            raise TypeError("tool_call must be a ToolCall")

        self.exit_tool = exit_tool
        self.tool_call = tool_call

    @property
    def call_id(self) -> str:
        return self.tool_call.id
