from __future__ import annotations

from pathlib import Path
from typing import Any

from qs.core.agents.capability.references import parse_references
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
        instructions_path: Path,
        instructions: str,
        folder_path: Path,
    ) -> None:
        if not isinstance(instructions_path, Path):
            raise TypeError("instructions_path must be a Path")
        if not isinstance(instructions, str):
            raise TypeError("instructions must be a string")
        if not instructions.strip():
            raise ValueError(
                f"Exit-tool instructions cannot be blank: {instructions_path}"
            )
        self.instructions_path = instructions_path
        self.instructions, self.references = parse_references(instructions.strip())
        self.folder_path = folder_path
        super().__init__(function, name=name, description=description)

    def render_prompt_section(self) -> str:
        return f'<exit-tool name="{self.name}">{self.instructions}</exit-tool>'

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

        self._exit_tool = exit_tool
        self._tool_call = tool_call

    @property
    def name(self) -> str:
        return self._exit_tool.name

    @property
    def call_id(self) -> str:
        return self._tool_call.get_id()

    @property
    def arguments(self) -> dict[str, Any]:
        return self._tool_call.get_arguments()

    @property
    def tool_call(self) -> ToolCall:
        return self._tool_call

    def execute(self) -> dict[str, str | list[str]]:
        return self._exit_tool.execute(**self._tool_call.get_arguments())
