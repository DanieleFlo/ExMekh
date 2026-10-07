from __future__ import annotations

from pathlib import Path
from typing import Any

from exmekh_lib.core.agents.capability.references import parse_references
from exmekh_lib.core.agents.exit_tool.data import ExitToolData, ExitToolResult
from exmekh_lib.core.agents.exit_tool.decorator import ExitToolFunction
from exmekh_lib.core.models.assistant_message import ToolCall
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.interfaces.message import Message
from exmekh_lib.interfaces.tool_hook import AfterTool, BeforeTool, ToolContext


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
        before: BeforeTool | None = None,
        after: AfterTool | None = None,
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
        super().__init__(
            function,
            name=name,
            description=description,
            background_enabled=False,
            before=before,
            after=after,
        )

    def render_prompt_section(self) -> str:
        return f'<exit-tool name="{self.name}">{self.instructions}</exit-tool>'

    def execute(self, *args: Any, **kwargs: Any) -> ExitToolResult:
        result = self.tool(*args, **kwargs)
        if not isinstance(result, ExitToolData):
            raise TypeError("exit tool must return ExitToolData")

        return result.dump()

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        super().validate_arguments(arguments)
        ExitToolData(**arguments)

    def _finalize_result(self, context: ToolContext, result: Any) -> ExitToolResult:
        result = super()._finalize_result(context, result)
        if not isinstance(result, dict):
            raise TypeError("Exit tool result must be a dictionary")
        return ExitToolData(**result).dump()

    def __str__(self) -> str:
        return f"Name: {self.name}\nType: ExitTool\nInstructions: {self.instructions[:50]}..."


class CalledExitTool:
    def __init__(self, exit_tool: ExitTool, tool_call: ToolCall) -> None:
        if not isinstance(exit_tool, ExitTool):
            raise TypeError("exit_tool must be an ExitTool")
        if not isinstance(tool_call, ToolCall):
            raise TypeError("tool_call must be a ToolCall")

        self._exit_tool = exit_tool
        self._tool_call = tool_call
        self._validate_matching()

    def _validate_matching(self) -> None:
        call_name = self._tool_call.get_name()
        if call_name != self.name:
            raise ValueError(
                f"Tool call name '{call_name}' does not match exit tool name '{self.name}'"
            )

    @property
    def name(self) -> str:
        return self._exit_tool.get_name()

    @property
    def call_id(self) -> str:
        return self._tool_call.get_id()

    @property
    def arguments(self) -> dict[str, Any]:
        return self._tool_call.get_arguments()

    @property
    def tool_call(self) -> ToolCall:
        return self._tool_call

    def validate_arguments(self) -> None:
        self._exit_tool.validate_arguments(self.arguments)

    def execute(
        self,
        *,
        work_root: Path | None = None,
        history: tuple[Message[object], ...] = (),
    ) -> ExitToolResult:
        return self._exit_tool.execute_with_hooks(
            self.arguments,
            call_id=self.call_id,
            work_root=work_root if work_root is not None else Path.cwd(),
            history=history,
        )
