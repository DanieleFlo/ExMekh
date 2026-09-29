"""A runtime Bash tool with cancellable executions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qs.core.agents.tools.decorator import bind_deep_tool
from qs.core.models.tool import Tool, ToolExecutionOperation
from qs.interfaces.tool_hook import AfterTool, BeforeTool

from .bash_process import _BashProcess
from .main import run


class BashTool(Tool):
    """Launch a fresh Bash interface for each original tool call."""

    def __init__(
        self,
        *,
        executable: str | None = None,
        work_root: Path | None = None,
        cache_result: bool = True,
        before: BeforeTool | None = None,
        after: AfterTool | None = None,
    ) -> None:
        self._executable = executable
        bound = bind_deep_tool(
            run, work_root or Path.cwd(), executable, cache_result=cache_result
        )
        super().__init__(
            bound.tool,
            name=bound.name,
            description=bound.description,
            cache_result=cache_result,
            before=before,
            after=after,
        )

    def create_execution(
        self,
        arguments: dict[str, Any],
        *,
        call_id: str,
        work_root: Path,
        history: tuple[dict[str, str], ...],
    ) -> ToolExecutionOperation:
        del call_id, history
        command = arguments["command"]
        if not isinstance(command, str):
            raise TypeError("command must be a string")
        process = _BashProcess(command, work_root, self._executable)
        return ToolExecutionOperation(process.run, process.cancel, process.read_partial)
