"""A runtime Bash tool with cancellable executions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.tools.deep_tool import DeepTool
from exmekh_lib.core.models.tool import ToolExecutionOperation
from exmekh_lib.interfaces.message import Message
from exmekh_lib.interfaces.tool_hook import AfterTool, BeforeTool

from .bash_process import BashProcess
from .main import run


class BashTool(DeepTool):
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
        super().__init__(
            run,
            work_root or Path.cwd(),
            executable,
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
        history: tuple[Message[object], ...],
    ) -> ToolExecutionOperation:
        del call_id, history
        command = arguments["command"]
        if not isinstance(command, str):
            raise TypeError(RUNTIME.bash_command_required)
        process = BashProcess(command, work_root, self._executable)
        return ToolExecutionOperation(process.run, process.cancel, process.read_partial)
