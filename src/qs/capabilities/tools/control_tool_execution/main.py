from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from qs.config.runtime import RUNTIME
from qs.core.agents.tools.decorator import bind_deep_tool, deep_tool
from qs.core.models.tool import Tool
from qs.core.tool_calls.execution_manager import ToolExecutionManager


@deep_tool(
    name=RUNTIME.execution_tool_name,
    description=(
        "Wait for a tool execution by its original tool_call_id, or request "
        "cancellation. Waiting returns as soon as the work finishes or "
        "the wait limit expires; read its output with read_tool_output."
    ),
    background_enabled=False,
)
def run(
    work_root: Path,
    control: Callable[[str, Literal["wait", "cancel"], float], dict[str, Any]],
    tool_call_id: str,
    action: Literal["wait", "cancel"] = "wait",
    wait_seconds: float = RUNTIME.default_tool_wait_seconds,
) -> dict[str, Any]:
    del work_root
    return control(tool_call_id, action, wait_seconds)


class ExecutionControlTool(Tool):
    """Expose bounded waits and cancellation without owning conversation state."""

    def __init__(
        self, manager: ToolExecutionManager, work_root: Path | None = None
    ) -> None:
        self._manager = manager
        bound = bind_deep_tool(
            run,
            work_root or Path.cwd(),
            self.control_tool_execution,
            cache_result=False,
        )
        super().__init__(
            bound.tool,
            name=bound.name,
            description=bound.description,
            cache_result=False,
            background_enabled=False,
        )

    def control_tool_execution(
        self,
        tool_call_id: str,
        action: Literal["wait", "cancel"] = "wait",
        wait_seconds: float = RUNTIME.default_tool_wait_seconds,
    ) -> dict[str, Any]:
        if action == "wait":
            return self._manager.wait(tool_call_id, wait_seconds).status_info()
        if action == "cancel":
            supported, cancelled, snapshot = self._manager.cancel(tool_call_id)
            result = {
                **snapshot.status_info(),
                "cancellation_supported": supported,
                "cancelled": cancelled,
            }
            if not supported:
                result["message"] = "This tool does not support cancellation."
            return result
        raise ValueError("action must be 'wait' or 'cancel'")
