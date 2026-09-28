from __future__ import annotations

from typing import Any, Literal

from qs.config.runtime import RUNTIME
from qs.core.models.tool import Tool
from qs.core.tool_calls.execution_manager import ToolExecutionManager


class ExecutionControlTool(Tool):
    """Expose bounded waits and cancellation without owning conversation state."""

    def __init__(self, manager: ToolExecutionManager) -> None:
        self._manager = manager
        super().__init__(
            self.control_tool_execution,
            name=RUNTIME.execution_tool_name,
            description=(
                "Wait for a tool execution by its original tool_call_id, or request "
                "cancellation. Waiting returns as soon as the work finishes or "
                "the wait limit expires; read its output with read_tool_output."
            ),
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
