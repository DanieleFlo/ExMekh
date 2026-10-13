from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.tools.deep_tool import DeepTool
from exmekh_lib.core.tool_calls.execution_manager import ToolExecutionManager

from .main import run


class ExecutionControlTool(DeepTool):
    """Expose bounded waits and cancellation without owning conversation state."""

    def __init__(
        self, manager: ToolExecutionManager, work_root: Path | None = None
    ) -> None:
        self._manager = manager
        super().__init__(
            run,
            work_root or Path.cwd(),
            self.control_tool_execution,
            cache_result=False,
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
                result["message"] = RUNTIME.cancellation_unsupported
            return result
        raise ValueError(RUNTIME.invalid_control_action)
