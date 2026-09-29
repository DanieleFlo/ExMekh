from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from qs.config.runtime import RUNTIME
from qs.core.agents.tools.decorator import deep_tool


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
