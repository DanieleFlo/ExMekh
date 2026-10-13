from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.tools.decorator import deep_tool


@deep_tool(
    name=RUNTIME.execution_tool_name,
    description=(RUNTIME.control_tool_execution_description),
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
