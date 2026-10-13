"""Expose the node's execution history to the model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.models.tool import PreformattedToolResult


@deep_tool(
    name=RUNTIME.history_tool_name,
    description=(RUNTIME.get_execution_history_description),
    background_enabled=False,
)
def run(
    work_root: Path,
    read_history: Callable[[int | None, int], PreformattedToolResult],
    start_position: int | None = None,
    limit: int = RUNTIME.memory_default_page_limit,
) -> PreformattedToolResult:
    del work_root
    return read_history(start_position, limit)
