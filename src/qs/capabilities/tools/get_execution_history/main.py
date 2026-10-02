"""Expose the node's execution history to the model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from qs.config.runtime import RUNTIME
from qs.core.agents.tools.decorator import deep_tool
from qs.core.models.tool import PreformattedToolResult


@deep_tool(
    name=RUNTIME.history_tool_name,
    description=(
        "Return navigable XML history of capability calls and saved user steering. Omit "
        "start_position for a head-and-tail overview, or use a "
        f"1-based position and limit from {RUNTIME.history_min_page_limit} "
        f"to {RUNTIME.memory_max_page_limit} to read a page."
    ),
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
