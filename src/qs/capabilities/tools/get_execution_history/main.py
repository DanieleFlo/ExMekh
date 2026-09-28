"""Expose the node's execution history to the model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qs.config.runtime import RUNTIME
from qs.core.models.tool import Tool

if TYPE_CHECKING:
    from qs.core.tool_calls.memory_tools import HarnessMemoryTools


def create_tool(memory: HarnessMemoryTools) -> Tool:
    return Tool(
        memory.get_execution_history,
        name=RUNTIME.history_tool_name,
        description=(
            "Return navigable XML history of capability calls. Omit "
            "start_position for a head-and-tail overview, or use a "
            f"1-based position and limit from {RUNTIME.history_min_page_limit} "
            f"to {RUNTIME.memory_max_page_limit} to read a page."
        ),
        cache_result=False,
        background_enabled=False,
    )
