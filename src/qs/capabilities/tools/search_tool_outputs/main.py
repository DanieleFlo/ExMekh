"""Expose saved and running output search to the model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qs.config.runtime import RUNTIME
from qs.core.models.tool import Tool

if TYPE_CHECKING:
    from qs.core.tool_calls.memory_tools import HarnessMemoryTools
    from qs.core.tool_calls.search import ToolCallSearch


def create_tool(memory: HarnessMemoryTools) -> Tool:
    return Tool(
        memory.search_tool_outputs,
        name=RUNTIME.search_tool_name,
        description=(
            "Search this node's saved and runtime tool outputs, including "
            "background work. Return paginated XML previews with status "
            "and output IDs. All filters are optional."
        ),
        cache_result=False,
        background_enabled=False,
    )


def create_raw_search_tool(search: ToolCallSearch) -> Tool:
    """Expose the structured search adapter for callers that need raw records."""
    return Tool(
        search.search,
        name=RUNTIME.search_tool_name,
        cache_result=False,
        background_enabled=False,
    )
