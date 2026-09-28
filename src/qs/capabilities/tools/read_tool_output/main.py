"""Expose output reads to the model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qs.config.runtime import RUNTIME
from qs.core.models.tool import Tool

if TYPE_CHECKING:
    from qs.core.tool_calls.memory_tools import HarnessMemoryTools


def create_tool(memory: HarnessMemoryTools) -> Tool:
    return Tool(
        memory.read_tool_output,
        name=RUNTIME.read_tool_name,
        description=(
            "Read the current status and available output of a tool by "
            "output ID. Use an optional cursor to read only new output; "
            "omit it to read the complete output. Includes the final "
            "result when available."
        ),
        cache_result=False,
        background_enabled=False,
    )
