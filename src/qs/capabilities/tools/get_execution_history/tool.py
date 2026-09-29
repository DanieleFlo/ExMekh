from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from qs.core.agents.tools.decorator import bind_deep_tool
from qs.core.models.tool import PreformattedToolResult, Tool

from .main import run


def create_tool(
    read_history: Callable[[int | None, int], PreformattedToolResult],
    work_root: Path,
) -> Tool:
    return bind_deep_tool(run, work_root, read_history, cache_result=False)
