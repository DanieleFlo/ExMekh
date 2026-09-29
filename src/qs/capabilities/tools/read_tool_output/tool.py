from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from qs.core.agents.tools.deep_tool import DeepTool
from qs.core.models.tool import PreformattedToolResult, Tool

from .main import run


def create_tool(
    read_output: Callable[[str, int | None], PreformattedToolResult],
    work_root: Path,
) -> Tool:
    return DeepTool(run, work_root, read_output, cache_result=False)
