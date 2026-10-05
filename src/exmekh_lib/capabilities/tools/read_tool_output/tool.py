from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from exmekh_lib.capabilities.tools.read_tool_output.summary import SummaryMode
from exmekh_lib.core.agents.tools.deep_tool import DeepTool
from exmekh_lib.core.models.tool import PreformattedToolResult, Tool

from .main import run


def create_tool(
    read_output: Callable[[str, int | None, SummaryMode], PreformattedToolResult],
    work_root: Path,
) -> Tool:
    return DeepTool(run, work_root, read_output, cache_result=False)
