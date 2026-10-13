"""Expose output reads to the model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from exmekh_lib.capabilities.tools.read_tool_output.summary import SummaryMode
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.models.tool import PreformattedToolResult


@deep_tool(
    name=RUNTIME.read_tool_name,
    description=(RUNTIME.read_tool_output_description),
    background_enabled=False,
)
def run(
    work_root: Path,
    read_output: Callable[[str, int | None, SummaryMode], PreformattedToolResult],
    output_id: str,
    cursor: int | None = None,
    summary_mode: SummaryMode = "none",
) -> PreformattedToolResult:
    del work_root
    return read_output(output_id, cursor, summary_mode)
