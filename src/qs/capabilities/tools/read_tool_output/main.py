"""Expose output reads to the model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from qs.config.runtime import RUNTIME
from qs.core.agents.tools.decorator import deep_tool
from qs.core.models.tool import PreformattedToolResult


@deep_tool(
    name=RUNTIME.read_tool_name,
    description=(
        "Read the current status and available output of a tool by "
        "output ID. Use an optional cursor to read only new output; "
        "omit it to read the complete output. Includes the final "
        "result when available."
    ),
    background_enabled=False,
)
def run(
    work_root: Path,
    read_output: Callable[[str, int | None], PreformattedToolResult],
    output_id: str,
    cursor: int | None = None,
) -> PreformattedToolResult:
    del work_root
    return read_output(output_id, cursor)
