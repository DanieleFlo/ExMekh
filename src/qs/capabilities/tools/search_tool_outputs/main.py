"""Expose saved and running output search to the model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from qs.config.runtime import RUNTIME
from qs.core.agents.tools.decorator import deep_tool
from qs.core.models.tool import PreformattedToolResult

type SearchOutputs = Callable[
    [str, str | None, str | None, int, int], PreformattedToolResult
]


@deep_tool(
    name=RUNTIME.search_tool_name,
    description=(
        "Search this node's saved and runtime tool outputs, including "
        "background work and user steering saved inside skills. Return paginated XML previews with status "
        "and output IDs. All filters are optional."
    ),
    background_enabled=False,
)
def run(
    work_root: Path,
    search_outputs: SearchOutputs,
    query: str = "",
    tool_name: str | None = None,
    skill_path: str | None = None,
    offset: int = 0,
    limit: int = RUNTIME.memory_default_page_limit,
) -> PreformattedToolResult:
    del work_root
    return search_outputs(query, tool_name, skill_path, offset, limit)
