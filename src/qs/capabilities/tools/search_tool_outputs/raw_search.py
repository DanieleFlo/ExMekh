"""Structured search adapter for callers outside the model-facing XML tool."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from qs.config.runtime import RUNTIME
from qs.core.agents.tools.decorator import deep_tool


@deep_tool(
    name=RUNTIME.search_tool_name,
    description="Search saved and running tool outputs as structured records.",
    background_enabled=False,
)
def run(
    work_root: Path,
    search_records: Callable[[str, str | None, str | None, int, int], dict[str, Any]],
    query: str = "",
    tool_name: str | None = None,
    skill_path: str | None = None,
    offset: int = 0,
    limit: int = RUNTIME.memory_default_page_limit,
) -> dict[str, Any]:
    del work_root
    return search_records(query, tool_name, skill_path, offset, limit)
