from __future__ import annotations

from pathlib import Path

from qs.core.agents.tools.decorator import bind_deep_tool
from qs.core.models.tool import Tool

from .main import SearchOutputs, run


def create_tool(search_outputs: SearchOutputs, work_root: Path) -> Tool:
    return bind_deep_tool(run, work_root, search_outputs, cache_result=False)
