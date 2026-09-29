from __future__ import annotations

from pathlib import Path

from qs.core.agents.tools.deep_tool import DeepTool
from qs.core.models.tool import Tool

from .main import SearchOutputs, run


def create_tool(search_outputs: SearchOutputs, work_root: Path) -> Tool:
    return DeepTool(run, work_root, search_outputs, cache_result=False)
