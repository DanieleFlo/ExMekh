from __future__ import annotations

from pathlib import Path

from exmekh_lib.core.agents.tools.deep_tool import DeepTool
from exmekh_lib.core.models.tool import Tool

from .main import SearchOutputs, run


def create_tool(search_outputs: SearchOutputs, work_root: Path) -> Tool:
    return DeepTool(run, work_root, search_outputs, cache_result=False)
