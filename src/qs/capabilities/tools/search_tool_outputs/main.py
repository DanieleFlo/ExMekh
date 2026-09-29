"""Expose saved and running output search to the model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

from qs.config.runtime import RUNTIME
from qs.core.agents.tools.decorator import bind_deep_tool, deep_tool
from qs.core.models.tool import PreformattedToolResult, Tool
from qs.core.tool_calls.output_view import ToolOutputView
from qs.utils.xml import append_text_element, render_xml

from .search import ToolCallSearch

if TYPE_CHECKING:
    from qs.core.cache.node_cache import NodeCache
    from qs.core.cache.tool_call_record import ToolCallRecord
    from qs.core.tool_calls.execution_manager import ToolExecutionManager


type SearchOutputs = Callable[
    [str, str | None, str | None, int, int], PreformattedToolResult
]


@deep_tool(
    name=RUNTIME.search_tool_name,
    description=(
        "Search this node's saved and runtime tool outputs, including "
        "background work. Return paginated XML previews with status "
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


def render_search_results(
    cache: NodeCache,
    execution_manager: ToolExecutionManager | None,
    query: str = "",
    tool_name: str | None = None,
    skill_path: str | None = None,
    offset: int = 0,
    limit: int = RUNTIME.memory_default_page_limit,
) -> PreformattedToolResult:
    records, total, next_offset = ToolCallSearch(
        cache, execution_manager
    ).search_records(query, tool_name, skill_path, offset, limit)
    root = ET.Element("tool-outputs")
    append_text_element(root, "total", str(total))
    if next_offset is not None:
        append_text_element(root, "next-offset", str(next_offset))
    records_element = ET.SubElement(root, "records")
    for record in records:
        _append_search_record(records_element, record)
    return PreformattedToolResult(render_xml(root))


def _append_search_record(root: ET.Element, record: ToolCallRecord) -> None:
    element = ET.SubElement(root, "record")
    output = ToolOutputView(record)
    for name, value in output.search_fields().items():
        if value is not None:
            append_text_element(element, name, str(value))
    append_text_element(
        element, "preview", output.preview(RUNTIME.memory_preview_limit)
    )


def create_tool(search_outputs: SearchOutputs, work_root: Path) -> Tool:
    return bind_deep_tool(run, work_root, search_outputs, cache_result=False)
