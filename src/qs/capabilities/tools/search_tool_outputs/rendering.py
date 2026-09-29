from __future__ import annotations

from xml.etree import ElementTree as ET

from qs.config.runtime import RUNTIME
from qs.core.cache.node_cache import NodeCache
from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.models.tool import PreformattedToolResult
from qs.core.tool_calls.execution_manager import ToolExecutionManager
from qs.core.tool_calls.output_view import ToolOutputView
from qs.utils.xml import append_text_element, render_xml

from .tool_call_search import ToolCallSearch


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
