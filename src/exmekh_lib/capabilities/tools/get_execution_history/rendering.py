from __future__ import annotations

from xml.etree import ElementTree as ET

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.tool_calls.execution_history import (
    ExecutionHistory,
    ExecutionHistoryEntry,
    OmittedHistoryRange,
)
from exmekh_lib.utils.xml import append_text_element, render_xml


def render_execution_history(
    history: ExecutionHistory,
    cache: NodeCache,
    start_position: int | None = None,
    limit: int = RUNTIME.memory_default_page_limit,
) -> PreformattedToolResult:
    view = history.select(start_position=start_position, limit=limit)
    root = ET.Element("execution-history")
    append_text_element(root, "total", str(view.total))
    append_text_element(root, "view", view.mode)
    for name, value in (
        ("start-position", view.start_position),
        ("end-position", view.end_position),
        ("previous-position", view.previous_position),
        ("next-position", view.next_position),
    ):
        if value is not None:
            append_text_element(root, name, str(value))
    for entry in view.entries:
        if view.omitted is not None and entry.position == view.omitted.end_position + 1:
            _append_omitted(root, view.omitted)
        _append_history_entry(root, entry, cache)
    return PreformattedToolResult(render_xml(root))


def _append_history_entry(
    root: ET.Element, entry: ExecutionHistoryEntry, cache: NodeCache
) -> None:
    call = ET.SubElement(root, "call")
    append_text_element(call, "position", str(entry.position))
    append_text_element(call, "name", entry.name)
    append_text_element(call, "id", entry.call_id)
    output_id = entry.recoverable_output_id(cache)
    if output_id is not None:
        append_text_element(call, "saved-output-id", output_id)


def _append_omitted(root: ET.Element, omitted: OmittedHistoryRange) -> None:
    element = ET.SubElement(root, "omitted")
    append_text_element(element, "from-position", str(omitted.start_position))
    append_text_element(element, "to-position", str(omitted.end_position))
    append_text_element(element, "count", str(omitted.count))
