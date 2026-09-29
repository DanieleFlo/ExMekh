"""Expose the node's execution history to the model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

from qs.config.runtime import RUNTIME
from qs.core.agents.tools.decorator import bind_deep_tool, deep_tool
from qs.core.models.tool import PreformattedToolResult, Tool
from qs.utils.xml import append_text_element, render_xml

if TYPE_CHECKING:
    from qs.core.cache.node_cache import NodeCache
    from qs.core.tool_calls.memory_tools import (
        ExecutionHistory,
        ExecutionHistoryEntry,
        OmittedHistoryRange,
    )


@deep_tool(
    name=RUNTIME.history_tool_name,
    description=(
        "Return navigable XML history of capability calls. Omit "
        "start_position for a head-and-tail overview, or use a "
        f"1-based position and limit from {RUNTIME.history_min_page_limit} "
        f"to {RUNTIME.memory_max_page_limit} to read a page."
    ),
    background_enabled=False,
)
def run(
    work_root: Path,
    read_history: Callable[[int | None, int], PreformattedToolResult],
    start_position: int | None = None,
    limit: int = RUNTIME.memory_default_page_limit,
) -> PreformattedToolResult:
    del work_root
    return read_history(start_position, limit)


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


def create_tool(
    read_history: Callable[[int | None, int], PreformattedToolResult],
    work_root: Path,
) -> Tool:
    return bind_deep_tool(run, work_root, read_history, cache_result=False)
