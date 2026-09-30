from __future__ import annotations

from collections.abc import Callable
from xml.etree import ElementTree as ET

from qs.core.cache.node_cache import NodeCache
from qs.core.models.tool import PreformattedToolResult
from qs.core.tool_calls.execution_manager import ToolExecutionManager
from qs.core.tool_calls.output_formatter import ToolOutputFormatter
from qs.core.tool_calls.output_view import ToolOutputView
from qs.utils.xml import append_text_element, render_xml


class ToolOutputReader:
    """Read node-local outputs and format their current state for the model."""

    def __init__(
        self,
        cache_provider: Callable[[], NodeCache],
        execution_manager: ToolExecutionManager | None = None,
    ) -> None:
        self._cache_provider = cache_provider
        self._execution_manager = execution_manager

    def read(self, output_id: str, cursor: int | None = None) -> PreformattedToolResult:
        """Read output; a successful final read acknowledges automatic delivery."""
        if cursor is not None and (type(cursor) is not int or cursor < 0):
            raise ValueError("cursor must be a non-negative integer or None")
        manager = self._execution_manager
        if manager is not None:
            active_output = self._read_active_output(manager, output_id, cursor)
            if active_output is not None:
                return active_output

        record = self._cache_provider().get(output_id)
        if record is not None:
            return PreformattedToolResult(ToolOutputView(record).format_saved(cursor))
        root = ET.Element("saved-tool-output")
        append_text_element(root, "id", output_id)
        append_text_element(root, "found", "false")
        return PreformattedToolResult(render_xml(root))

    def _read_active_output(
        self, manager: ToolExecutionManager, output_id: str, cursor: int | None
    ) -> PreformattedToolResult | None:
        snapshot = manager.by_output_id(output_id)
        if snapshot is None:
            return None

        partial = None
        if cursor is not None or not snapshot.is_finished:
            partial = manager.partial_output(output_id, cursor or 0)
            if cursor is not None and partial is None:
                raise ValueError("This tool output does not support cursors")

        content = ToolOutputFormatter.format_execution_output(
            output_id, snapshot, partial
        )
        if snapshot.is_finished:
            manager.mark_delivered(output_id)
        return PreformattedToolResult(content)
