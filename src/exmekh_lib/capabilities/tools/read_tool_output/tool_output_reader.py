from __future__ import annotations

from collections.abc import Callable
from xml.etree import ElementTree as ET

from exmekh_lib.capabilities.tools.read_tool_output.summary import (
    SummaryMode,
    ToolOutputSummarizer,
)
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.tool_calls.execution_manager import ToolExecutionManager
from exmekh_lib.core.tool_calls.output_formatter import ToolOutputFormatter
from exmekh_lib.core.tool_calls.output_view import ToolOutputView
from exmekh_lib.utils.xml import append_text_element, render_xml


class ToolOutputReader:
    """Read node-local outputs and format their current state for the model."""

    def __init__(
        self,
        cache_provider: Callable[[], NodeCache],
        execution_manager: ToolExecutionManager | None = None,
        summarizer: ToolOutputSummarizer | None = None,
    ) -> None:
        self._cache_provider = cache_provider
        self._execution_manager = execution_manager
        self._summarizer = summarizer

    def read(
        self,
        output_id: str,
        cursor: int | None = None,
        summary_mode: SummaryMode = "none",
    ) -> PreformattedToolResult:
        """Read output; a successful final read acknowledges automatic delivery."""
        if cursor is not None and (type(cursor) is not int or cursor < 0):
            raise ValueError("cursor must be a non-negative integer or None")
        ToolOutputSummarizer.validate_mode(summary_mode)
        if summary_mode != "none" and self._summarizer is None:
            raise ValueError("Output summaries require a configured provider")
        result = self._read_output(output_id, cursor)
        return (
            self._summarizer.summarize(result, summary_mode)
            if self._summarizer is not None
            else result
        )

    def _read_output(
        self, output_id: str, cursor: int | None
    ) -> PreformattedToolResult:
        manager = self._execution_manager
        if manager is not None:
            runtime_output = self._read_runtime_output(manager, output_id, cursor)
            if runtime_output is not None:
                return runtime_output
        record = self._cache_provider().get(output_id)
        if record is not None:
            view = ToolOutputView(record)
            return PreformattedToolResult(
                view.format_saved(cursor), attachments=view.attachments()
            )
        root = ET.Element("saved-tool-output")
        append_text_element(root, "id", output_id)
        append_text_element(root, "found", "false")
        return PreformattedToolResult(render_xml(root))

    def _read_runtime_output(
        self, manager: ToolExecutionManager, output_id: str, cursor: int | None
    ) -> PreformattedToolResult | None:
        snapshot = manager.by_output_id(output_id)
        if snapshot is None:
            return None

        partial = None
        partial_error = None
        if cursor is not None or not snapshot.is_finished:
            try:
                partial = manager.partial_output(output_id, cursor or 0)
            except Exception as error:
                partial_error = str(error)
            else:
                if cursor is not None and partial is None:
                    raise ValueError("This tool output does not support cursors")

        content = ToolOutputFormatter.format_execution_output(
            output_id, snapshot, partial, partial_error=partial_error
        )
        result = PreformattedToolResult(
            content, attachments=snapshot.output_attachments()
        )
        if snapshot.is_finished:
            manager.mark_delivered(output_id)
        return result
