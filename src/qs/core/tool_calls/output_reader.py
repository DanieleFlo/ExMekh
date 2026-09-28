from __future__ import annotations

from collections.abc import Callable
from xml.etree import ElementTree as ET

from qs.core.cache.node_cache import NodeCache
from qs.core.models.tool import PreformattedToolResult
from qs.core.models.xml_header import XmlHeader

from .execution_manager import ExecutionSnapshot, ToolExecutionManager
from .output_view import ToolOutputView


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
        snapshot = (
            self._execution_manager.by_output_id(output_id)
            if self._execution_manager is not None
            else None
        )
        if snapshot is not None:
            content = self._read_execution(output_id, snapshot, cursor)
            if snapshot.is_finished:
                self._execution_manager.mark_delivered(output_id)
            return PreformattedToolResult(content)

        record = self._cache_provider().get(output_id)
        if record is not None:
            return PreformattedToolResult(ToolOutputView(record).format_saved(cursor))
        root = ET.Element("saved-tool-output")
        ET.SubElement(root, "id").text = output_id
        ET.SubElement(root, "found").text = "false"
        ET.indent(root)
        return PreformattedToolResult(ET.tostring(root, encoding="unicode"))

    def _read_execution(
        self, output_id: str, snapshot: ExecutionSnapshot, cursor: int | None
    ) -> str:
        header = XmlHeader(
            source="Tool", name=snapshot.name, id=output_id, args=snapshot.arguments
        )
        header.add_fields(
            **{
                "status": snapshot.status,
                "tool-call-id": snapshot.call_id,
                "started-at": snapshot.started_at.isoformat(),
                "elapsed-seconds": snapshot.elapsed_seconds,
                "finished-at": (
                    snapshot.finished_at.isoformat()
                    if snapshot.finished_at is not None
                    else None
                ),
            }
        )
        partial = self._read_partial(output_id, snapshot, cursor)
        if partial is None:
            content = snapshot.output_content()
            response_type = snapshot.response_type
        else:
            content, next_cursor = partial
            header.add_field("next-cursor", next_cursor)
            response_type = "partial" if content else "running"
            if snapshot.is_finished:
                response_type = snapshot.response_type
                final = snapshot.output_content()
                content = f"{content}\n{final}" if content else final
        header.add_field("response-type", response_type)
        return header.prepend_to(content)

    def _read_partial(
        self, output_id: str, snapshot: ExecutionSnapshot, cursor: int | None
    ) -> tuple[str, int] | None:
        if cursor is None and snapshot.is_finished:
            return None
        partial = self._execution_manager.partial_output(output_id, cursor or 0)
        if cursor is not None and partial is None:
            raise ValueError("This tool output does not support cursors")
        return partial
