from __future__ import annotations

from collections.abc import Callable
from xml.etree import ElementTree as ET

from qs.core.cache.node_cache import NodeCache
from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.models.tool import PreformattedToolResult
from qs.core.models.xml_header import XmlHeader
from qs.utils.json import stringify_tool_result

from .execution_manager import ExecutionSnapshot, ToolExecutionManager


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
            return PreformattedToolResult(self._read_saved(record, cursor))
        root = ET.Element("saved-tool-output")
        ET.SubElement(root, "id").text = output_id
        ET.SubElement(root, "found").text = "false"
        ET.indent(root)
        return PreformattedToolResult(ET.tostring(root, encoding="unicode"))

    @staticmethod
    def _read_saved(record: ToolCallRecord, cursor: int | None) -> str:
        header = XmlHeader(
            source=record.source, name=record.name, id=record.id, args=record.args
        )
        execution = record.execution or {}
        status = execution.get("status", "completed")
        if record.execution is not None:
            for name in (
                "tool_call_id",
                "started_at",
                "finished_at",
                "elapsed_seconds",
            ):
                if name in execution:
                    header.add_field(name.replace("_", "-"), execution[name])
            header.add_field("status", "unavailable" if status == "running" else status)
        response_type = "final" if status == "completed" else status
        if status == "running":
            response_type = "failed"
        header.add_field("response-type", response_type)
        content = tool_output_content(record)
        if cursor is not None:
            header.add_field("next-cursor", len(content))
            content = content[cursor:]
        if status == "running":
            content = (
                "Execution is unavailable in this runtime; it was not resumed. "
                "The saved running response is not a final result.\n" + content
            )
        return header.prepend_to(content)

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
            content = self._execution_content(snapshot)
            response_type = snapshot.response_type
        else:
            content, next_cursor = partial
            header.add_field("next-cursor", next_cursor)
            response_type = "partial" if content else "running"
            if snapshot.is_finished:
                response_type = snapshot.response_type
                final = self._execution_content(snapshot)
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

    @staticmethod
    def _execution_content(snapshot: ExecutionSnapshot) -> str:
        if snapshot.status == "completed":
            if isinstance(snapshot.result, PreformattedToolResult):
                return snapshot.result.content
            return stringify_tool_result(snapshot.result)
        if snapshot.error is not None:
            return snapshot.error
        if snapshot.status == "cancelled":
            return "Tool execution was cancelled."
        return "Tool execution is still running."


def tool_output_content(record: ToolCallRecord) -> str:
    """Render saved content consistently for memory reads and skill returns."""
    if (
        record.source == "DeepSkill"
        and isinstance(record.result, dict)
        and isinstance(record.result.get("text"), str)
    ):
        return record.result["text"]
    return stringify_tool_result(record.result)
