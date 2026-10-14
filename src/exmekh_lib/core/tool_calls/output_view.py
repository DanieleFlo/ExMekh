from __future__ import annotations

from dataclasses import replace
from typing import Any

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.config.runtime_text import RUNTIME_TEXT
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.cache.user_message_codec import deserialize_user_message
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.models.xml_header import XmlHeader
from exmekh_lib.utils.xml import (
    append_text_element,
    create_text_element,
    render_xml,
    stringify_tool_result,
)


class ToolOutputView:
    """Interpret output records for presentation without reading or changing state."""

    _TIMING_FIELDS = ("started_at", "finished_at", "elapsed_seconds")

    def __init__(self, record: ToolCallRecord) -> None:
        self._record = record

    def content(self) -> str:
        """Render a result, preserving the text of a completed skill."""
        record = self._record
        if record.external_output is not None:
            return record.external_output.render()
        if (
            record.source == "DeepSkill"
            and isinstance(record.result, dict)
            and isinstance(record.result.get("text"), str)
        ):
            return record.result["text"]
        return stringify_tool_result(record.result)

    def attachments(self) -> UserMessage | None:
        """Restore saved companion content and identify its associated output."""
        record = self._record
        if record.user_message is None:
            return None
        message = deserialize_user_message(record.user_message)
        if record.source == "UserMessage" and not message.has_attachments():
            return None
        return message.with_text_prefix(
            RUNTIME.saved_attachment_message.format(output_id=record.id)
        )

    def preview(self, limit: int) -> str:
        """Show useful progress or diagnostics before truncating a preview."""
        if type(limit) is not int or limit < 1:
            raise ValueError(RUNTIME_TEXT.preview_limit_must_be_a_positive_integer)
        preview = " ".join(self._preview_content().split())
        if len(preview) <= limit:
            return preview
        suffix = RUNTIME.preview_truncation_suffix[:limit]
        return preview[: limit - len(suffix)] + suffix

    def format_embedded(self) -> str:
        """Render a selected output with the shared XML escaping rules."""
        record = self._record
        root = create_text_element(
            "tool-output",
            "",
            attributes={"id": record.id, "name": record.name, "type": record.source},
        )
        append_text_element(root, "content", self.content())
        return render_xml(root)

    def _preview_content(self) -> str:
        record = self._record
        content = self.content()
        if (record.execution or {}).get("status") in (
            "running",
            "failed",
            "cancelled",
        ) and isinstance(record.result, dict):
            value = (
                record.result.get("partial_output")
                or record.result.get("error")
                or record.result.get("partial_output_error")
                or record.result.get("message", content)
            )
            return stringify_tool_result(value)
        return content

    def search_fields(self) -> dict[str, Any]:
        """Expose XML metadata without making callers interpret execution data."""
        record = self._record
        execution = record.execution or {}
        return {
            "name": record.name,
            "type": record.source,
            "id": record.id,
            "skill-path": record.skill_path,
            "response-type": execution.get(
                "response_type",
                self._response_type(execution.get("status", "completed")),
            ),
            "status": execution.get("status", "completed"),
            "storage": execution.get("storage", "saved"),
            **(
                {
                    "content-storage": "files",
                    "file-count": len(record.external_output.files),
                }
                if record.external_output is not None
                else {}
            ),
            "tool-call-id": record.tool_call_id,
            **self._detail_fields(execution, self._TIMING_FIELDS),
        }

    def saved_record(self) -> ToolCallRecord:
        """Describe stored work whose running state cannot prove it is still active."""
        record = self._record
        if record.execution is None:
            return record
        execution = self._saved_execution_info()
        result = (
            self._saved_content(execution)
            if execution["status"] == "unavailable" and record.external_output is None
            else record.result
        )
        return replace(record, result=result, execution=execution)

    def format_saved(self, cursor: int | None) -> str:
        """Render a saved read; cursors count result characters, excluding warnings."""
        record = self._record
        header = XmlHeader(
            source=record.source, name=record.name, id=record.id, args=record.args
        )
        if record.used_tools is not None:
            header.add_field(
                "used-tools", [reference.to_dict() for reference in record.used_tools]
            )
        execution = self._saved_execution_info()
        if execution:
            header.add_fields(
                **self._detail_fields(execution, ("tool_call_id", *self._TIMING_FIELDS))
            )
            header.add_field("status", execution["status"])
        header.add_field("response-type", execution.get("response_type", "final"))
        if cursor is not None and record.external_output is None:
            header.add_field("next-cursor", max(cursor, len(self.content())))
        return header.prepend_to(self._saved_content(execution, cursor))

    def _saved_execution_info(self) -> dict[str, Any]:
        if self._record.execution is None:
            return {}
        execution = dict(self._record.execution)
        status = execution.get("status", "completed")
        execution["status"] = "unavailable" if status == "running" else status
        current_status = execution["status"]
        execution["response_type"] = (
            "failed"
            if current_status == "unavailable"
            else execution.get("response_type", self._response_type(current_status))
        )
        execution["storage"] = "saved"
        return execution

    @staticmethod
    def _response_type(status: str) -> str:
        return {"completed": "final", "unavailable": "failed"}.get(status, status)

    def _saved_content(
        self, execution: dict[str, Any], cursor: int | None = None
    ) -> str:
        content = self.content()
        if cursor is not None and self._record.external_output is None:
            content = content[cursor:]
        if execution.get("status") == "unavailable":
            content = RUNTIME.unavailable_execution_message + "\n" + content
        return content

    @staticmethod
    def _detail_fields(
        execution: dict[str, Any], names: tuple[str, ...]
    ) -> dict[str, Any]:
        return {
            name.replace("_", "-"): execution[name]
            for name in names
            if name in execution
        }
