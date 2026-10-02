from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal
from uuid import uuid4

from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.models.xml_header import XmlHeader
from exmekh_lib.core.tool_calls.execution_manager import ExecutionSnapshot
from exmekh_lib.utils.xml import stringify_tool_result

logger = logging.getLogger(__name__)


class ToolOutputFormatter:
    """Render tool outputs and skill returns without changing execution state."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def generate_output_id(self) -> str:
        try:
            output_id = self._client.generate_id()
            if not isinstance(output_id, str) or not output_id.strip():
                raise ValueError("Provider generated an invalid output ID")
            return output_id.strip()
        except Exception as error:
            logger.warning("Output ID unavailable; using a UUID: %s", error)
            return uuid4().hex

    def format_tool_result(
        self,
        name: str,
        result: Any,
        *,
        execution: ExecutionSnapshot | None = None,
        annotation_provider: Callable[[], str | None] | None = None,
    ) -> tuple[str, str | None]:
        if isinstance(result, PreformattedToolResult) and execution is None:
            content = result.content
            if XmlHeader.has_leading_header(content):
                annotation = self._thinking_annotation(None, annotation_provider)
                if annotation:
                    content = XmlHeader.add_field_to_content(
                        content, "thinking", annotation
                    )
            return content, None
        content = (
            result.content
            if isinstance(result, PreformattedToolResult)
            else stringify_tool_result(result)
        )
        header = (
            self._new_header("Tool", name)
            if execution is None
            else self._execution_header(
                execution.output_id or self.generate_output_id(), execution
            )
        )
        header.add_field(
            "response-type", "final" if execution is None else execution.response_type
        )
        return self._format_output(header, content, None, annotation_provider)

    @classmethod
    def format_execution_output(
        cls,
        output_id: str,
        execution: ExecutionSnapshot,
        partial: tuple[str, int] | None,
        *,
        partial_error: str | None = None,
    ) -> str:
        """Render a read with the same execution metadata as its immediate reply."""
        header = cls._execution_header(output_id, execution)
        if partial_error is not None:
            header.add_field("partial-output-error", partial_error)
        if partial is None:
            content = execution.output_content()
            response_type = execution.response_type
        else:
            content, next_cursor = partial
            header.add_field("next-cursor", next_cursor)
            response_type = "partial" if content else "running"
            if execution.is_finished:
                response_type = execution.response_type
                final = execution.output_content()
                content = f"{content}\n{final}" if content else final
        header.add_field("response-type", response_type)
        return header.prepend_to(content)

    @staticmethod
    def _execution_header(output_id: str, execution: ExecutionSnapshot) -> XmlHeader:
        header = XmlHeader(
            source="Tool", name=execution.name, id=output_id, args=execution.arguments
        )
        for name, value in execution.status_info().items():
            if name != "output_id":
                header.add_field(name.replace("_", "-"), value)
        return header

    def format_skill_return(
        self,
        name: str,
        content: str,
        *,
        thinking: str | None = None,
        annotation_provider: Callable[[], str | None] | None = None,
        background_tools: Sequence[ExecutionSnapshot] = (),
        used_tools: Sequence[ToolOutputReference] = (),
    ) -> tuple[str, str]:
        header = self._new_header("DeepSkill", name)
        header.add_field("response-type", "final")
        header.add_field(
            "used-tools", [reference.to_dict() for reference in used_tools]
        )
        if background_tools:
            header.add_field(
                "background-tools",
                [
                    {
                        "name": snapshot.name,
                        "tool-call-id": snapshot.call_id,
                        "output-id": snapshot.output_id,
                        "skill-path": snapshot.skill_path,
                        "status": snapshot.status,
                    }
                    for snapshot in background_tools
                ],
            )
        return self._format_output(header, content, thinking, annotation_provider)

    def _new_header(self, source: Literal["Tool", "DeepSkill"], name: str) -> XmlHeader:
        return XmlHeader(
            source=source,
            name=name,
            id=self.generate_output_id(),
        )

    def _format_output(
        self,
        header: XmlHeader,
        content: str,
        thinking: str | None,
        annotation_provider: Callable[[], str | None] | None,
    ) -> tuple[str, str]:
        annotation = self._thinking_annotation(thinking, annotation_provider)
        if annotation:
            header.add_field("thinking", annotation)
        return header.prepend_to(content), header.id

    def _thinking_annotation(
        self,
        thinking: str | None,
        annotation_provider: Callable[[], str | None] | None,
    ) -> str | None:
        try:
            if annotation_provider is not None:
                annotation = annotation_provider()
            elif thinking and thinking.strip():
                annotation = self._client.get_tokenizer().truncate_text(thinking, 100)
            else:
                return None
            return annotation if annotation and annotation.strip() else None
        except Exception as error:
            logger.warning("Thinking annotation unavailable: %s", error)
            return None
