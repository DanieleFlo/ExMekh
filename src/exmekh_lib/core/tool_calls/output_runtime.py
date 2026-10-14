"""Publish prepared execution output through the manager's versioned public API."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from exmekh_lib.config.runtime_text import RUNTIME_TEXT
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.models.tool_output_presentation import ToolOutputPresentation
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.tool_calls.execution_manager import (
    ExecutionSnapshot,
    ToolExecutionManager,
)
from exmekh_lib.core.tool_calls.output_formatter import ToolOutputFormatter
from exmekh_lib.core.tool_calls.output_preparer import (
    PreparedToolOutput,
    ToolOutputPreparer,
)
from exmekh_lib.utils.xml import stringify_tool_result


@dataclass(frozen=True, slots=True)
class ToolOutputDelivery:
    result: PreformattedToolResult
    execution: ExecutionSnapshot | None = None
    failure: dict[str, Any] | None = None

    @property
    def final_execution(self) -> ExecutionSnapshot | None:
        if (
            self.failure is not None
            or self.result.delivery_failed
            or self.execution is None
            or not self.execution.is_finished
        ):
            return None
        return self.execution


class ToolOutputRuntime:
    def __init__(
        self,
        manager: ToolExecutionManager | None = None,
        preparer: ToolOutputPreparer | None = None,
    ) -> None:
        self._manager = manager
        self._preparer = preparer

    def publish_final(
        self,
        execution: ExecutionSnapshot,
        presentation: ToolOutputPresentation | None = None,
    ) -> ToolOutputDelivery:
        """Prepare outside the manager lock, then publish only the same terminal state."""
        if self._manager is None:
            raise ValueError(RUNTIME_TEXT.execution_manager_required)
        if not execution.is_finished:
            raise ValueError(RUNTIME_TEXT.final_output_required)
        while True:
            identity = execution.output_reference()
            result = (
                presentation.result
                if presentation is not None
                else PreformattedToolResult(
                    ToolOutputFormatter.format_execution_output(
                        identity.id, execution, None
                    ),
                    attachments=execution.output_attachments(),
                    external_output=execution.external_output,
                    delivery_failed=execution.delivery_failed,
                )
            )
            prepared = (
                self._preparer.prepare(
                    presentation or ToolOutputPresentation(identity, result)
                )
                if self._preparer is not None
                else PreparedToolOutput(result.content, result.external_output)
            )
            current = self._manager.publish_output(execution, prepared.external_output)
            if not current.same_version(execution):
                execution = current
                presentation = None
                continue
            return ToolOutputDelivery(
                prepared.as_result(result), current, prepared.failure
            )

    def prepare_records(
        self, stored: Mapping[str, ToolCallRecord]
    ) -> tuple[ToolCallRecord, ...]:
        """Prepare searchable views without acknowledging delivery or saving JSON records."""
        records = dict(stored)
        if self._manager is not None:
            for execution in self._manager.output_snapshots():
                if execution.is_finished:
                    self.publish_final(execution)
            records.update(
                {record.id: record for record in self._manager.output_records(stored)}
            )
        if self._preparer is None:
            return tuple(records.values())
        prepared_records: list[ToolCallRecord] = []
        for record in records.values():
            if record.external_output is not None:
                prepared_records.append(record)
                continue
            presentation = ToolOutputPresentation(
                ToolOutputReference(record.name, record.id),
                PreformattedToolResult(stringify_tool_result(record.result)),
            )
            prepared = self._preparer.prepare(presentation)
            if prepared.external_output is not None:
                record = record.externalized(prepared.external_output)
            elif prepared.failure is not None:
                record = record.with_delivery_failure(prepared.failure)
            prepared_records.append(record)
        return tuple(prepared_records)
