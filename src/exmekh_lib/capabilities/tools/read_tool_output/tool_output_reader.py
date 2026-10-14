from __future__ import annotations

from collections.abc import Callable
from xml.etree import ElementTree as ET

from exmekh_lib.capabilities.tools.read_tool_output.summary import (
    SummaryMode,
    ToolOutputSummarizer,
)
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.config.tool_errors import ERROR_HINTS
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.models.tool_output_presentation import ToolOutputPresentation
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.tool_calls.execution_manager import ToolExecutionManager
from exmekh_lib.core.tool_calls.output_formatter import ToolOutputFormatter
from exmekh_lib.core.tool_calls.output_preparer import ToolOutputPreparer
from exmekh_lib.core.tool_calls.output_runtime import (
    ToolOutputDelivery,
    ToolOutputRuntime,
)
from exmekh_lib.core.tool_calls.output_view import ToolOutputView
from exmekh_lib.utils.tool_errors import ToolInputError
from exmekh_lib.utils.xml import append_text_element, render_xml


class ToolOutputReader:
    """Read node-local outputs and format their current state for the model."""

    def __init__(
        self,
        cache_provider: Callable[[], NodeCache],
        execution_manager: ToolExecutionManager | None = None,
        summarizer: ToolOutputSummarizer | None = None,
        output_preparer: ToolOutputPreparer | None = None,
    ) -> None:
        self._cache_provider = cache_provider
        self._execution_manager = execution_manager
        self._summarizer = summarizer
        self._output_preparer = output_preparer
        self._output_runtime = (
            ToolOutputRuntime(execution_manager, output_preparer)
            if execution_manager is not None
            else None
        )

    def read(
        self,
        output_id: str,
        cursor: int | None = None,
        summary_mode: SummaryMode = "none",
    ) -> PreformattedToolResult:
        """Read output; a successful final read acknowledges automatic delivery."""
        if not isinstance(output_id, str) or not output_id.strip():
            raise ToolInputError(RUNTIME.output_id_required, code="invalid_output_id")
        if cursor is not None and (type(cursor) is not int or cursor < 0):
            raise ToolInputError(
                RUNTIME.invalid_output_cursor, code="invalid_output_cursor"
            )
        ToolOutputSummarizer.validate_mode(summary_mode)
        if summary_mode != "none" and self._summarizer is None:
            raise ToolInputError(
                RUNTIME.summary_provider_required, code="summary_provider_required"
            )
        delivery = self._read_output(output_id, cursor)
        result = delivery.result
        result = (
            self._summarizer.summarize(result, summary_mode)
            if self._summarizer is not None
            else result
        )
        if result is not delivery.result:
            result = self._prepare_read(
                ToolOutputReference(name=self._output_name(delivery), id=output_id),
                result,
            )
        if (
            not result.delivery_failed
            and self._execution_manager is not None
            and delivery.final_execution is not None
        ):
            self._execution_manager.acknowledge_output(delivery.final_execution)
        return result

    @staticmethod
    def _output_name(delivery: ToolOutputDelivery) -> str:
        return (
            delivery.execution.name
            if delivery.execution is not None
            else RUNTIME.read_tool_name
        )

    def _prepare_read(
        self, identity: ToolOutputReference, result: PreformattedToolResult
    ) -> PreformattedToolResult:
        if self._output_preparer is None or result.delivery_failed:
            return result
        return self._output_preparer.prepare(
            ToolOutputPresentation(identity, result)
        ).as_result(result)

    def _read_output(self, output_id: str, cursor: int | None) -> ToolOutputDelivery:
        manager = self._execution_manager
        if manager is not None:
            runtime_output = self._read_runtime_output(manager, output_id, cursor)
            if runtime_output is not None:
                return runtime_output
        record = self._cache_provider().get(output_id)
        if record is not None:
            if record.external_output is None and self._output_preparer is not None:
                original = PreformattedToolResult(
                    ToolOutputView(record).format_saved(None),
                    attachments=ToolOutputView(record).attachments(),
                )
                prepared = self._output_preparer.prepare(
                    ToolOutputPresentation(
                        ToolOutputReference(record.name, record.id), original
                    )
                )
                if prepared.failure is not None:
                    return ToolOutputDelivery(prepared.as_result(original))
                if prepared.external_output is not None:
                    record = record.externalized(prepared.external_output)
                    self._cache_provider().store(record)
            view = ToolOutputView(record)
            return ToolOutputDelivery(
                PreformattedToolResult(
                    view.format_saved(cursor),
                    attachments=view.attachments(),
                    external_output=record.external_output,
                )
            )
        root = ET.Element("saved-tool-output")
        append_text_element(root, "id", output_id)
        append_text_element(root, "found", "false")
        append_text_element(root, "error_code", "output_not_found")
        append_text_element(root, "hint", ERROR_HINTS["output_not_found"])
        return ToolOutputDelivery(PreformattedToolResult(render_xml(root)))

    def _read_runtime_output(
        self, manager: ToolExecutionManager, output_id: str, cursor: int | None
    ) -> ToolOutputDelivery | None:
        snapshot = manager.by_output_id(output_id)
        if snapshot is None:
            return None

        partial = None
        partial_error = None
        if snapshot.is_finished and self._output_runtime is not None:
            published = self._output_runtime.publish_final(snapshot)
            assert published.execution is not None
            snapshot = published.execution
            if published.result.delivery_failed or snapshot.external_output is not None:
                return published
        if snapshot.external_output is None and (
            cursor is not None or not snapshot.is_finished
        ):
            try:
                partial = manager.partial_output(output_id, cursor or 0)
            except Exception as error:
                partial_error = str(error)
            else:
                if cursor is not None and partial is None:
                    raise ToolInputError(
                        RUNTIME.output_cursor_unsupported,
                        code="output_cursor_unsupported",
                    )

        content = ToolOutputFormatter.format_execution_output(
            output_id, snapshot, partial, partial_error=partial_error
        )
        result = PreformattedToolResult(
            content,
            attachments=snapshot.output_attachments(),
            external_output=snapshot.external_output,
            delivery_failed=snapshot.delivery_failed,
        )
        result = self._prepare_read(snapshot.output_reference(), result)
        return ToolOutputDelivery(result, snapshot)
