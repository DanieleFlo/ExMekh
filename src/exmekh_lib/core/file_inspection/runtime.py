"""Match real inspection calls and unchanged full replies in current history."""

from collections.abc import Sequence
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.tool_calls.execution_manager import ToolExecutionManager
from exmekh_lib.interfaces.message import Message
from exmekh_lib.utils.xml import stringify_tool_result

from .evidence import InspectionEvidence, InspectionReport
from .reading import FileReadContext, file_read_context


@dataclass(frozen=True, slots=True)
class _DeliveredInspection:
    call: ToolCall
    response_call_id: str
    response_content: str
    evidence: InspectionEvidence


class FileInspectionHistory:
    """Node-local evidence does not authorize reads unless its messages are visible."""

    def __init__(self) -> None:
        self._delivered: list[_DeliveredInspection] = []

    def record(self, call: ToolCall, report: object, response: ToolMessage) -> None:
        if not isinstance(report, InspectionReport) or not report.evidence.successful:
            return
        if stringify_tool_result(report) != report.evidence.report_content:
            return
        # A summarized/filtered reply cannot stand in for the actual analysis.
        if not response.get_message().endswith(report.evidence.report_content):
            return
        self._delivered.append(
            _DeliveredInspection(
                deepcopy(call),
                response.get_tool_call_id(),
                response.get_message(),
                report.evidence,
            )
        )

    def visible(
        self, history: Sequence[Message[object]]
    ) -> tuple[InspectionEvidence, ...]:
        calls: list[ToolCall] = []
        visible: list[InspectionEvidence] = []
        for message in history:
            if isinstance(message, AssistantMessage):
                calls.extend(message.get_tool_calls() or ())
            elif isinstance(message, ToolMessage):
                for delivered in self._delivered:
                    if (
                        delivered.call in calls
                        and message.get_tool_call_id() == delivered.response_call_id
                        and message.get_message() == delivered.response_content
                    ):
                        visible.append(delivered.evidence)
        return tuple(visible)


class FileToolRuntime:
    """Own inspection provenance and read prerequisites for one node runtime."""

    def __init__(self, manager: ToolExecutionManager | None = None) -> None:
        self._history = FileInspectionHistory()
        self._manager = manager
        self._batch_evidence: tuple[InspectionEvidence, ...] | None = None

    def begin_batch(self, history: Sequence[Message[object]]) -> None:
        self._batch_evidence = self._history.visible(history)

    def visible_evidence(
        self, history: Sequence[Message[object]]
    ) -> tuple[InspectionEvidence, ...]:
        return self._history.visible(history)

    def read_context(
        self, history: Sequence[Message[object]]
    ) -> AbstractContextManager[FileReadContext]:
        evidence = (
            self._batch_evidence
            if self._batch_evidence is not None
            else self._history.visible(history)
        )
        return file_read_context(evidence)

    def record_result(
        self, call: ToolCall, result: object, response: ToolMessage
    ) -> None:
        self._history.record(call, result, response)
        if call.get_name() != RUNTIME.read_tool_name or self._manager is None:
            return
        arguments = call.get_arguments()
        if (
            arguments.get("summary_mode", "none") != "none"
            or arguments.get("cursor") is not None
        ):
            return
        output_id = arguments.get("output_id")
        if not isinstance(output_id, str):
            return
        snapshot = self._manager.by_output_id(output_id)
        if snapshot is not None and snapshot.is_finished:
            self._history.record(
                snapshot.tool_call(), snapshot.record_result(), response
            )


class FileInspectionRequests:
    """Deduplicate recovery work and keep it in the scope that requested it."""

    def __init__(self, work_root: Path, owner_scope: tuple[str, ...]) -> None:
        self._work_root = work_root
        self._owner_scope = owner_scope
        self._paths: dict[str, str] = {}

    def add(self, path: str) -> None:
        candidate = self._work_root / path
        try:
            key = str(candidate.resolve())
        except (OSError, RuntimeError, ValueError):
            # Let the actual inspection report path resolution failures.
            key = str(candidate)
        self._paths.setdefault(key, path)

    def create_calls(self, current_scope: tuple[str, ...]) -> tuple[ToolCall, ...]:
        if current_scope != self._owner_scope:
            return ()
        return tuple(
            ToolCall(uuid4().hex, FunctionCall("inspect-file", {"path": path}))
            for path in self._paths.values()
        )
