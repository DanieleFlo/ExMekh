"""Node-local behavior registry and snapshot exchange, separate from HTTP."""

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from exmekh_lib.config.file_editing import FILE_EDIT_FORMAT, FILE_EDITING
from exmekh_lib.core.models.assistant_message import ToolCall
from exmekh_lib.core.models.interactive import InteractiveProtocolError
from exmekh_lib.interfaces.message import Message

from .edit_filesystem import resolve_edit_target
from .editing import (
    EditApproval,
    EditPrerequisiteError,
    approve_edit,
    file_edit_context,
)
from .errors import FileInspectionError
from .evidence import InspectionEvidence
from .runtime import FileInspectionRequests, FileToolRuntime


class FileInteractiveCalls:
    def __init__(
        self,
        runtime: FileToolRuntime,
        work_root: Path,
        history: Callable[[], Sequence[Message[object]]],
        cancelled: Callable[[], bool],
        scope_ids: Callable[[], tuple[str, ...]],
    ) -> None:
        self._runtime = runtime
        self._work_root = work_root
        self._history = history
        self._cancelled = cancelled
        self._scope_ids = scope_ids
        self._evidence: tuple[InspectionEvidence, ...] = ()
        self._approvals: dict[str, list[EditApproval]] = {}
        self._pending_inspections: FileInspectionRequests | None = None
        self._history_snapshot: list[dict[str, str]] | None = None
        self._scope_snapshot: tuple[str, ...] = ()

    def begin(self) -> None:
        self._evidence = self._runtime.visible_evidence(self._history())
        self._approvals.clear()
        self._pending_inspections = FileInspectionRequests(
            self._work_root, self._scope_ids()
        )
        self._history_snapshot = [message.message_dump() for message in self._history()]
        self._scope_snapshot = self._scope_ids()

    def cancellation_requested(self) -> bool:
        return self._cancelled() or (
            self._history_snapshot is not None
            and (
                self._scope_ids() != self._scope_snapshot
                or self._history_snapshot
                != [message.message_dump() for message in self._history()]
            )
        )

    def set_cancellation_check(self, cancelled: Callable[[], bool]) -> None:
        self._cancelled = cancelled

    def prepare(
        self, event: dict[str, Any], definition: dict[str, Any]
    ) -> dict[str, object]:
        if definition["behavior"] != FILE_EDIT_FORMAT.behavior:
            raise InteractiveProtocolError(FILE_EDITING.runtime_handler_required)
        return self._prepare_edit(event, definition)

    def _prepare_edit(
        self, event: dict[str, Any], definition: dict[str, Any]
    ) -> dict[str, object]:
        resource = event["request"]["values"]["path"]
        call_id = event["call_id"]
        approvals = self._approvals.setdefault(call_id, [])
        try:
            if event["request"]["edit_index"] != len(approvals) + 1:
                raise InteractiveProtocolError(FILE_EDITING.authorization_sequence)
            target = resolve_edit_target(self._work_root, resource)
            previous = next((item for item in approvals if item.target == target), None)
            if previous is None:
                approval = approve_edit(self._work_root, resource, self._evidence)
            else:
                approval = previous.authorize_alias(self._work_root, resource)
        except EditPrerequisiteError as error:
            self._approvals.pop(call_id, None)
            if error.code == FILE_EDIT_FORMAT.uninspected_error_code:
                assert self._pending_inspections is not None
                self._pending_inspections.add(resource)
            return {"status": "reject", "code": error.code, "message": str(error)}
        except (FileInspectionError, OSError, ValueError, UnicodeError) as error:
            self._approvals.pop(call_id, None)
            return {
                "status": "reject",
                "code": "invalid_resource",
                "message": str(error),
            }
        approvals.append(approval)
        return approval.reply()

    def validate_completed(self, calls: list[ToolCall]) -> None:
        # Wire validation freezes each path/index. The execution service checks bytes
        # again after hooks and immediately before committing the mutation.
        if self.cancellation_requested():
            raise InteractiveProtocolError(FILE_EDITING.cancelled_before_dispatch)

    def execution_context(self, call_id: str) -> AbstractContextManager[None]:
        return file_edit_context(
            tuple(self._approvals.pop(call_id, [])),
            self._execution_evidence,
        )

    def _execution_evidence(self) -> tuple[InspectionEvidence, ...]:
        if self._cancelled() or self._scope_ids() != self._scope_snapshot:
            raise FileInspectionError(FILE_EDITING.cancelled_before_write)
        return self._runtime.visible_evidence(self._history())

    def pending_inspections(self) -> tuple[ToolCall, ...]:
        pending, self._pending_inspections = self._pending_inspections, None
        if pending is None or self._cancelled():
            return ()
        return pending.create_calls(self._scope_ids())
