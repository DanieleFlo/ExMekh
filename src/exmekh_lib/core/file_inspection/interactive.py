"""Node-local behavior registry and snapshot exchange, separate from HTTP."""

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from exmekh_lib.core.models.assistant_message import ToolCall
from exmekh_lib.core.models.interactive import InteractiveProtocolError
from exmekh_lib.interfaces.message import Message

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
        self._approvals: dict[str, EditApproval] = {}
        self._preparers = {"edit_file.v1": self._prepare_edit}
        self._pending_inspections: FileInspectionRequests | None = None
        self._history_snapshot: list[dict[str, str]] | None = None

    def begin(self) -> None:
        self._evidence = self._runtime.visible_evidence(self._history())
        self._approvals.clear()
        self._pending_inspections = FileInspectionRequests(
            self._work_root, self._scope_ids()
        )
        self._history_snapshot = [message.message_dump() for message in self._history()]

    def cancellation_requested(self) -> bool:
        return self._cancelled() or (
            self._history_snapshot is not None
            and self._history_snapshot
            != [message.message_dump() for message in self._history()]
        )

    def set_cancellation_check(self, cancelled: Callable[[], bool]) -> None:
        self._cancelled = cancelled

    def prepare(
        self, event: dict[str, Any], definition: dict[str, Any]
    ) -> dict[str, object]:
        prepare = self._preparers.get(definition["behavior"])
        if prepare is None:
            raise InteractiveProtocolError(
                "Interactive behavior has no registered runtime handler"
            )
        return prepare(event, definition)

    def _prepare_edit(
        self, event: dict[str, Any], definition: dict[str, Any]
    ) -> dict[str, object]:
        resource = event["request"]["values"][definition["fields"]["resource"]]
        try:
            approval = approve_edit(self._work_root, resource, self._evidence)
        except EditPrerequisiteError as error:
            if error.code == "uninspected":
                assert self._pending_inspections is not None
                self._pending_inspections.add(resource)
            return {"status": "reject", "code": error.code, "message": str(error)}
        except (FileInspectionError, OSError, ValueError, UnicodeError) as error:
            return {
                "status": "reject",
                "code": "invalid_resource",
                "message": str(error),
            }
        self._approvals[event["call_id"]] = approval
        return approval.reply()

    def validate_completed(self, calls: list[ToolCall]) -> None:
        # Wire validation freezes the first values. The execution service checks bytes
        # again after hooks and immediately before committing the mutation.
        if self.cancellation_requested():
            raise InteractiveProtocolError(
                "Interactive call invalidated by steering before dispatch"
            )

    def execution_context(self, call_id: str) -> AbstractContextManager[None]:
        return file_edit_context(
            self._approvals.pop(call_id, None),
            lambda: self._runtime.visible_evidence(self._history()),
        )

    def pending_inspections(self) -> tuple[ToolCall, ...]:
        pending, self._pending_inspections = self._pending_inspections, None
        if pending is None or self._cancelled():
            return ()
        return pending.create_calls(self._scope_ids())
