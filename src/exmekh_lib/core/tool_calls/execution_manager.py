from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from threading import Condition, Event, Lock, Thread
from time import monotonic
from typing import Any, Literal

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.models.assistant_message import FunctionCall, ToolCall
from exmekh_lib.core.models.tool import (
    ExecutionOptions,
    PreformattedToolResult,
    ToolExecutionOperation,
)
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.utils.tool_errors import tool_failure_result
from exmekh_lib.utils.xml import stringify_tool_result

type ExecutionStatus = Literal["running", "completed", "failed", "cancelled"]


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    call_id: str
    name: str
    arguments: dict[str, Any]
    status: ExecutionStatus
    started_at: datetime
    finished_at: datetime | None
    output_id: str | None
    elapsed_seconds: float
    result: Any = None
    error: str | None = None
    skill_path: str = "/"

    def tool_call(self) -> ToolCall:
        """Expose the originating call without leaking the snapshot's representation."""
        return ToolCall(self.call_id, FunctionCall(self.name, deepcopy(self.arguments)))

    @property
    def is_finished(self) -> bool:
        return self.status != "running"

    @property
    def response_type(self) -> str:
        return "final" if self.status == "completed" else self.status

    def status_info(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "tool_call_id": self.call_id,
            "started_at": self.started_at.isoformat(),
            "elapsed_seconds": self.elapsed_seconds,
        }
        if self.output_id is not None:
            result["output_id"] = self.output_id
        if self.finished_at is not None:
            result["finished_at"] = self.finished_at.isoformat()
        return result

    def output_result(self) -> Any:
        if self.status == "completed":
            return self.result
        if self.status == "failed":
            failure = (
                self.result
                if isinstance(self.result, dict)
                else {
                    "ok": False,
                    "error_kind": "tool_execution_failed",
                }
            )
            return {
                **self.status_info(),
                **failure,
                "error": self._diagnostic_content(),
            }
        result = self.status_info()
        diagnostic_field = "error" if self.error is not None else "message"
        result[diagnostic_field] = self._diagnostic_content()
        return result

    def record_result(self) -> Any:
        """Return the result value to store, unwrapping preformatted content."""
        result = self.output_result()
        return result.content if isinstance(result, PreformattedToolResult) else result

    def output_attachments(self) -> UserMessage | None:
        """Expose the completed companion without requiring result introspection."""
        result = self.output_result()
        return (
            result.attachments if isinstance(result, PreformattedToolResult) else None
        )

    def output_content(self) -> str:
        """Render the current result or diagnostic without exposing its representation."""
        return stringify_tool_result(self.record_result())

    def _diagnostic_content(self) -> str:
        if self.error is not None:
            return self.error
        if self.status == "cancelled":
            return "Tool execution was cancelled."
        if self.status == "failed":
            return "Tool execution failed without a diagnostic."
        return "Tool execution is still running."


@dataclass(slots=True)
class _Execution:
    """Mutable execution state, updated under the manager's lock."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    operation: ToolExecutionOperation
    owner_scope: tuple[str, ...]
    skill_path: str = "/"
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_clock: float = field(default_factory=monotonic)
    finished_clock: float | None = None
    finished: Event = field(default_factory=Event)
    status: ExecutionStatus = "running"
    finished_at: datetime | None = None
    result: Any = None
    error: str | None = None
    output_id: str | None = None
    initial_record: ToolCallRecord | None = None
    delivered: bool = False
    persisted: bool = False

    def snapshot(self) -> ExecutionSnapshot:
        return ExecutionSnapshot(
            call_id=self.call_id,
            name=self.name,
            arguments=deepcopy(self.arguments),
            status=self.status,
            started_at=self.started_at,
            finished_at=self.finished_at,
            output_id=self.output_id,
            elapsed_seconds=(
                self.finished_clock if self.finished_clock is not None else monotonic()
            )
            - self.started_clock,
            result=self.result,
            error=self.error,
            skill_path=self.skill_path,
        )

    def complete(self, result: Any) -> None:
        if not self.finished.is_set():
            self.result = result
            self._finish("completed")

    def fail(self, error: BaseException) -> None:
        if not self.finished.is_set():
            self.result = tool_failure_result(error)
            self.error = f"{type(error).__name__}: {error}"
            self._finish("failed")

    def cancel(self) -> None:
        # A confirmed cancellation wins even if terminating the process unblocked
        # its worker before the cancellation callback returned.
        self.result = None
        self.error = None
        self.persisted = False
        self._finish("cancelled")

    def _finish(self, status: ExecutionStatus) -> None:
        self.status = status
        self.finished_at = datetime.now(timezone.utc)
        self.finished_clock = monotonic()
        self.finished.set()

    def attach_output(self, output_id: str, record: ToolCallRecord | None) -> None:
        self.output_id = output_id
        self.initial_record = record
        self.persisted = (
            record is not None
            and record.execution is not None
            and record.execution.get("status") == self.status
            and self.finished.is_set()
        )


class ToolExecutionManager:
    """Track ordinary tool work without touching conversation state in workers."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._state_changed = Condition(self._lock)
        self._waits_released = False
        self._wait_release_generation = 0
        self._executions: list[_Execution] = []

    def release_waits(self) -> None:
        """Release current and subsequent waits until steering is consumed."""
        with self._state_changed:
            self._waits_released = True
            self._wait_release_generation += 1
            self._state_changed.notify_all()

    def resume_waits(self) -> None:
        """Restore ordinary waits after pending steering has been consumed."""
        with self._state_changed:
            self._waits_released = False

    def _wait_for_execution(self, execution: _Execution, wait_seconds: float) -> None:
        with self._state_changed:
            release_generation = self._wait_release_generation
            # A resumed wait must still observe any release that woke it earlier.
            self._state_changed.wait_for(
                lambda: (
                    execution.finished.is_set()
                    or self._waits_released
                    or release_generation != self._wait_release_generation
                ),
                timeout=wait_seconds,
            )

    def start(
        self,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        operation: ToolExecutionOperation,
        owner_scope: tuple[str, ...],
        options: ExecutionOptions,
        output_id: str | None = None,
        *,
        skill_path: str = "/",
    ) -> ExecutionSnapshot:
        if not isinstance(skill_path, str) or not skill_path.startswith("/"):
            raise ValueError("skill_path must be an absolute skill path")
        execution = _Execution(
            call_id,
            name,
            deepcopy(arguments),
            operation,
            owner_scope,
            skill_path=skill_path,
        )
        execution.output_id = output_id
        with self._lock:
            if any(
                item.call_id == call_id and not item.finished.is_set()
                for item in self._executions
            ):
                raise ValueError(f"Tool call ID '{call_id}' already has active work")
            if sum(not item.finished.is_set() for item in self._executions) >= (
                RUNTIME.max_active_tool_executions
            ):
                raise ValueError("Too many active tool executions")
            self._executions.append(execution)
        try:
            Thread(target=self._run, args=(execution,), daemon=True).start()
        except RuntimeError as error:
            self._fail_execution(execution, error)
        if options.mode == "wait":
            self._wait_for_execution(execution, options.wait_seconds)
        with self._lock:
            if execution.finished.is_set():
                execution.delivered = True
            return execution.snapshot()

    def _run(self, execution: _Execution) -> None:
        try:
            result = execution.operation.run()
        except BaseException as error:
            # SystemExit in a worker must also release waits and produce a terminal state.
            self._fail_execution(execution, error)
            return
        with self._state_changed:
            execution.complete(result)
            self._state_changed.notify_all()

    def _fail_execution(self, execution: _Execution, error: BaseException) -> None:
        with self._state_changed:
            execution.fail(error)
            self._state_changed.notify_all()

    def attach_output(
        self, call_id: str, output_id: str, record: ToolCallRecord | None
    ) -> None:
        with self._lock:
            execution = next(
                item for item in reversed(self._executions) if item.call_id == call_id
            )
            execution.attach_output(output_id, record)

    def by_output_id(self, output_id: str) -> ExecutionSnapshot | None:
        with self._lock:
            for execution in self._executions:
                if execution.output_id == output_id:
                    return execution.snapshot()
        return None

    def output_snapshots(self) -> tuple[ExecutionSnapshot, ...]:
        """Inspect published outputs without acknowledging their delivery."""
        with self._lock:
            return tuple(
                execution.snapshot()
                for execution in self._executions
                if execution.output_id is not None
            )

    def running_outputs(
        self, owner_scope: tuple[str, ...]
    ) -> tuple[ExecutionSnapshot, ...]:
        """Inspect active outputs owned by this invocation or its descendants."""
        with self._lock:
            return tuple(
                execution.snapshot()
                for execution in self._executions
                if execution.owner_scope[: len(owner_scope)] == owner_scope
                and not execution.finished.is_set()
                and execution.output_id is not None
            )

    def output_records(
        self, stored_records: Mapping[str, ToolCallRecord]
    ) -> tuple[ToolCallRecord, ...]:
        """Describe published outputs and progress without consuming delivery."""
        return tuple(
            self._output_record(snapshot, stored_records)
            for snapshot in self.output_snapshots()
        )

    def _output_record(
        self,
        snapshot: ExecutionSnapshot,
        stored_records: Mapping[str, ToolCallRecord],
    ) -> ToolCallRecord:
        if snapshot.output_id is None:
            raise ValueError("A published execution must have an output ID")
        stored = stored_records.get(snapshot.output_id)
        result = snapshot.record_result()
        execution = {
            **snapshot.status_info(),
            "response_type": snapshot.response_type,
            "storage": "saved" if stored is not None else "runtime",
        }
        if not snapshot.is_finished:
            try:
                partial = self.partial_output(snapshot.output_id, 0)
                if partial is not None and partial[0]:
                    result = {**result, "partial_output": partial[0]}
                    execution["response_type"] = "partial"
            except Exception as error:
                result = {**result, "partial_output_error": str(error)}
        if stored is not None:
            return replace(stored, result=result, execution=execution)
        return ToolCallRecord(
            name=snapshot.name,
            id=snapshot.output_id,
            args=snapshot.arguments,
            result=result,
            skill_path=snapshot.skill_path,
            tool_call_id=snapshot.call_id,
            execution=execution,
        )

    def partial_output(self, output_id: str, cursor: int) -> tuple[str, int] | None:
        with self._lock:
            execution = next(
                (item for item in self._executions if item.output_id == output_id),
                None,
            )
        if execution is None or execution.operation.read_partial is None:
            return None
        content, next_cursor = execution.operation.read_partial(cursor)
        if (
            not isinstance(content, str)
            or type(next_cursor) is not int
            or next_cursor < cursor
        ):
            raise ValueError(
                "Partial output must contain text and a non-decreasing integer cursor"
            )
        return content, next_cursor

    def wait(self, call_id: str, wait_seconds: float) -> ExecutionSnapshot:
        wait_seconds = ExecutionOptions.validate_wait_seconds(wait_seconds)
        execution = self._by_call_id(call_id)
        self._wait_for_execution(execution, wait_seconds)
        with self._lock:
            return execution.snapshot()

    def cancel(self, call_id: str) -> tuple[bool, bool, ExecutionSnapshot]:
        execution = self._by_call_id(call_id)
        callback = execution.operation.cancel
        with self._lock:
            if callback is None:
                return False, False, execution.snapshot()
            if execution.finished.is_set():
                return True, False, execution.snapshot()
        try:
            cancelled = callback()
            if not isinstance(cancelled, bool):
                raise TypeError("Cancellation callback must return a bool")
        except Exception as error:
            raise ValueError(f"Cancellation failed: {error}") from error
        with self._state_changed:
            if cancelled:
                execution.cancel()
                self._state_changed.notify_all()
            return True, cancelled, execution.snapshot()

    def _by_call_id(self, call_id: str) -> _Execution:
        with self._lock:
            matches = [item for item in self._executions if item.call_id == call_id]
        if not matches:
            raise ValueError(f"Unknown tool call ID '{call_id}'")
        active = [item for item in matches if not item.finished.is_set()]
        if len(active) == 1:
            return active[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous tool call ID '{call_id}'")
        return matches[0]

    def completed_to_persist(
        self,
    ) -> tuple[tuple[ExecutionSnapshot, ToolCallRecord], ...]:
        with self._lock:
            return tuple(
                (item.snapshot(), item.initial_record)
                for item in self._executions
                if item.finished.is_set()
                and item.initial_record is not None
                and not item.persisted
            )

    def mark_persisted(self, output_id: str) -> None:
        with self._lock:
            for execution in self._executions:
                if execution.output_id == output_id:
                    execution.persisted = True
                    return

    def next_delivery(self, owner_scope: tuple[str, ...]) -> ExecutionSnapshot | None:
        with self._lock:
            for execution in self._executions:
                if (
                    execution.owner_scope == owner_scope
                    and execution.finished.is_set()
                    and not execution.delivered
                    and execution.output_id is not None
                ):
                    return execution.snapshot()
        return None

    def mark_delivered(self, output_id: str) -> None:
        with self._lock:
            for execution in self._executions:
                if execution.output_id == output_id:
                    execution.delivered = True
                    return
