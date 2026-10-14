from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from threading import Condition, Event, Lock, Thread
from time import monotonic
from typing import Any, Literal

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.config.runtime_text import RUNTIME_TEXT
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.models.assistant_message import FunctionCall, ToolCall
from exmekh_lib.core.models.external_tool_output import ExternalToolOutput
from exmekh_lib.core.models.tool import (
    ExecutionOptions,
    PreformattedToolResult,
    ToolExecutionOperation,
)
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.models.tool_result import ToolResult
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.utils.tool_errors import ToolInputError, tool_failure_result
from exmekh_lib.utils.xml import stringify_tool_result

type ExecutionStatus = Literal["running", "completed", "failed", "cancelled"]


@dataclass(frozen=True, slots=True)
class ExecutionLaunch:
    """Originating call, execution policy and ownership of one unit of work."""

    call: ToolCall
    operation: ToolExecutionOperation
    owner_scope: tuple[str, ...]
    options: ExecutionOptions = field(default_factory=ExecutionOptions)
    output_id: str | None = None
    skill_path: str = "/"

    def __post_init__(self) -> None:
        if not isinstance(self.skill_path, str) or not self.skill_path.startswith("/"):
            raise ValueError(RUNTIME_TEXT.skill_path_must_be_an_absolute_skill_path)


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
    executed_arguments: dict[str, Any] | None = None
    external_output: ExternalToolOutput | None = None
    external_attachments: UserMessage | None = None
    delivery_failed: bool = False

    def output_reference(self) -> ToolOutputReference:
        if self.output_id is None:
            raise ValueError(RUNTIME_TEXT.a_published_execution_must_have_an_output_id)
        return ToolOutputReference(self.name, self.output_id)

    def same_version(self, other: ExecutionSnapshot) -> bool:
        return (
            self.call_id,
            self.output_id,
            self.status,
            self.started_at,
            self.finished_at,
        ) == (
            other.call_id,
            other.output_id,
            other.status,
            other.started_at,
            other.finished_at,
        )

    def output_arguments(self) -> dict[str, Any]:
        """Expose work inputs while tool_call() retains the provider's call."""
        return deepcopy(
            self.executed_arguments
            if self.executed_arguments is not None
            else self.arguments
        )

    def tool_call(self) -> ToolCall:
        """Expose the originating call without leaking the snapshot's representation."""
        return ToolCall(self.call_id, FunctionCall(self.name, deepcopy(self.arguments)))

    @property
    def is_finished(self) -> bool:
        return self.status != "running"

    @property
    def response_type(self) -> str:
        if self.delivery_failed:
            return "failed"
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
        if self.external_output is not None:
            return PreformattedToolResult(
                self.external_output.render(),
                attachments=self.external_attachments,
                external_output=self.external_output,
                delivery_failed=self.delivery_failed,
            )
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
        return (
            None
            if self.external_output is not None
            else ToolResult(self.output_result()).storage_value()
        )

    def output_attachments(self) -> UserMessage | None:
        """Expose the completed companion without requiring result introspection."""
        return ToolResult(self.output_result()).attachments()

    def output_content(self) -> str:
        """Render the current result or diagnostic without exposing its representation."""
        return stringify_tool_result(ToolResult(self.output_result()).value())

    def _diagnostic_content(self) -> str:
        if self.error is not None:
            return self.error
        if self.status == "cancelled":
            return RUNTIME_TEXT.tool_execution_was_cancelled
        if self.status == "failed":
            return RUNTIME_TEXT.tool_execution_failed_without_a_diagnostic
        return RUNTIME_TEXT.tool_execution_is_still_running


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
    external_output: ExternalToolOutput | None = None
    external_attachments: UserMessage | None = None
    delivery_failed: bool = False

    @classmethod
    def from_launch(cls, launch: ExecutionLaunch) -> _Execution:
        return cls(
            call_id=launch.call.get_id(),
            name=launch.call.get_name(),
            arguments=deepcopy(launch.call.get_arguments()),
            operation=launch.operation,
            owner_scope=launch.owner_scope,
            skill_path=launch.skill_path,
            output_id=launch.output_id,
        )

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
            executed_arguments=self.operation.get_executed_arguments(),
            external_output=self.external_output,
            external_attachments=self.external_attachments,
            delivery_failed=self.delivery_failed,
        )

    def complete(self, result: Any) -> None:
        if not self.finished.is_set():
            output = ToolResult(result)
            self.delivery_failed = output.delivery_failed()
            self.external_output = output.external_output()
            self.external_attachments = (
                output.attachments() if self.external_output is not None else None
            )
            self.result = None if self.external_output is not None else result
            self._finish("completed")

    def fail(self, error: BaseException) -> None:
        if not self.finished.is_set():
            self.result = tool_failure_result(error)
            self.error = RUNTIME.failure_message.format(
                error_type=type(error).__name__, error=error
            )
            self._finish("failed")

    def cancel(self) -> None:
        # A confirmed cancellation wins even if terminating the process unblocked
        # its worker before the cancellation callback returned.
        self.result = None
        self.external_output = None
        self.external_attachments = None
        self.delivery_failed = False
        self.error = None
        self.persisted = False
        self.delivered = False
        self._finish("cancelled")

    def publish_output(
        self, expected: ExecutionSnapshot, reference: ExternalToolOutput | None
    ) -> ExecutionSnapshot:
        current = self.snapshot()
        if not current.same_version(expected) or reference is None:
            return current
        if self.external_output != reference:
            self.external_attachments = current.output_attachments()
            self.external_output = reference
            self.result = None
            self.persisted = False
        return self.snapshot()

    def acknowledge_output(self, expected: ExecutionSnapshot) -> None:
        if expected.is_finished and self.snapshot().same_version(expected):
            self.delivered = True

    def mark_persisted_output(self, expected: ExecutionSnapshot) -> None:
        if expected.is_finished and self.snapshot().same_version(expected):
            self.persisted = True

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
        """Compatibility entry point; new callers pass an ExecutionLaunch to launch()."""
        return self.launch(
            ExecutionLaunch(
                call=ToolCall(call_id, FunctionCall(name, arguments)),
                operation=operation,
                owner_scope=owner_scope,
                options=options,
                output_id=output_id,
                skill_path=skill_path,
            )
        )

    def launch(self, request: ExecutionLaunch) -> ExecutionSnapshot:
        execution = _Execution.from_launch(request)
        with self._lock:
            if any(
                item.call_id == execution.call_id and not item.finished.is_set()
                for item in self._executions
            ):
                raise ToolInputError(
                    RUNTIME_TEXT.tool_call_id_already_has_active_work.format(
                        call_id=execution.call_id
                    ),
                    code="execution_already_active",
                )
            if sum(not item.finished.is_set() for item in self._executions) >= (
                RUNTIME.max_active_tool_executions
            ):
                raise ToolInputError(
                    RUNTIME_TEXT.too_many_active_tool_executions,
                    code="execution_limit_reached",
                )
            self._executions.append(execution)
        try:
            Thread(target=self._run, args=(execution,), daemon=True).start()
        except RuntimeError as error:
            self._fail_execution(execution, error)
        if request.options.mode == "wait":
            self._wait_for_execution(execution, request.options.wait_seconds)
        with self._lock:
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

    def publish_output(
        self, expected: ExecutionSnapshot, reference: ExternalToolOutput | None
    ) -> ExecutionSnapshot:
        """Check publication against a snapshot, including cancellation confirmed meanwhile."""
        output_id = expected.output_reference().id
        if not expected.is_finished:
            raise ValueError(RUNTIME_TEXT.final_output_required)
        with self._lock:
            execution = self._by_output_id(output_id)
            return execution.publish_output(expected, reference)

    def acknowledge_output(self, expected: ExecutionSnapshot) -> None:
        """Only acknowledge the final version actually returned by a successful read."""
        if not expected.is_finished:
            return
        output_id = expected.output_reference().id
        with self._lock:
            execution = self._by_output_id(output_id)
            execution.acknowledge_output(expected)

    def _by_output_id(self, output_id: str) -> _Execution:
        for execution in self._executions:
            if execution.output_id == output_id:
                return execution
        raise ValueError(RUNTIME_TEXT.unknown_output_id.format(output_id=output_id))

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
            raise ValueError(RUNTIME_TEXT.a_published_execution_must_have_an_output_id)
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
        reference = snapshot.external_output
        if stored is not None:
            return replace(
                stored,
                args=snapshot.output_arguments(),
                result=result,
                execution=execution,
                external_output=reference,
            )
        return ToolCallRecord(
            name=snapshot.name,
            id=snapshot.output_id,
            args=snapshot.output_arguments(),
            result=result,
            skill_path=snapshot.skill_path,
            tool_call_id=snapshot.call_id,
            execution=execution,
            external_output=reference,
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
                RUNTIME_TEXT.partial_output_must_contain_text_and_a_non_decreasing
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
                raise TypeError(RUNTIME_TEXT.cancellation_callback_must_return_a_bool)
        except Exception as error:
            raise ToolInputError(
                RUNTIME_TEXT.cancellation_failed.format(error=error),
                code="cancellation_failed",
            ) from error
        with self._state_changed:
            if cancelled:
                execution.cancel()
                self._state_changed.notify_all()
            return True, cancelled, execution.snapshot()

    def _by_call_id(self, call_id: str) -> _Execution:
        with self._lock:
            matches = [item for item in self._executions if item.call_id == call_id]
        if not matches:
            raise ToolInputError(
                RUNTIME_TEXT.unknown_tool_call_id.format(call_id=call_id),
                code="unknown_tool_call_id",
            )
        active = [item for item in matches if not item.finished.is_set()]
        if len(active) == 1:
            return active[0]
        if len(matches) > 1:
            raise ToolInputError(
                RUNTIME_TEXT.ambiguous_tool_call_id.format(call_id=call_id),
                code="ambiguous_tool_call_id",
            )
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
                    execution.mark_persisted_output(execution.snapshot())
                    return

    def mark_persisted_output(self, expected: ExecutionSnapshot) -> None:
        with self._lock:
            execution = self._by_output_id(expected.output_reference().id)
            execution.mark_persisted_output(expected)

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
                    execution.acknowledge_output(execution.snapshot())
                    return
