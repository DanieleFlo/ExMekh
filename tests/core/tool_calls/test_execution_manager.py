from dataclasses import replace
from threading import Event

import pytest

from qs.core.models.tool import ExecutionOptions, ToolExecutionOperation
from qs.core.tool_calls import execution_manager
from qs.core.tool_calls.execution_manager import ToolExecutionManager


def test_background_execution_can_be_waited_without_changing_its_call_id() -> None:
    release = Event()
    manager = ToolExecutionManager()

    def work() -> str:
        release.wait()
        return "done"

    first = manager.start(
        "origin",
        "work",
        {},
        ToolExecutionOperation(work),
        ("root",),
        ExecutionOptions(mode="background"),
    )

    assert first.status == "running"
    assert manager.wait("origin", 0.01).status == "running"
    release.set()
    finished = manager.wait("origin", 1)
    assert finished.call_id == "origin"
    assert finished.status == "completed"
    assert finished.result == "done"
    assert finished.finished_at is not None


def test_completed_work_is_offered_only_to_its_original_scope() -> None:
    manager = ToolExecutionManager()
    snapshot = manager.start(
        "origin",
        "work",
        {},
        ToolExecutionOperation(lambda: "done"),
        ("root", "skill-opening"),
        ExecutionOptions(mode="wait"),
    )
    assert snapshot.status == "completed"
    assert manager.next_delivery(("root", "skill-opening")) is None


def test_reused_active_call_id_is_rejected() -> None:
    release = Event()
    manager = ToolExecutionManager()
    manager.start(
        "same",
        "work",
        {},
        ToolExecutionOperation(lambda: release.wait()),
        ("root",),
        ExecutionOptions(mode="background"),
    )
    try:
        with pytest.raises(ValueError, match="already has active work"):
            manager.start(
                "same",
                "work",
                {},
                ToolExecutionOperation(lambda: None),
                ("root",),
                ExecutionOptions(mode="background"),
            )
    finally:
        release.set()


def test_failures_and_unsupported_cancellation_have_explicit_states() -> None:
    manager = ToolExecutionManager()

    def fail() -> None:
        raise RuntimeError("failure")

    snapshot = manager.start(
        "failed",
        "work",
        {},
        ToolExecutionOperation(fail),
        ("root",),
        ExecutionOptions(mode="wait"),
    )
    assert snapshot.status == "failed"
    assert snapshot.error == "RuntimeError: failure"
    supported, cancelled, unchanged = manager.cancel("failed")
    assert supported is False
    assert cancelled is False
    assert unchanged.status == "failed"


def test_wait_timeout_keeps_work_alive_and_queues_one_delivery() -> None:
    release = Event()
    manager = ToolExecutionManager()
    snapshot = manager.start(
        "origin",
        "work",
        {},
        ToolExecutionOperation(lambda: release.wait() or "done"),
        ("session", "skill-1"),
        ExecutionOptions(wait_seconds=0.01),
        output_id="output",
    )
    assert snapshot.status == "running"
    try:
        assert manager.next_delivery(("session", "skill-1")) is None
    finally:
        release.set()
    manager.wait("origin", 1)
    assert manager.next_delivery(("other-session", "skill-1")) is None
    assert manager.next_delivery(("session", "skill-2")) is None
    assert manager.next_delivery(("session", "skill-1")).output_id == "output"
    manager.mark_delivered("output")
    assert manager.next_delivery(("session", "skill-1")) is None


def test_supported_cancellation_does_not_publish_late_worker_output() -> None:
    release = Event()
    manager = ToolExecutionManager()
    manager.start(
        "origin",
        "work",
        {},
        ToolExecutionOperation(release.wait, cancel=lambda: True),
        ("root",),
        ExecutionOptions(mode="background"),
    )
    try:
        supported, cancelled, snapshot = manager.cancel("origin")
        assert supported and cancelled
        assert snapshot.status == "cancelled"
        assert snapshot.finished_at is not None
        assert manager.wait("origin", 0.01).status == "cancelled"
    finally:
        release.set()


@pytest.mark.parametrize("confirmed", [True, False])
def test_cancellation_outcome_when_worker_finishes_during_callback(confirmed) -> None:
    release = Event()
    manager = ToolExecutionManager()

    def cancel_and_join() -> bool:
        # Process termination often unblocks the reader before cancel() returns.
        release.set()
        assert manager.wait("origin", 1).is_finished
        return confirmed

    manager.start(
        "origin",
        "work",
        {},
        ToolExecutionOperation(
            lambda: release.wait() and "interrupted output", cancel_and_join
        ),
        ("root",),
        ExecutionOptions(mode="background"),
        output_id="output",
    )
    try:
        supported, cancelled, snapshot = manager.cancel("origin")
        assert supported and cancelled == confirmed
        assert snapshot.status == ("cancelled" if confirmed else "completed")
        assert snapshot.result == (None if confirmed else "interrupted output")
        assert snapshot.error is None
        assert manager.by_output_id("output").status == snapshot.status
    finally:
        release.set()


def test_new_active_call_can_reuse_a_completed_id_without_losing_old_output() -> None:
    release = Event()
    manager = ToolExecutionManager()
    manager.start(
        "same",
        "work",
        {},
        ToolExecutionOperation(lambda: "old"),
        ("root",),
        ExecutionOptions(),
        output_id="old-output",
    )
    manager.start(
        "same",
        "work",
        {},
        ToolExecutionOperation(release.wait),
        ("root",),
        ExecutionOptions(mode="background"),
        output_id="new-output",
    )
    try:
        assert manager.wait("same", 0.01).status == "running"
        assert manager.by_output_id("old-output").result == "old"
    finally:
        release.set()


def test_multiple_results_are_delivered_serially_in_launch_order() -> None:
    release = Event()
    manager = ToolExecutionManager()
    for call_id in ("first", "second"):
        manager.start(
            call_id,
            "work",
            {},
            ToolExecutionOperation(release.wait),
            ("root",),
            ExecutionOptions(mode="background"),
            output_id=call_id,
        )
    release.set()
    for call_id in ("first", "second"):
        manager.wait(call_id, 1)
    assert manager.next_delivery(("root",)).call_id == "first"
    manager.mark_delivered("first")
    assert manager.next_delivery(("root",)).call_id == "second"


@pytest.mark.parametrize(
    "seconds", [0, -1, 301, True, "60", float("nan"), float("inf")]
)
def test_invalid_wait_limits_are_rejected(seconds) -> None:
    with pytest.raises((TypeError, ValueError)):
        ExecutionOptions.from_arguments({"wait_seconds": seconds})


def test_unknown_execution_is_reported() -> None:
    manager = ToolExecutionManager()
    with pytest.raises(ValueError, match="Unknown tool call ID"):
        manager.wait("missing", 1)


def test_worker_system_exit_is_a_failed_execution() -> None:
    manager = ToolExecutionManager()

    def work():
        raise SystemExit("stopped")

    snapshot = manager.start(
        "call", "work", {}, ToolExecutionOperation(work), ("root",), ExecutionOptions()
    )
    assert snapshot.status == "failed"
    assert snapshot.error == "SystemExit: stopped"


def test_worker_start_failure_is_terminal(monkeypatch) -> None:
    def unavailable(self):
        raise RuntimeError("Cannot start worker")

    monkeypatch.setattr(execution_manager.Thread, "start", unavailable)
    snapshot = ToolExecutionManager().start(
        "call",
        "work",
        {},
        ToolExecutionOperation(lambda: None),
        ("root",),
        ExecutionOptions(),
    )
    assert snapshot.status == "failed"
    assert snapshot.error == "RuntimeError: Cannot start worker"


def test_active_execution_limit_rejects_extra_work(monkeypatch) -> None:
    monkeypatch.setattr(
        execution_manager,
        "RUNTIME",
        replace(execution_manager.RUNTIME, max_active_tool_executions=1),
    )
    release = Event()
    manager = ToolExecutionManager()
    manager.start(
        "first",
        "work",
        {},
        ToolExecutionOperation(release.wait),
        ("root",),
        ExecutionOptions(mode="background"),
    )
    try:
        with pytest.raises(ValueError, match="Too many active"):
            manager.start(
                "second",
                "work",
                {},
                ToolExecutionOperation(lambda: None),
                ("root",),
                ExecutionOptions(),
            )
    finally:
        release.set()


def test_partial_output_errors_do_not_change_the_execution_state() -> None:
    release = Event()
    manager = ToolExecutionManager()
    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(release.wait, read_partial=lambda cursor: ("text", -1)),
        ("root",),
        ExecutionOptions(mode="background"),
        output_id="output",
    )
    try:
        with pytest.raises(ValueError, match="non-decreasing"):
            manager.partial_output("output", 0)
        assert manager.by_output_id("output").status == "running"
    finally:
        release.set()
