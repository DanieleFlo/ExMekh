from concurrent.futures import Future
from dataclasses import replace
from threading import Condition, Event, Thread

import pytest

from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.models.tool import ExecutionOptions, ToolExecutionOperation
from qs.core.tool_calls import execution_manager
from qs.core.tool_calls.execution_manager import ToolExecutionManager


def test_resuming_waits_does_not_lose_an_already_released_wait(monkeypatch):
    waiting = Event()
    notified = Event()
    inspect_notification = Event()
    finish_work = Event()

    class PausedCondition(Condition):
        def wait(self, timeout=None):
            waiting.set()
            result = super().wait(timeout)
            # Let the consuming run resume waits before this waiter checks its predicate.
            self.release()
            try:
                notified.set()
                assert inspect_notification.wait(3)
            finally:
                self.acquire()
            return result

    monkeypatch.setattr(execution_manager, "Condition", PausedCondition)
    manager = ToolExecutionManager()
    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(finish_work.wait),
        ("root",),
        ExecutionOptions(mode="background"),
    )
    result = Future()

    def wait():
        try:
            result.set_result(manager.wait("call", 10))
        except BaseException as error:
            result.set_exception(error)

    waiter = Thread(target=wait, daemon=True)
    waiter.start()
    try:
        assert waiting.wait(3)
        manager.release_waits()
        assert notified.wait(3)
        manager.resume_waits()
        inspect_notification.set()
        assert result.result(1).status == "running"
    finally:
        inspect_notification.set()
        finish_work.set()
        waiter.join(3)
        assert not waiter.is_alive()


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
    ("scope", "expected"),
    [
        (("session", "outer:1"), ["parent", "child", "deep-child"]),
        (("session", "outer:1", "child:2"), ["child", "deep-child"]),
        (("session", "outer:1", "child:3"), []),
        (("other-session", "outer:1"), ["other-session"]),
    ],
)
def test_running_outputs_follow_invocation_scopes_at_every_depth(scope, expected):
    release = Event()
    manager = ToolExecutionManager()
    owners = {
        "root": ("session",),
        "parent": ("session", "outer:1"),
        "child": ("session", "outer:1", "child:2"),
        "deep-child": ("session", "outer:1", "child:2", "deeper:3", "leaf:4"),
        "sibling": ("session", "outer:10"),
        "other-session": ("other-session", "outer:1"),
        "unpublished": ("session", "outer:1"),
    }
    for call_id, owner in owners.items():
        manager.start(
            call_id,
            "work",
            {},
            ToolExecutionOperation(release.wait),
            owner,
            ExecutionOptions(mode="background"),
            output_id=None if call_id == "unpublished" else call_id,
        )
    try:
        assert [item.call_id for item in manager.running_outputs(scope)] == expected
        assert [item.call_id for item in manager.running_outputs(scope)] == expected
    finally:
        release.set()
        for call_id in owners:
            assert manager.wait(call_id, 1).is_finished
    assert manager.running_outputs(scope) == ()
    assert manager.next_delivery(owners["child"]).call_id == "child"


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
@pytest.mark.parametrize("delivered", [True, False])
def test_running_outputs_exclude_terminal_work_regardless_of_delivery(
    status, delivered
):
    release = Event()
    manager = ToolExecutionManager()

    def work():
        release.wait()
        if status == "failed":
            raise RuntimeError("work failed")
        return "done"

    manager.start(
        "work",
        "work",
        {},
        ToolExecutionOperation(work, cancel=lambda: True),
        ("session", "skill"),
        ExecutionOptions(mode="background"),
        output_id="output",
    )
    try:
        assert [item.output_id for item in manager.running_outputs(("session",))] == [
            "output"
        ]
        if status == "cancelled":
            assert manager.cancel("work")[1]
        release.set()
        assert manager.wait("work", 1).status == status
        if delivered:
            manager.mark_delivered("output")
        assert manager.running_outputs(("session",)) == ()
        assert (manager.next_delivery(("session", "skill")) is None) == delivered
    finally:
        release.set()
        manager.wait("work", 1)


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


@pytest.mark.parametrize("skill_path", [None, 42, "relative/path"])
def test_invalid_skill_paths_are_rejected_before_starting_work(skill_path):
    manager = ToolExecutionManager()

    def work():
        pytest.fail("Invalid scope must be rejected before starting a worker")

    with pytest.raises(ValueError, match="skill_path must be an absolute skill path"):
        manager.start(
            "call",
            "work",
            {},
            ToolExecutionOperation(work),
            ("root",),
            ExecutionOptions(),
            skill_path=skill_path,
        )

    assert manager.output_snapshots() == ()


def test_unpublished_execution_status_omits_the_output_identifier():
    manager = ToolExecutionManager()
    snapshot = manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(lambda: "done"),
        ("root",),
        ExecutionOptions(),
    )

    status = snapshot.status_info()

    assert status["status"] == "completed"
    assert status["tool_call_id"] == "call"
    assert "finished_at" in status
    assert "output_id" not in status


def test_output_records_reject_a_snapshot_without_a_published_identifier(monkeypatch):
    manager = ToolExecutionManager()
    snapshot = manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(lambda: "done"),
        ("root",),
        ExecutionOptions(),
    )
    monkeypatch.setattr(manager, "output_snapshots", lambda: (snapshot,))

    with pytest.raises(ValueError, match="published execution must have an output ID"):
        manager.output_records({})


def test_finished_work_does_not_invoke_its_cancellation_callback():
    def cancel():
        pytest.fail("Finished work must not be cancelled")

    manager = ToolExecutionManager()
    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(lambda: "done", cancel),
        ("root",),
        ExecutionOptions(),
        output_id="output",
    )

    supported, cancelled, snapshot = manager.cancel("call")

    assert supported is True and cancelled is False
    assert snapshot.status == "completed" and snapshot.result == "done"


@pytest.mark.parametrize(
    "failure", [RuntimeError("cannot stop worker"), "not a boolean"]
)
def test_cancellation_failures_keep_the_worker_running(failure):
    release = Event()

    def cancel():
        if isinstance(failure, Exception):
            raise failure
        return failure

    manager = ToolExecutionManager()
    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(release.wait, cancel),
        ("root",),
        ExecutionOptions(mode="background"),
        output_id="output",
    )
    try:
        with pytest.raises(ValueError, match="Cancellation failed") as caught:
            manager.cancel("call")
        expected_cause = RuntimeError if isinstance(failure, Exception) else TypeError
        assert isinstance(caught.value.__cause__, expected_cause)
        assert manager.by_output_id("output").status == "running"
    finally:
        release.set()
        assert manager.wait("call", 1).is_finished


def test_late_worker_failure_does_not_replace_confirmed_cancellation(monkeypatch):
    release = Event()
    workers = []

    def create_thread(**arguments):
        worker = Thread(**arguments)
        workers.append(worker)
        return worker

    def work():
        release.wait()
        raise RuntimeError("process terminated after cancellation")

    monkeypatch.setattr(execution_manager, "Thread", create_thread)
    manager = ToolExecutionManager()
    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(work, cancel=lambda: True),
        ("root",),
        ExecutionOptions(mode="background"),
        output_id="output",
    )
    try:
        assert manager.cancel("call")[1] is True
    finally:
        release.set()
        workers[0].join(timeout=1)

    assert not workers[0].is_alive()
    snapshot = manager.by_output_id("output")
    assert snapshot.status == "cancelled"
    assert snapshot.result is None and snapshot.error is None


@pytest.mark.parametrize("method", ["wait", "cancel"])
def test_reused_completed_call_ids_require_an_unambiguous_output_identifier(method):
    manager = ToolExecutionManager()
    for output_id in ("first", "second"):
        manager.start(
            "same",
            "work",
            {},
            ToolExecutionOperation(lambda: "done"),
            ("root",),
            ExecutionOptions(),
            output_id=output_id,
        )

    with pytest.raises(ValueError, match="Ambiguous tool call ID"):
        if method == "wait":
            manager.wait("same", 1)
        else:
            manager.cancel("same")

    assert manager.by_output_id("first").result == "done"
    assert manager.by_output_id("second").result == "done"


def test_marking_unknown_outputs_does_not_acknowledge_known_work():
    manager = ToolExecutionManager()
    manager.mark_persisted("missing")
    manager.mark_delivered("missing")
    release = Event()
    record = ToolCallRecord(
        "work", "output", {}, "pending", execution={"status": "running"}
    )
    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(release.wait),
        ("root",),
        ExecutionOptions(mode="background"),
        output_id="output",
    )
    try:
        manager.attach_output("call", "output", record)
    finally:
        release.set()
        assert manager.wait("call", 1).is_finished

    manager.mark_persisted("missing")
    manager.mark_delivered("missing")

    assert manager.next_delivery(("root",)).output_id == "output"
    assert [stored for _, stored in manager.completed_to_persist()] == [record]
