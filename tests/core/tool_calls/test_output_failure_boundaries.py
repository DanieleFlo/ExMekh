"""Invalid output metadata and publication failures remain explicit and retryable."""

import errno
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from exmekh_lib.capabilities.tools.read_tool_output.tool_output_reader import (
    ToolOutputReader,
)
from exmekh_lib.config.cache import CACHE_FORMAT
from exmekh_lib.config.tool_output import ToolOutputSettings
from exmekh_lib.core.agents.node.tool_result_recorder import ToolResultRecorder
from exmekh_lib.core.agents.node.types import ToolExecutionResult
from exmekh_lib.core.cache import output_file_store
from exmekh_lib.core.cache.output_file_store import OutputFileSession
from exmekh_lib.core.cache.output_file_workspace import OutputFileWorkspace
from exmekh_lib.core.cache.paths import cache_component
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.models.assistant_message import FunctionCall, ToolCall
from exmekh_lib.core.models.external_tool_output import (
    ExternalToolOutput,
    OutputTextFile,
)
from exmekh_lib.core.models.tool import (
    ExecutionOptions,
    PreformattedToolResult,
    ToolExecutionOperation,
)
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tool_output_presentation import ToolOutputPresentation
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.tool_calls.execution_history import ExecutionHistory
from exmekh_lib.core.tool_calls.execution_manager import (
    ExecutionLaunch,
    ToolExecutionManager,
)
from exmekh_lib.core.tool_calls.output_runtime import (
    ToolOutputDelivery,
    ToolOutputRuntime,
)
from tests.core.tool_calls.test_external_output_reads import (
    output_system as output_system,
)
from tests.core.tool_calls.test_output_preparer import preparer as preparer


@pytest.mark.parametrize(
    "field,value",
    [("path", ""), ("path", None), ("token_count", True), ("line_count", -1)],
)
def test_file_output_reference_rejects_invalid_metadata(field, value):
    fields = {"path": "output.txt", "token_count": 1, "line_count": 1, field: value}
    with pytest.raises(ValueError):
        OutputTextFile(**fields)


@pytest.mark.parametrize(
    "field,value",
    [
        ("limit", 0),
        ("limit", True),
        ("token_count", -1),
        ("line_count", False),
        ("files", ()),
        ("files", []),
        ("files", ("invalid",)),
    ],
)
def test_external_output_rejects_invalid_metadata(field, value):
    fields = {
        "token_count": 1,
        "line_count": 1,
        "limit": 1,
        "files": (OutputTextFile("output.txt", 1, 1),),
        field: value,
    }
    with pytest.raises(ValueError):
        ExternalToolOutput(**fields)


@pytest.mark.parametrize("kind", ["record", "presentation", "delivery"])
def test_output_boundary_rejects_wrong_reference_or_delivery_type(kind):
    with pytest.raises(TypeError):
        if kind == "record":
            ToolCallRecord("work", "id", {}, None, external_output={})
        elif kind == "presentation":
            PreformattedToolResult("text", external_output={})
        else:
            PreformattedToolResult("text", delivery_failed=1)
    with pytest.raises(TypeError):
        ToolOutputSettings(directory="not-a-path")


@pytest.mark.parametrize("invalid", [-1, True, 1.5])
def test_invalid_tokenizer_count_fails_without_publishing_raw_output(preparer, invalid):
    service, _, tokenizer = preparer
    tokenizer.count_tokens.side_effect = None
    tokenizer.count_tokens.return_value = invalid
    prepared = service.prepare(
        ToolOutputPresentation(
            ToolOutputReference("work", "id"), PreformattedToolResult("secret")
        )
    )
    assert prepared.failure["error_kind"] == "tool_output_export_failed"
    assert "Recover the existing output ID" in prepared.failure["hint"]
    assert "may already have occurred" in prepared.failure["hint"]
    assert "secret" not in prepared.content
    assert not service.files.root.exists()


@pytest.mark.parametrize("parts", [(), ("wrong",), ("x" * 20,), ("", "x" * 20)])
def test_invalid_splitter_result_fails_without_publishing_files(preparer, parts):
    service, _, tokenizer = preparer
    tokenizer.split_text.side_effect = None
    tokenizer.split_text.return_value = parts
    prepared = service.prepare(
        ToolOutputPresentation(
            ToolOutputReference("work", "id"), PreformattedToolResult("x" * 20)
        )
    )
    assert prepared.failure is not None
    assert prepared.external_output is None
    assert not service.files.root.exists()


def test_legacy_read_and_search_failure_keep_saved_record_retryable(
    output_system, monkeypatch
):
    cache, preparer, _, _ = output_system
    original = "saved output " * 20
    record = ToolCallRecord("work", "id", {}, original)
    cache.store(record)
    reader = ToolOutputReader(lambda: cache, output_preparer=preparer)
    runtime = ToolOutputRuntime(preparer=preparer)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_bytes", MagicMock(side_effect=OSError("disk full")))
        assert reader.read("id").delivery_failed
        prepared = runtime.prepare_records({"id": record})
        assert prepared[0].execution["response_type"] == "failed"
        assert cache.get("id").result == original
    assert reader.read("id").external_output is not None
    assert cache.get("id").result is None


def test_delivery_failure_replaces_raw_value_but_success_keeps_it():
    original = ToolExecutionResult(ToolMessage("raw", "call", "work"), raw_result="raw")
    failure = {"error_kind": "export_failed"}
    failed = original.with_delivery(
        ToolOutputDelivery(
            PreformattedToolResult("failed", delivery_failed=True), failure=failure
        ),
        "id",
    )
    assert failed.raw_result == failure and failed.is_error
    delivered = original.with_delivery(
        ToolOutputDelivery(PreformattedToolResult("formatted")), "id"
    )
    assert delivered.raw_result == "raw" and not delivered.is_error


def test_publication_requires_terminal_version_and_never_acknowledges_stale_version():
    manager = ToolExecutionManager()
    release = Event()
    launch = ExecutionLaunch(
        ToolCall("call", FunctionCall("work", {})),
        ToolExecutionOperation(lambda: release.wait(2) and "done"),
        ("owner",),
        ExecutionOptions("background"),
        "id",
    )
    running = manager.launch(launch)
    try:
        with pytest.raises(ValueError):
            ToolOutputRuntime().publish_final(running)
        with pytest.raises(ValueError):
            ToolOutputRuntime(manager).publish_final(running)
        with pytest.raises(ValueError):
            manager.publish_output(running, None)
        with pytest.raises(ValueError):
            replace(running, output_id=None).output_reference()
        manager.acknowledge_output(running)
        manager.mark_persisted_output(running)
        with pytest.raises(ValueError):
            manager.mark_persisted_output(replace(running, output_id="unknown"))
        release.set()
        final = manager.wait("call", 1)
        manager.acknowledge_output(replace(final, finished_at=None))
        manager.mark_persisted_output(replace(final, finished_at=None))
        assert manager.next_delivery(("owner",)) is not None
        manager.mark_persisted("id")
        manager.acknowledge_output(final)
        assert manager.next_delivery(("owner",)) is None
    finally:
        release.set()
        manager.wait("call", 1)


def test_output_workspace_and_leases_reject_symlinked_roots(tmp_path):
    root = tmp_path / "real"
    root.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        OutputFileSession.clean_abandoned(link)
    with pytest.raises(ValueError):
        OutputFileWorkspace(
            lambda: None, tmp_path, ToolOutputSettings(directory=Path("linked"))
        )
    target = root / "lease"
    target.write_text("0")
    lease = tmp_path / "lease-link"
    lease.symlink_to(target)
    with pytest.raises(ValueError):
        output_file_store._FileLease(lease)


def test_cleanup_only_claims_canonical_session_directories_and_write_rejects_redirects(
    tmp_path,
):
    root = tmp_path / "outputs"
    root.mkdir()
    keep = []
    for suffix in ("A" * 32, "b" * 32):
        path = root / (CACHE_FORMAT.session_prefix + suffix)
        path.mkdir()
        keep.append(path)
    OutputFileSession.clean_abandoned(root)
    assert all(path.exists() for path in keep)
    session = OutputFileSession(root)
    try:
        assert session.read_limit("owner", tmp_path / "outside", 7) == 7
        other = tmp_path / "foreign"
        other.mkdir()
        (session.directory / cache_component("owner")).symlink_to(
            other, target_is_directory=True
        )
        with pytest.raises(ValueError):
            session.write("owner", "id", ("text",))
        assert list(other.iterdir()) == []
    finally:
        session.close()


def test_unexpected_lease_errors_propagate_instead_of_claiming_abandoned_storage(
    tmp_path, monkeypatch
):
    lease = output_file_store._FileLease(tmp_path / "lease")
    try:
        monkeypatch.setattr(
            output_file_store.fcntl,
            "flock",
            MagicMock(side_effect=OSError(errno.EIO, "I/O failed")),
        )
        with pytest.raises(OSError):
            lease.acquire(blocking=False)
    finally:
        lease.close()


def test_windows_lease_adapter_preserves_blocking_policy_and_contention(
    tmp_path, monkeypatch
):
    # Exercise the platform adapter with a simulated locking API; this is not a Windows live test.
    windows = SimpleNamespace(LK_LOCK=1, LK_NBLCK=2, locking=MagicMock())
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", "win32")
        patch.setitem(sys.modules, "msvcrt", windows)
        namespace = runpy.run_path(output_file_store.__file__)
        lease = namespace["_FileLease"](tmp_path / "lease")
        try:
            assert lease.acquire(blocking=True)
            assert windows.locking.call_args.args[1] == windows.LK_LOCK
            assert lease.acquire(blocking=False)
            assert windows.locking.call_args.args[1] == windows.LK_NBLCK
            windows.locking.side_effect = OSError(errno.EACCES, "busy")
            assert not lease.acquire(blocking=False)
            with pytest.raises(OSError):
                lease.acquire(blocking=True)
        finally:
            lease.close()


def test_background_persistence_failure_saves_retryable_diagnostic(
    output_system, monkeypatch
):
    cache, preparer, manager, _ = output_system
    recorder = ToolResultRecorder(ExecutionHistory(), lambda: cache, manager, preparer)
    launch = ExecutionLaunch(
        ToolCall("call", FunctionCall("work", {})),
        ToolExecutionOperation(lambda: "private" * 20),
        ("owner",),
        output_id="id",
    )
    manager.launch(launch)
    manager.attach_output("call", "id", ToolCallRecord("work", "id", {}, "pending"))
    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_bytes", MagicMock(side_effect=OSError("disk full")))
        recorder.sync_completed()
    failed = cache.get("id")
    assert failed.execution["response_type"] == "failed"
    assert failed.result["error_kind"] == "tool_output_export_failed"
    assert manager.next_delivery(("owner",)) is not None
