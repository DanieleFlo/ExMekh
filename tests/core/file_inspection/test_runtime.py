"""Regression cases at file tool/runtime boundaries."""

from functools import partial
from xml.etree import ElementTree as ET

import pytest

from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.core.file_inspection.edit_filesystem import (
    delete_file,
    new_directory,
    new_file,
)
from exmekh_lib.core.file_inspection.editing import approve_edit
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.file_inspection.reading import file_read_context, read_file
from exmekh_lib.core.file_inspection.runtime import FileInspectionRequests
from exmekh_lib.core.file_inspection.service import inspect_file
from exmekh_lib.core.file_inspection.workspace import read_snapshot
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tool_hooks import ToolHooks
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.interfaces.tool_hook import AfterTool
from tests.core.file_inspection.test_reading import (
    assert_paired,
    batch,
    call,
    make_node,
    open_files,
    read_call,
    replies,
    result_for,
)
from tests.tool_result_support import parse_tool_result


def test_read_recovery_does_not_escape_a_skill_closed_in_the_same_batch(tmp_path):
    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(read_call(), call("exit_agent", "exit", text="finished", items=[])),
            AssistantMessage("done"),
        ],
    )
    result = node.run("read then exit")
    assert not any(
        m.get_name() == "inspect-file" for m in replies(provider.requests[-1])
    )
    assert (
        "inspect-file"
        not in provider.requests[-1].get_extra()["agentic"]["allowed_tools"]
    )
    transcript = result.skill_executions[0].history
    failed_read = next(
        m
        for m in transcript
        if isinstance(m, ToolMessage) and m.get_tool_call_id() == "read"
    )
    assert (
        parse_tool_result(failed_read.get_message())["error_kind"]
        == "file_inspection_required"
    )
    assert_paired(provider.requests[-1].get_messages())


def test_inspection_requests_are_deduplicated_and_tied_to_the_exact_scope(tmp_path):
    requests = FileInspectionRequests(tmp_path, ("node", "first-opening"))
    requests.add("notes.txt")
    requests.add("./notes.txt")
    requests.add("bad\x00path")
    assert requests.create_calls(("other-node", "first-opening")) == ()
    assert requests.create_calls(("node", "reopened-skill")) == ()
    calls = requests.create_calls(("node", "first-opening"))
    assert [c.get_arguments()["path"] for c in calls] == ["notes.txt", "bad\x00path"]
    assert calls[0].get_id() != calls[1].get_id()


def test_read_override_preserves_synchronous_prerequisite_recovery(tmp_path):
    (tmp_path / "notes.txt").write_text("content")

    def replacement(path: str, start_index: int, end_index: int):
        return read_file(tmp_path, path, start_index, end_index)

    override = Tool(replacement, name="read-file")
    assert override.background_enabled
    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(read_call()),
            batch(read_call("retry")),
            AssistantMessage("done"),
        ],
        overrides={"read-file": override},
    )
    node.run("read")
    assert replies(provider.requests[2])[-1].get_name() == "inspect-file"
    assert result_for(provider.requests[-1], "retry")["ok"]
    assert override.background_enabled  # Composition must copy the caller's tool.


def test_read_after_hook_failure_keeps_the_recovery_request(tmp_path):
    (tmp_path / "notes.txt").write_text("content")

    class FailingAfter(AfterTool):
        def handle(self, context, result):
            raise RuntimeError("filter failed")

    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(read_call()),
            AssistantMessage("done"),
        ],
        hooks={"read-file": ToolHooks(after=FailingAfter())},
    )
    node.run("read")
    outputs = replies(provider.requests[-1])[-2:]
    assert [m.get_name() for m in outputs] == ["read-file", "inspect-file"]
    header = ET.fromstring(outputs[0].get_message().split("</header>")[0] + "</header>")
    assert header.findtext("response-type") == "failed"
    assert (
        parse_tool_result(outputs[0].get_message())["error_kind"]
        == "tool_execution_failed"
    )
    assert parse_tool_result(outputs[1].get_message())["ok"]
    assert_paired(provider.requests[-1].get_messages())


def test_missing_workroot_is_a_domain_error_and_invalidates_read_evidence(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "notes.txt").write_text("content")
    report = inspect_file(root, "notes.txt")
    root.rename(tmp_path / "moved-workspace")
    with pytest.raises(FileInspectionError, match="workroot.*restore"):
        read_snapshot(root, "notes.txt", FILE_INSPECTION)
    with file_read_context((report.evidence,)):
        assert (
            read_file(root, "notes.txt", 1, 1)["error_kind"]
            == "file_inspection_required"
        )


def test_workroot_must_be_a_directory(tmp_path):
    root = tmp_path / "root.txt"
    root.write_text("content")
    with pytest.raises(FileInspectionError, match="workroot.*directory"):
        read_snapshot(root, str(root), FILE_INSPECTION)


@pytest.mark.parametrize(
    "path,diagnostic", [("bad\x00path", "NUL"), ("bad\ud800path", "Unicode")]
)
def test_invalid_paths_are_rejected_before_automatic_inspection(
    tmp_path, path, diagnostic
):
    with pytest.raises(FileInspectionError, match=diagnostic):
        inspect_file(tmp_path, path)
    node, provider = make_node(
        tmp_path,
        [
            open_files(),
            batch(read_call(path=path)),
            AssistantMessage("done"),
        ],
    )
    node.run("read")
    assert (
        result_for(provider.requests[-1], "read")["error_kind"] == "invalid_file_read"
    )
    assert diagnostic in result_for(provider.requests[-1], "read")["error"]
    assert replies(provider.requests[-1])[-1].get_name() == "read-file"


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(inspect_file, id="inspect"),
        pytest.param(partial(read_file, start_index=1, end_index=1), id="read"),
        pytest.param(partial(approve_edit, evidence=()), id="edit"),
        pytest.param(partial(new_file, content="OK"), id="new-file"),
        pytest.param(new_directory, id="new-directory"),
        pytest.param(delete_file, id="delete"),
    ],
)
def test_surrogate_paths_have_one_shared_validation_error(tmp_path, operation):
    with pytest.raises(FileInspectionError) as caught:
        operation(tmp_path, "bad\ud800path")
    assert caught.value.code == "invalid_path"
    assert "Unicode" in str(caught.value)


def test_inspection_proof_is_read_only_and_never_a_report_field(tmp_path):
    (tmp_path / "notes.txt").write_text("content")
    report = inspect_file(tmp_path, "notes.txt")
    with pytest.raises(AttributeError):
        report.evidence = None
    assert "evidence" not in report
