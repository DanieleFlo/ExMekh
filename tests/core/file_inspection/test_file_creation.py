"""Ordered creation batches preserve bytes and isolate rejected resources."""

import os
from dataclasses import replace
from xml.etree import ElementTree as ET

import pytest

from exmekh_lib.config import file_editing
from exmekh_lib.core.agents.capabilities.capabilities import Capabilities
from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.file_inspection import edit_filesystem
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.models import creation_document
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.tool_hooks import ToolHooks
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.interfaces.tool_hook import AfterTool, BeforeTool, ToolArguments
from exmekh_lib.utils.xml import stringify_tool_result
from tests.core.agents.node.test_node import SequenceProvider
from tests.tool_result_support import parse_tool_result


@pytest.fixture
def creation_tool(tmp_path):
    loaded = Capabilities(None, default_capabilities=DefaultCapabilities()).load(
        work_root=tmp_path
    )
    return next(tool for tool in loaded.tools if tool.name == "new-file")


def test_creates_directories_and_literal_files_in_order(tmp_path, creation_tool):
    content = "\r\n<all/>&lt;<![CDATA[raw]]>😊\t "
    result = creation_tool.execute(
        document='<new-exmekh-dir path="reports"/>'
        '<new-exmekh-dir path="reports/nested"/>'
        '<new-exmekh-file path="reports/nested/a.txt"/>'
        + content
        + '<new-exmekh-file path="reports/b.txt"/>SECOND'
    )
    assert result["ok"] is True
    assert [item["resource_type"] for item in result["results"]] == [
        "directory",
        "directory",
        "file",
        "file",
    ]
    assert [item["index"] for item in result["results"]] == [1, 2, 3, 4]
    assert (tmp_path / "reports/nested/a.txt").read_bytes() == content.encode()
    assert (tmp_path / "reports/b.txt").read_bytes() == b"SECOND"
    assert parse_tool_result(stringify_tool_result(result)) == result


def test_continues_after_independent_failures_without_rollback(tmp_path, creation_tool):
    (tmp_path / "existing").write_bytes(b"KEEP")
    result = creation_tool.execute(
        document='<new-exmekh-file path="first"/>FIRST'
        '<new-exmekh-file path="existing"/>BAD'
        '<new-exmekh-file path="empty"/>'
        '<new-exmekh-file path="short"/>x'
        '<new-exmekh-file path="missing/file"/>BAD'
        '<new-exmekh-dir path="existing"/>'
        '<new-exmekh-file path="last"/>LAST'
    )
    assert result["ok"] is False
    assert [item["ok"] for item in result["results"]] == [
        True,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert [item["error_code"] for item in result["results"][1:-1]] == [
        "file_already_exists",
        "invalid_content",
        "invalid_content",
        "parent_directory_missing",
        "file_already_exists",
    ]
    assert all(item["hint"] for item in result["results"][1:-1])
    assert (tmp_path / "existing").read_bytes() == b"KEEP"
    assert (tmp_path / "first").read_bytes() == b"FIRST"
    assert (tmp_path / "last").read_bytes() == b"LAST"
    assert not (tmp_path / "empty").exists()
    assert not (tmp_path / "short").exists()


def test_directory_content_produces_warning_without_failure(tmp_path, creation_tool):
    result = creation_tool.execute(
        document='<new-exmekh-dir path="folder"/>ignored <markup/>\r\n'
        '<new-exmekh-file path="folder/a"/>OK'
    )
    assert result["ok"] is True
    assert result["results"][0]["warnings"]
    assert result["results"][1].get("warnings", []) == []
    assert list((tmp_path / "folder").iterdir()) == [tmp_path / "folder/a"]


@pytest.mark.parametrize(
    "malformed",
    [
        '<new-exmekh-file path="bad">BAD',
        '<new-exmekh-file path="bad" extra="x"/>BAD',
        "<new-exmekh-file/>BAD",
        '<new-exmekh-dir path="bad"',
    ],
)
def test_recovers_at_next_marker_after_malformed_block(
    tmp_path, creation_tool, malformed
):
    result = creation_tool.execute(
        document='<new-exmekh-file path="first"/>FIRST'
        + malformed
        + '<new-exmekh-file path="last"/>LAST'
    )
    assert result["ok"] is False
    assert [item["ok"] for item in result["results"]] == [True, False, True]
    assert result["results"][1]["error_code"] == "invalid_creation_document"
    assert (tmp_path / "first").read_bytes() == b"FIRST"
    assert (tmp_path / "last").read_bytes() == b"LAST"
    assert not (tmp_path / "bad").exists()


def test_reports_leading_garbage_and_processes_following_entries(
    tmp_path, creation_tool
):
    result = creation_tool.execute(
        document='unexpected<new-exmekh-file path="good"/>OK'
    )
    assert result["ok"] is False
    assert result["results"][0]["error_code"] == "invalid_creation_document"
    assert result["results"][1]["ok"] is True
    assert (tmp_path / "good").read_bytes() == b"OK"


@pytest.mark.parametrize("document", ["", " \t\r\n", "not a creation", None, []])
def test_rejects_invalid_documents_without_writes(tmp_path, creation_tool, document):
    result = creation_tool.execute(document=document)
    assert result["ok"] is False
    assert result["error_code"] == "invalid_creation_document"
    assert not list(tmp_path.iterdir())


def test_duplicate_destinations_do_not_overwrite(tmp_path, creation_tool):
    result = creation_tool.execute(
        document='<new-exmekh-file path="same"/>FIRST'
        '<new-exmekh-file path="same"/>SECOND'
        '<new-exmekh-dir path="same"/>'
    )
    assert [item["ok"] for item in result["results"]] == [True, False, False]
    assert (tmp_path / "same").read_bytes() == b"FIRST"


def test_decodes_only_path_entities_and_preserves_whitespace(tmp_path, creation_tool):
    result = creation_tool.execute(
        document=' \n<new-exmekh-file path="a&amp;b" />&lt;\r\n'
        '<new-exmekh-file path="spaces"/>\t '
    )
    assert result["ok"] is True
    assert result["results"][0]["path"] == "a&b"
    assert (tmp_path / "a&b").read_bytes() == b"&lt;\r\n"
    assert (tmp_path / "spaces").read_bytes() == b"\t "


@pytest.mark.parametrize("kind", ["file", "dir"])
@pytest.mark.parametrize("path", ["", "../outside", "missing/child"])
def test_invalid_paths_do_not_stop_later_entries(tmp_path, creation_tool, kind, path):
    result = creation_tool.execute(
        document=f'<new-exmekh-{kind} path="{path}"/>'
        + ("BAD" if kind == "file" else "")
        + '<new-exmekh-file path="good"/>OK'
    )
    assert result["ok"] is False
    assert result["results"][0]["ok"] is False
    assert result["results"][1]["ok"] is True
    assert (tmp_path / "good").read_bytes() == b"OK"


@pytest.mark.parametrize(
    "limit", ["max_creation_document_bytes", "max_creation_elements"]
)
def test_document_limits_reject_before_any_write(
    tmp_path, creation_tool, monkeypatch, limit
):
    settings = replace(file_editing.FILE_EDITING, **{limit: 1})
    monkeypatch.setattr(file_editing, "FILE_EDITING", settings)
    monkeypatch.setattr(creation_document, "FILE_EDITING", settings)
    result = creation_tool.execute(
        document='<new-exmekh-file path="first"/>OK<new-exmekh-file path="last"/>OK'
    )
    assert result["ok"] is False
    assert result["error_code"] == "file_limit_exceeded"
    assert "1" in result["error"] and "no resources were created" in result["error"]
    assert "smaller creation calls" in result["hint"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["file", "dir"])
@pytest.mark.parametrize("existing", ["file", "directory", "symlink", "dangling"])
def test_existing_entries_are_never_replaced(tmp_path, creation_tool, kind, existing):
    path = tmp_path / "existing"
    target = tmp_path / "target"
    if existing == "file":
        path.write_bytes(b"KEEP")
    elif existing == "directory":
        path.mkdir()
    else:
        if existing == "symlink":
            target.write_bytes(b"KEEP")
        path.symlink_to(target)
    before = path.lstat()
    result = creation_tool.execute(document=f'<new-exmekh-{kind} path="existing"/>BAD')
    assert result["ok"] is False
    assert path.lstat().st_ino == before.st_ino
    if existing == "file":
        assert path.read_bytes() == b"KEEP"
    if existing in {"symlink", "dangling"}:
        assert path.is_symlink()


def test_directory_parent_symlink_cannot_escape_workroot(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "alias").symlink_to(outside, target_is_directory=True)
    with pytest.raises(FileInspectionError) as caught:
        edit_filesystem.new_directory(root, "alias/new")
    assert caught.value.code == "path_outside_workroot"
    assert not list(outside.iterdir())


@pytest.mark.parametrize("redirect", [False, True])
def test_directory_parent_change_prevents_creation_and_closes_descriptor(
    tmp_path, monkeypatch, redirect
):
    parent = tmp_path / "parent"
    parent.mkdir()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    original_open = edit_filesystem._open_parent_directory
    descriptors = []

    def change_parent(root, target):
        descriptor = original_open(root, target)
        descriptors.append(descriptor)
        parent.rename(tmp_path / "original")
        if redirect:
            parent.symlink_to(replacement, target_is_directory=True)
        else:
            parent.mkdir()
        return descriptor

    monkeypatch.setattr(edit_filesystem, "_open_parent_directory", change_parent)
    with pytest.raises(FileInspectionError) as caught:
        edit_filesystem.new_directory(tmp_path, "parent/new")
    assert caught.value.code == "file_changed"
    assert not list((tmp_path / "original").iterdir())
    assert not list(parent.iterdir())
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_directory_creation_reports_unsupported_platform(tmp_path, monkeypatch):
    monkeypatch.setattr(edit_filesystem.os, "supports_dir_fd", set())
    with pytest.raises(FileInspectionError) as caught:
        edit_filesystem.new_directory(tmp_path, "new")
    assert caught.value.code == "unsupported_platform"
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize(
    "failure", [PermissionError("denied"), FileExistsError("race")]
)
def test_directory_mkdir_failures_are_reported_without_stopping_files(
    tmp_path, creation_tool, monkeypatch, failure
):
    def fail_mkdir(*args, **kwargs):
        raise failure

    monkeypatch.setattr(edit_filesystem.os, "mkdir", fail_mkdir)
    monkeypatch.setattr(
        edit_filesystem.os, "supports_dir_fd", os.supports_dir_fd | {fail_mkdir}
    )
    result = creation_tool.execute(
        document='<new-exmekh-dir path="folder"/><new-exmekh-file path="good"/>OK'
    )
    assert result["ok"] is False
    assert result["results"][0]["error_code"] == (
        "permission_denied"
        if isinstance(failure, PermissionError)
        else "file_already_exists"
    )
    assert result["results"][1]["ok"] is True
    assert not (tmp_path / "folder").exists()


@pytest.mark.parametrize("content", ["BAD\x00BODY", "BAD\ud800BODY"])
def test_invalid_content_is_rejected_per_file(tmp_path, creation_tool, content):
    result = creation_tool.execute(
        document='<new-exmekh-file path="bad"/>'
        + content
        + '<new-exmekh-file path="good"/>OK'
    )
    assert [item["ok"] for item in result["results"]] == [False, True]
    assert result["results"][0]["error_code"] == "invalid_content"
    assert not (tmp_path / "bad").exists()
    assert (tmp_path / "good").read_bytes() == b"OK"


def test_creation_hooks_run_once_for_batch_and_saved_output_retains_xml(tmp_path):
    document = (
        '<new-exmekh-dir path="folder"/>ignored<new-exmekh-file path="folder/good"/>OK'
    )
    hook_calls = []

    class Before(BeforeTool):
        def handle(self, context):
            hook_calls.append("before")
            assert context.arguments == {"document": "replace this"}
            return ToolArguments({"document": document})

    class After(AfterTool):
        def handle(self, context, result):
            hook_calls.append("after")
            assert context.arguments == {"document": document}
            assert len(result["results"]) == 2
            return result

    provider = SequenceProvider(
        [
            AssistantMessage(
                None, tool_calls=[ToolCall("open", FunctionCall("file-processing", {}))]
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "create", FunctionCall("new-file", {"document": "replace this"})
                    )
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "exit",
                        FunctionCall("exit_agent", {"text": "done", "items": []}),
                    )
                ],
            ),
            AssistantMessage("done"),
        ]
    )
    node = Node(
        name_hds="main",
        default_capabilities=DefaultCapabilities(
            include={"main"},
            tool_hooks={"new-file": ToolHooks(before=Before(), after=After())},
        ),
    )
    try:
        node.initialize(provider, work_root=tmp_path)
        result = node.run("create")
        assert hook_calls == ["before", "after"]
        messages = result.skill_executions[0].history
        created = next(
            item
            for item in messages
            if isinstance(item, ToolMessage) and item.get_name() == "new-file"
        )
        receipt = parse_tool_result(created.get_message())
        assert receipt["ok"] is True and receipt["results"][0]["warnings"]
        assert (tmp_path / "folder/good").read_bytes() == b"OK"
        assert not any(
            item.get_name() in {"inspect-file", "read-file"} for item in messages
        )
        assert not any(
            isinstance(item, ToolMessage) and item.get_name() == "new-file"
            for item in provider.requests[-1].get_messages()
        )
        header = ET.fromstring(
            created.get_message().split("</header>", 1)[0] + "</header>"
        )
        output_id = header.findtext("id")
        provider.queue_responses(
            [
                AssistantMessage(
                    None,
                    tool_calls=[
                        ToolCall(
                            "reread",
                            FunctionCall("read_tool_output", {"output_id": output_id}),
                        )
                    ],
                ),
                AssistantMessage("retrieved"),
            ]
        )
        assert node.run("retrieve").get_message() == "retrieved"
        reread = next(
            item
            for item in reversed(provider.requests[-1].get_messages())
            if isinstance(item, ToolMessage)
        )
        assert parse_tool_result(reread.get_message()) == receipt
        assert hook_calls == ["before", "after"]
    finally:
        provider.get_session().close()


def test_rejected_nul_path_survives_shared_xml_renderer(creation_tool):
    result = creation_tool.execute(document='<new-exmekh-file path="bad\x00name"/>OK')
    assert result["results"][0]["error_code"] == "invalid_path"
    rendered = stringify_tool_result(result)
    assert 'encoding="base64"' in rendered
    assert parse_tool_result(rendered) == result


def test_per_file_byte_limit_does_not_prevent_smaller_creation(
    tmp_path, creation_tool, monkeypatch
):
    monkeypatch.setattr(
        edit_filesystem,
        "FILE_EDITING",
        replace(file_editing.FILE_EDITING, max_snapshot_bytes=3),
    )
    result = creation_tool.execute(
        document='<new-exmekh-file path="large"/>éé<new-exmekh-file path="small"/>OK'
    )
    assert result["results"][0]["error_code"] == "file_limit_exceeded"
    assert result["results"][1]["ok"] is True
    assert not (tmp_path / "large").exists()
    assert (tmp_path / "small").read_bytes() == b"OK"


def test_concurrent_file_creation_has_the_same_diagnostic_as_directory_collision(
    tmp_path, creation_tool, monkeypatch
):
    original_link = os.link

    def competing_create(source, destination, **kwargs):
        if destination == "raced":
            (tmp_path / "raced").write_bytes(b"CONCURRENT")
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", competing_create)
    monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, competing_create})
    result = creation_tool.execute(
        document='<new-exmekh-file path="raced"/>BAD<new-exmekh-file path="good"/>OK'
    )
    assert result["results"][0]["error_code"] == "file_already_exists"
    assert result["results"][1]["ok"] is True
    assert (tmp_path / "raced").read_bytes() == b"CONCURRENT"
    assert not list(tmp_path.glob(".exmekh-edit-*"))
