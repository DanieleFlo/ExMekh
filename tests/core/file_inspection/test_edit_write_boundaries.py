"""Failure boundaries must preserve external writes and report partial commits."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.core.file_inspection import edit_filesystem
from exmekh_lib.core.file_inspection.edit_filesystem import (
    FileWriteTarget,
    delete_file,
    new_file,
)
from exmekh_lib.core.file_inspection.edit_selection import select_replacements
from exmekh_lib.core.file_inspection.editing import (
    approve_edit,
    edit_files,
    file_edit_context,
)
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.file_inspection.evidence import ReadReport
from exmekh_lib.core.file_inspection.interactive import FileInteractiveCalls
from exmekh_lib.core.file_inspection.runtime import FileToolRuntime
from exmekh_lib.core.file_inspection.workspace import read_snapshot
from exmekh_lib.core.models.edit_document import EditDocument, literal_selection_offset
from tests.core.file_inspection.test_edit_filesystem import evidence_for
from tests.interactive_support import edit_fragment


@pytest.mark.parametrize("content", ["bad\x00text", "ééé"])
def test_invalid_creation_never_publishes_a_file(tmp_path, monkeypatch, content):
    monkeypatch.setattr(
        edit_filesystem, "FILE_EDITING", replace(FILE_EDITING, max_snapshot_bytes=4)
    )
    with pytest.raises(FileInspectionError):
        new_file(tmp_path, "new.txt", content)
    assert list(tmp_path.iterdir()) == []


def test_deletion_rejects_a_write_between_initial_stat_and_directory_open(
    tmp_path, monkeypatch
):
    path = tmp_path / "notes.txt"
    path.write_text("old")
    open_parent = edit_filesystem._open_parent_directory

    def competing_write(root, target):
        descriptor = open_parent(root, target)
        target.write_text("external writer")
        return descriptor

    monkeypatch.setattr(edit_filesystem, "_open_parent_directory", competing_write)
    with pytest.raises(FileInspectionError, match="changed"):
        delete_file(tmp_path, path.name)
    assert path.read_text() == "external writer"


def test_write_target_rejects_redirected_and_existing_resources(tmp_path):
    existing = tmp_path / "existing.txt"
    existing.write_text("keep")
    for destination in (existing, tmp_path / "wrong.txt"):
        with pytest.raises(FileInspectionError, match="changed"):
            FileWriteTarget(tmp_path, existing.name, destination).commit(
                b"must not publish"
            )
    assert existing.read_text() == "keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == [existing.name]


def test_second_commit_failure_reports_the_first_commit_and_keeps_second_file(
    tmp_path, monkeypatch
):
    paths = [tmp_path / "one.txt", tmp_path / "two.txt"]
    for path in paths:
        path.write_text("OLD")
    evidence = tuple(
        item for path in paths for item in evidence_for(tmp_path, path.name)
    )
    approvals = tuple(approve_edit(tmp_path, path.name, evidence) for path in paths)
    original_replace = edit_filesystem.os.replace

    def replace_once(source, destination, **kwargs):
        if destination == paths[1].name:
            raise PermissionError("second commit denied")
        return original_replace(source, destination, **kwargs)

    monkeypatch.setattr(edit_filesystem.os, "replace", replace_once)
    document = "".join(edit_fragment(path.name, "OLD", "NEW") for path in paths)
    with file_edit_context(approvals, evidence):
        result = edit_files(tmp_path, document)
    assert result["ok"] is False
    assert result["error_kind"] == "partial_file_edit"
    assert result["failed_path"] == paths[1].name
    assert result["error_code"] == "permission_denied"
    assert "do not replay" in result["hint"]
    assert [entry["path"] for entry in result["files"]] == [paths[0].name]
    assert [path.read_text() for path in paths] == ["NEW", "OLD"]
    assert not list(tmp_path.glob(FILE_EDITING.temporary_file_prefix + "*"))


@pytest.mark.parametrize("lines", [None, [1], [{"text": 1}]])
def test_read_report_does_not_invent_text_for_malformed_lines(tmp_path, lines):
    path = tmp_path / "text.txt"
    path.write_text("text")
    report = ReadReport(
        {"lines": lines}, read_snapshot(tmp_path, path.name, FILE_INSPECTION)
    )
    assert report.selected_text() is None


def test_empty_file_requires_one_empty_old_body_and_ids_cannot_repeat():
    assert literal_selection_offset("", "") == 0
    assert literal_selection_offset("", "nonempty") is None
    document = EditDocument.parse(edit_fragment("empty.txt", "wrong", "new"))
    with pytest.raises(FileInspectionError):
        select_replacements("", document.elements)
    repeated = edit_fragment("one.txt", "old", "new", "427") + edit_fragment(
        "two.txt", "old", "new", "427"
    )
    with pytest.raises(ValueError, match="[Ii][Dd]"):
        EditDocument.parse(repeated)


@pytest.mark.parametrize("change", ["cancel", "scope"])
def test_approved_edit_rechecks_cancellation_and_scope_before_writing(tmp_path, change):
    path = tmp_path / "notes.txt"
    path.write_text("OLD")
    evidence = evidence_for(tmp_path, path.name)
    state = {"cancel": False, "scope": ("owner",)}
    runtime = MagicMock(spec=FileToolRuntime)
    runtime.visible_evidence.return_value = evidence
    handler = FileInteractiveCalls(
        runtime, tmp_path, lambda: [], lambda: state["cancel"], lambda: state["scope"]
    )
    handler.begin()
    reply = handler.prepare(
        {
            "call_id": "call",
            "request": {"edit_index": 1, "values": {"path": path.name}},
        },
        {"behavior": "edit_file.v3"},
    )
    assert reply["status"] == "provide"
    state[change] = True if change == "cancel" else ("another-owner",)
    with handler.execution_context("call"), pytest.raises(FileInspectionError):
        edit_files(tmp_path, edit_fragment(path.name, "OLD", "NEW"))
    assert path.read_text() == "OLD"


def test_reused_separator_id_cannot_authorize_an_opening_without_an_id():
    first = edit_fragment("one.txt", "old", "new", "427")
    second = edit_fragment("two.txt", "old", "new").replace(
        "<exmekh_new_text/>", '<exmekh_new_text id="427"/>'
    )
    with pytest.raises(ValueError, match="Duplicate edit ID"):
        EditDocument.parse(first + second, snapshots=("old", "old"))
