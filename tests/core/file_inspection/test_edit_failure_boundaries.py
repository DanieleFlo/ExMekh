"""Public file-edit APIs reject unsupported resources and stale authorization."""

import codecs
import os
from dataclasses import replace
from pathlib import Path

import pytest

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.agents.node import DefaultCapabilities, Node
from exmekh_lib.core.file_inspection import editing
from exmekh_lib.core.file_inspection.edit_filesystem import resolve_edit_target
from exmekh_lib.core.file_inspection.edit_selection import select_replacements
from exmekh_lib.core.file_inspection.editing import (
    approve_edit,
    edit_files,
    file_edit_context,
    require_edit_evidence,
)
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.file_inspection.interactive import FileInteractiveCalls
from exmekh_lib.core.file_inspection.reading import FileReadContext
from exmekh_lib.core.file_inspection.runtime import FileToolRuntime
from exmekh_lib.core.models.edit_document import EditDocument
from exmekh_lib.core.models.interactive import InteractiveProtocolError
from exmekh_lib.core.models.tool import Tool
from tests.core.file_inspection.test_edit_filesystem import evidence_for
from tests.interactive_support import edit_fragment, pause


@pytest.mark.parametrize(
    "source",
    [
        None,
        1,
        "<edit path=",
        '<edit path="unterminated',
        '<edit path="x"><old_text>old</old_text><new_text>unterminated',
        '<edit path="x"><old_text>&#99999999999;</old_text><new_text>x</new_text></edit>',
        '<edit path="x"><old_text>&#x110000;</old_text><new_text>x</new_text></edit>',
    ],
)
def test_incomplete_documents_and_invalid_scalars_fail(source):
    with pytest.raises(ValueError):
        EditDocument.parse(source)


def test_document_element_limit_has_no_truncation(monkeypatch):
    from_limit = replace(FILE_EDITING, max_exchanges=1)
    monkeypatch.setattr("exmekh_lib.core.models.edit_document.FILE_EDITING", from_limit)
    with pytest.raises(ValueError, match="element limit"):
        EditDocument.parse(
            edit_fragment("a", "<all/>", "") + edit_fragment("b", "<all/>", "")
        )


def test_empty_selections_and_missing_interactive_context_cannot_write(tmp_path):
    with pytest.raises(FileInspectionError):
        select_replacements("", (), exists=False)
    with pytest.raises(FileInspectionError, match="snapshot exchange"):
        edit_files(tmp_path, edit_fragment("x", "<all/>", "BAD"))
    document = EditDocument.parse(edit_fragment("x", "literal", "BAD"))
    with pytest.raises(FileInspectionError, match="Creation"):
        select_replacements("", document.elements, exists=False)


def test_unsupported_root_formats_and_decoded_text_are_not_creation(
    tmp_path, monkeypatch
):
    path = tmp_path / "notes.txt"
    path.write_text("KEEP")
    with pytest.raises(FileInspectionError, match="directory"):
        resolve_edit_target(path, "new.txt")
    with pytest.raises(FileInspectionError, match="not supported"):
        approve_edit(tmp_path, "new.pdf", ())
    evidence = evidence_for(tmp_path, path.name)
    unreadable = replace(evidence[0], readable=False)
    with pytest.raises(FileInspectionError, match="without editable"):
        require_edit_evidence(path, b"KEEP", (unreadable,))
    with pytest.raises(FileInspectionError, match="visible inspection"):
        require_edit_evidence(path, b"KEEP", (evidence[1],))
    assert FileReadContext(evidence).visible_evidence() == evidence
    monkeypatch.setattr(editing, "decode_text", lambda *_: None)
    with pytest.raises(FileInspectionError, match="Unicode text"):
        approve_edit(tmp_path, path.name, evidence)
    monkeypatch.setattr(editing, "decode_text", lambda *_: ("\x00", "utf-8"))
    with pytest.raises(FileInspectionError, match="Unicode text"):
        approve_edit(tmp_path, path.name, evidence)


def test_decoded_utf8_size_and_result_encoding_limits_are_distinct(
    tmp_path, monkeypatch
):
    path = tmp_path / "notes.txt"
    path.write_bytes(codecs.BOM_UTF16_LE + "中中".encode("utf-16-le"))
    evidence = evidence_for(tmp_path, path.name)
    monkeypatch.setattr(
        editing, "FILE_EDITING", replace(FILE_EDITING, max_snapshot_bytes=5)
    )
    with pytest.raises(FileInspectionError, match="Decoded"):
        approve_edit(tmp_path, path.name, evidence)
    monkeypatch.setattr(editing, "FILE_EDITING", FILE_EDITING)
    approval = approve_edit(tmp_path, path.name, evidence)
    monkeypatch.setattr(
        editing, "FILE_EDITING", replace(FILE_EDITING, max_snapshot_bytes=3)
    )
    with (
        file_edit_context((approval,), evidence),
        pytest.raises(FileInspectionError, match="Result"),
    ):
        edit_files(tmp_path, edit_fragment(path.name, "<all/>", "LONG"))


def test_inconsistent_alias_authorizations_cannot_be_combined(tmp_path):
    approval = approve_edit(tmp_path, "new.txt", ())
    different = replace(approval, text="unapproved")
    with (
        file_edit_context((approval, different), ()),
        pytest.raises(FileInspectionError, match="Aliases"),
    ):
        edit_files(
            tmp_path,
            edit_fragment("new.txt", '<line number="1"/>', "A")
            + edit_fragment("new.txt", '<line number="1"/>', "B"),
        )
    assert not (tmp_path / "new.txt").exists()


@pytest.mark.parametrize("kind", ["platform", "parent_permissions"])
def test_preflight_rejects_platform_and_directory_permissions(
    tmp_path, monkeypatch, kind
):
    approval = approve_edit(tmp_path, "new.txt", ())
    if kind == "platform":
        monkeypatch.setattr(os, "supports_dir_fd", set())
    else:
        monkeypatch.setattr(os, "access", lambda *_: False)
    with file_edit_context((approval,), ()), pytest.raises(FileInspectionError):
        edit_files(tmp_path, edit_fragment("new.txt", "<all/>", "BAD"))
    assert not (tmp_path / "new.txt").exists()


def test_nested_directory_creation_uses_directory_descriptors(tmp_path):
    (tmp_path / "nested" / "folder").mkdir(parents=True)
    approval = approve_edit(tmp_path, "nested/folder/new.txt", ())
    with file_edit_context((approval,), ()):
        result = edit_files(tmp_path, edit_fragment(approval.resource, "<all/>", "NEW"))
    assert result["ok"]
    assert (tmp_path / approval.resource).read_bytes() == b"NEW"


def test_disappearing_parent_is_rejected_during_resolution(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    is_dir = Path.is_dir
    monkeypatch.setattr(
        Path, "is_dir", lambda path: False if path == parent else is_dir(path)
    )
    with pytest.raises(FileInspectionError, match="parent directory"):
        resolve_edit_target(tmp_path, "parent/new.txt")
    assert not (parent / "new.txt").exists()


def test_parent_identity_is_rechecked_before_publication(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    approval = approve_edit(tmp_path, "parent/new.txt", ())
    fsync = os.fsync

    def move_parent(descriptor):
        fsync(descriptor)
        parent.rename(tmp_path / "original-parent")
        parent.mkdir()

    monkeypatch.setattr(os, "fsync", move_parent)
    with (
        file_edit_context((approval,), ()),
        pytest.raises(FileInspectionError, match="changed"),
    ):
        edit_files(tmp_path, edit_fragment("parent/new.txt", "<all/>", "BAD"))
    assert not (parent / "new.txt").exists()
    assert not (tmp_path / "original-parent" / "new.txt").exists()
    assert not list((tmp_path / "original-parent").iterdir())


def test_runtime_rejects_unknown_behavior_duplicate_indices_invalid_resources_and_steering(
    tmp_path,
):
    cancelled = False
    handler = FileInteractiveCalls(
        FileToolRuntime(), tmp_path, lambda: (), lambda: cancelled, lambda: ()
    )
    handler.begin()
    with pytest.raises(InteractiveProtocolError):
        handler.prepare(pause(), {"behavior": "unknown"})
    with pytest.raises(InteractiveProtocolError):
        handler.prepare(pause(edit_index=2), {"behavior": "edit_file.v2"})
    assert (
        handler.prepare(pause("../outside.txt"), {"behavior": "edit_file.v2"})["status"]
        == "reject"
    )
    cancelled = True
    with pytest.raises(InteractiveProtocolError):
        handler.validate_completed([])


def test_missing_file_alias_reuses_creation_snapshot_until_a_file_appears(tmp_path):
    handler = FileInteractiveCalls(
        FileToolRuntime(), tmp_path, lambda: (), lambda: False, lambda: ()
    )
    handler.begin()
    definition = {"behavior": "edit_file.v2"}
    assert handler.prepare(pause("new.txt"), definition) == {"status": "create"}
    assert handler.prepare(pause("./new.txt", edit_index=2), definition) == {
        "status": "create"
    }
    (tmp_path / "new.txt").write_text("CONCURRENT")
    reply = handler.prepare(pause("new.txt", edit_index=3), definition)
    assert reply["status"] == "reject"
    with (
        handler.execution_context("edit"),
        pytest.raises(FileInspectionError, match="snapshot exchange"),
    ):
        edit_files(tmp_path, edit_fragment("new.txt", "<all/>", "BAD"))
    assert (tmp_path / "new.txt").read_text() == "CONCURRENT"


def test_runtime_tool_override_keeps_xml_behavior_and_synchronous_execution(tmp_path):
    hds = tmp_path / "capabilities" / "hds"
    hds.mkdir(parents=True)
    (hds / "main.md").write_text(
        "---\ndescription: Edit files.\nexit-tool: exit_agent\nsub-skills:\n  - file-processing\n---\nUse file-processing."
    )

    def replacement(edits: str):
        return {"ok": True}

    node = Node(
        [hds.parent],
        name_hds="main",
        default_capabilities=DefaultCapabilities(
            harness=False,
            include={
                "file-processing",
                "inspect-file",
                "read-file",
                "edit-file",
                "exit_agent",
            },
            overrides={"edit-file": Tool(replacement, name="edit-file")},
        ),
    )
    loaded = node.load_capabilities(work_root=tmp_path)
    tool = next(tool for tool in loaded.tools if tool.name == "edit-file")
    assert not tool.background_enabled
    assert tool.interactive.behavior == "edit_file.v2"
