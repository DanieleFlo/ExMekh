"""Real filesystem failure paths, identity checks and exclusive atomic creation."""

import codecs
import os
from dataclasses import replace

import pytest

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.file_inspection import edit_filesystem, editing
from exmekh_lib.core.file_inspection.editing import (
    approve_edit,
    edit_file,
    file_edit_context,
)
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.file_inspection.reading import file_read_context, read_file
from exmekh_lib.core.file_inspection.service import inspect_file


def evidence_for(tmp_path, path):
    inspection = inspect_file(tmp_path, path)
    with file_read_context((inspection.evidence,)):
        lines = inspection["total_lines"]
        read = read_file(tmp_path, path, 1 if lines else 0, lines)
    return inspection.evidence, read.evidence


@pytest.mark.parametrize(
    "resource",
    [
        "../outside.txt",
        "/tmp/outside-edit-test.txt",
        "missing-parent/new.txt",
        "",
        "bad\x00path",
        "folder",
    ],
)
def test_invalid_resources_never_authorize_creation(tmp_path, resource):
    (tmp_path / "folder").mkdir()
    with pytest.raises((FileInspectionError, OSError)):
        approve_edit(tmp_path, resource, ())


def test_dangling_and_outside_symlinks_do_not_authorize_creation(tmp_path):
    dangling = tmp_path / "dangling.txt"
    dangling.symlink_to(tmp_path / "missing.txt")
    outside = tmp_path / "outside.txt"
    outside.symlink_to(tmp_path.parent / "elsewhere.txt")
    for resource in (dangling.name, outside.name):
        with pytest.raises((FileInspectionError, OSError)):
            approve_edit(tmp_path, resource, ())
    assert not (tmp_path / "missing.txt").exists()


def test_read_only_file_is_not_overwritten_using_parent_write_permissions(
    tmp_path, monkeypatch
):
    path = tmp_path / "notes.txt"
    path.write_text("KEEP")
    evidence = evidence_for(tmp_path, path.name)
    approval = approve_edit(tmp_path, path.name, evidence)
    monkeypatch.setattr(edit_filesystem.os, "access", lambda *args, **kwargs: False)
    with (
        file_edit_context(approval, evidence),
        pytest.raises(FileInspectionError, match="not writable"),
    ):
        edit_file(tmp_path, path.name, FILE_EDITING.all_marker, "BAD")
    assert path.read_text() == "KEEP"


def test_exclusive_commit_rejects_a_file_appearing_after_the_final_absence_check(
    tmp_path, monkeypatch
):
    path = tmp_path / "notes.txt"
    approval = approve_edit(tmp_path, path.name, ())
    original_link = os.link

    def competing_create(*args, **kwargs):
        path.write_text("CONCURRENT")
        return original_link(*args, **kwargs)

    monkeypatch.setattr(edit_filesystem.os, "link", competing_create)
    # supports_dir_fd describes the original implementation, not the test wrapper.
    monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, competing_create})
    with file_edit_context(approval, ()), pytest.raises(FileExistsError):
        edit_file(tmp_path, path.name, FILE_EDITING.all_marker, "BAD")
    assert path.read_text() == "CONCURRENT"
    assert not list(tmp_path.glob(".exmekh-edit-*"))


def test_replace_failure_leaves_original_bytes_and_removes_the_prepared_file(
    tmp_path, monkeypatch
):
    path = tmp_path / "notes.txt"
    path.write_text("KEEP")
    evidence = evidence_for(tmp_path, path.name)
    approval = approve_edit(tmp_path, path.name, evidence)

    def fail_replace(*args, **kwargs):
        raise PermissionError("commit failed")

    monkeypatch.setattr(edit_filesystem.os, "replace", fail_replace)
    with file_edit_context(approval, evidence), pytest.raises(PermissionError):
        edit_file(tmp_path, path.name, FILE_EDITING.all_marker, "BAD")
    assert path.read_text() == "KEEP"
    assert not list(tmp_path.glob(".exmekh-edit-*"))


@pytest.mark.parametrize(
    "raw,filename,new",
    [
        (
            codecs.BOM_UTF16_BE + "  uno\r\nKEEP".encode("utf-16-be"),
            "notes.txt",
            "  due",
        ),
        (codecs.BOM_UTF8 + b"  uno\r\nKEEP", "notes.txt", "  due"),
        (b"# coding: latin-1\n  uno\nKEEP", "notes.py", "snowman \u2603"),
    ],
)
def test_commit_preserves_encoding_bom_newlines_and_rejects_unrepresentable_text(
    tmp_path, raw, filename, new
):
    path = tmp_path / filename
    path.write_bytes(raw)
    # A valid Python fixture must parse for inspection to succeed.
    if filename == "notes.py":
        path.write_bytes(b"# coding: latin-1\nvalue = 'uno'\n")
    evidence = evidence_for(tmp_path, filename)
    approval = approve_edit(tmp_path, filename, evidence)
    if filename == "notes.py":
        with file_edit_context(approval, evidence), pytest.raises(UnicodeEncodeError):
            edit_file(tmp_path, filename, FILE_EDITING.all_marker, new)
        assert path.read_bytes() == approval.snapshot.content
    else:
        with file_edit_context(approval, evidence):
            edit_file(tmp_path, filename, "  uno", new)
        expected = (
            codecs.BOM_UTF16_BE + "  due\r\nKEEP".encode("utf-16-be")
            if raw.startswith(codecs.BOM_UTF16_BE)
            else codecs.BOM_UTF8 + b"  due\r\nKEEP"
        )
        assert path.read_bytes() == expected


def test_snapshot_and_generated_byte_limits_reject_without_truncating(
    tmp_path, monkeypatch
):
    path = tmp_path / "notes.txt"
    path.write_text("KEEP")
    evidence = evidence_for(tmp_path, path.name)
    monkeypatch.setattr(
        edit_filesystem, "FILE_EDITING", replace(FILE_EDITING, max_snapshot_bytes=3)
    )
    with pytest.raises(FileInspectionError, match="byte limit"):
        approve_edit(tmp_path, path.name, evidence)
    monkeypatch.setattr(edit_filesystem, "FILE_EDITING", FILE_EDITING)
    approval = approve_edit(tmp_path, path.name, evidence)
    monkeypatch.setattr(
        editing, "FILE_EDITING", replace(FILE_EDITING, max_generated_bytes=3)
    )
    with (
        file_edit_context(approval, evidence),
        pytest.raises(FileInspectionError, match="byte limit"),
    ):
        edit_file(tmp_path, path.name, FILE_EDITING.all_marker, "TOO LONG")
    assert path.read_text() == "KEEP"


def test_same_bytes_with_a_new_inode_still_invalidate_the_approved_resource(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("KEEP")
    evidence = evidence_for(tmp_path, path.name)
    approval = approve_edit(tmp_path, path.name, evidence)
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("KEEP")
    replacement.replace(path)
    with (
        file_edit_context(approval, evidence),
        pytest.raises(FileInspectionError, match="changed"),
    ):
        edit_file(tmp_path, path.name, FILE_EDITING.all_marker, "BAD")
    assert path.read_text() == "KEEP"


def test_visibility_is_rechecked_at_execution_after_a_hook_removes_evidence(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("KEEP")
    evidence = evidence_for(tmp_path, path.name)
    approval = approve_edit(tmp_path, path.name, evidence)
    visible = evidence
    with file_edit_context(approval, lambda: visible):
        # A Before hook may replace history through the node's public interface.
        visible = ()
        with pytest.raises(FileInspectionError, match="visible inspection"):
            edit_file(tmp_path, path.name, FILE_EDITING.all_marker, "BAD")
    assert path.read_text() == "KEEP"


def test_changed_ancestor_cannot_redirect_temporary_writes_outside_workroot(
    tmp_path, tmp_path_factory, monkeypatch
):
    parent = tmp_path / "nested" / "folder"
    parent.mkdir(parents=True)
    path = parent / "notes.txt"
    path.write_text("KEEP")
    evidence = evidence_for(tmp_path, "nested/folder/notes.txt")
    approval = approve_edit(tmp_path, "nested/folder/notes.txt", evidence)
    outside = tmp_path_factory.mktemp("outside-edit")
    (outside / "folder").mkdir()
    outside_file = outside / "folder" / "notes.txt"
    outside_file.write_text("OUTSIDE")
    original_open = os.open

    def redirect_parent(resource, flags, *args, **kwargs):
        if resource == tmp_path and flags & os.O_DIRECTORY:
            (tmp_path / "nested").rename(tmp_path / "original")
            (tmp_path / "nested").symlink_to(outside, target_is_directory=True)
        return original_open(resource, flags, *args, **kwargs)

    monkeypatch.setattr(edit_filesystem.os, "open", redirect_parent)
    monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, redirect_parent})
    with file_edit_context(approval, evidence), pytest.raises(OSError):
        edit_file(tmp_path, "nested/folder/notes.txt", FILE_EDITING.all_marker, "BAD")
    assert outside_file.read_text() == "OUTSIDE"
    assert not list(outside.rglob(".exmekh-edit-*"))
    assert (tmp_path / "original" / "folder" / "notes.txt").read_text() == "KEEP"
