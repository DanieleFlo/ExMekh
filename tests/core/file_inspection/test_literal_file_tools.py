"""Public v3 file contracts, literal framing and independent create/delete calls."""

import pytest

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.file_inspection import edit_filesystem
from exmekh_lib.core.file_inspection.edit_filesystem import delete_file, new_file
from exmekh_lib.core.file_inspection.edit_selection import select_replacements
from exmekh_lib.core.file_inspection.editing import (
    approve_edit,
    edit_files,
    file_edit_context,
)
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.models.edit_document import EditDocument
from exmekh_lib.core.models.tool_hooks import ToolHooks
from exmekh_lib.interfaces.tool_hook import BeforeTool
from tests.core.file_inspection.test_edit_filesystem import evidence_for
from tests.interactive_support import (
    edit_fragment,
    enter,
    exit_skill,
    make_sdk_node,
    response,
)
from tests.tool_result_support import parse_tool_result


@pytest.mark.parametrize("identifier", [None, "000", "427", "999"])
def test_bodies_are_literal_and_preserve_all_whitespace(identifier):
    old = '  <all/>\r\n<line number="1"/>&lt;<![CDATA[old]]>'
    new = "<delete/>\r\n</new_text></edit>&amp;\t "
    document = EditDocument.parse(edit_fragment("notes.txt", old, new, identifier))
    element = document.elements[0]
    assert element.old_text == old
    assert element.replacement == new
    assert select_replacements(old, document.elements) == new


@pytest.mark.parametrize(
    "opening,separator",
    [
        (' id="427"', ' id="427"'),
        (' id="427"', ""),
        ("", ' id="427"'),
        ("", ""),
    ],
)
def test_missing_ids_are_accepted_when_unambiguous(opening, separator):
    document = EditDocument.parse(
        f'<exmekh_edit path="x"{opening} />old<exmekh_new_text{separator}/>new'
    )
    assert document.elements[0].old_text == "old"
    assert document.elements[0].replacement == "new"


def test_multiple_blocks_without_ids_preserve_exact_trailing_text():
    document = EditDocument.parse(
        edit_fragment("a", "A", "NEW\r\n") + edit_fragment("b", "B", "END\t ")
    )
    assert [item.path for item in document.elements] == ["a", "b"]
    assert [item.replacement for item in document.elements] == ["NEW\r\n", "END\t "]


def test_other_ids_in_body_are_literal():
    old = '<exmekh_new_text id="999"/>'
    document = EditDocument.parse(edit_fragment("x", old, "NEW", "427"))
    assert document.elements[0].old_text == old


@pytest.mark.parametrize(
    "source",
    [
        '<exmekh_edit path="x" />old<exmekh_new_text/>a<exmekh_new_text/>b',
        '<exmekh_edit path="x" id="427" />old<exmekh_new_text id="999"/>new',
        '<exmekh_edit path="x" id="42" />old<exmekh_new_text/>new',
        '<exmekh_edit path="x" number="1" />old<exmekh_new_text/>new',
        '<exmekh_edit path="x" />old',
    ],
)
def test_ambiguous_or_invalid_framing_is_rejected(source):
    with pytest.raises(ValueError):
        EditDocument.parse(source)


@pytest.mark.parametrize(
    "old",
    [
        "<all/>",
        '<line number="1"/>',
        '<old_text start_line="1" end_line="2"></old_text>',
    ],
)
def test_old_commands_have_no_selection_semantics(old):
    document = EditDocument.parse(edit_fragment("x", old, "BAD"))
    with pytest.raises(FileInspectionError):
        select_replacements("KEEP", document.elements)


def test_empty_old_is_only_valid_for_empty_text():
    document = EditDocument.parse(edit_fragment("x", "", "FILLED"))
    assert select_replacements("", document.elements) == "FILLED"
    with pytest.raises(FileInspectionError):
        select_replacements("KEEP", document.elements)


@pytest.mark.parametrize("content", ["", "a", "😊"])
def test_new_requires_at_least_two_characters(tmp_path, content):
    with pytest.raises(FileInspectionError):
        new_file(tmp_path, "new.txt", content)
    assert not (tmp_path / "new.txt").exists()


@pytest.mark.parametrize("content", ["ab", "😊è", "\t ", "one\r\ntwo\r"])
def test_new_preserves_utf8_and_does_not_require_evidence(tmp_path, content):
    result = new_file(tmp_path, "new.txt", content)
    assert result["operation"] == "create"
    assert (tmp_path / "new.txt").read_bytes() == content.encode()


@pytest.mark.parametrize("kind", ["empty", "full", "directory", "symlink", "dangling"])
def test_new_never_overwrites_any_existing_entry(tmp_path, kind):
    path = tmp_path / "target"
    if kind == "directory":
        path.mkdir()
    elif kind in {"symlink", "dangling"}:
        target = tmp_path / "link-target"
        if kind == "symlink":
            target.write_bytes(b"KEEP")
        path.symlink_to(target)
    else:
        path.write_bytes(b"" if kind == "empty" else b"KEEP")
    with pytest.raises((FileInspectionError, OSError)):
        new_file(tmp_path, path.name, "BAD")
    assert path.exists() or path.is_symlink()


@pytest.mark.parametrize("path", ["missing", "../outside", "", "nested/missing"])
def test_delete_rejects_missing_or_invalid_resources(tmp_path, path):
    with pytest.raises((FileInspectionError, OSError)):
        delete_file(tmp_path, path)


def test_delete_does_not_decode_or_limit_file_contents(tmp_path):
    path = tmp_path / "large.bin"
    path.write_bytes(b"\x00\xff" * FILE_EDITING.max_snapshot_bytes)
    assert delete_file(tmp_path, path.name)["operation"] == "remove"
    assert not path.exists()
    folder = tmp_path / "folder"
    folder.mkdir()
    assert delete_file(tmp_path, folder.name)["operation"] == "remove"
    assert not folder.exists()


def test_delete_reports_unsupported_platform_before_opening_directory(
    tmp_path, monkeypatch
):
    path = tmp_path / "notes.txt"
    path.write_text("KEEP")
    monkeypatch.setattr(edit_filesystem.os, "supports_dir_fd", set())
    monkeypatch.setattr(
        edit_filesystem,
        "_open_parent_directory",
        lambda *args: pytest.fail("unsupported operations must not open a directory"),
    )
    with pytest.raises(FileInspectionError, match="platform"):
        delete_file(tmp_path, path.name)
    assert path.read_text() == "KEEP"


def test_separately_approved_aliases_use_snapshot_identity(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("FIRST\nSECOND\n")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(path)
    evidence = evidence_for(tmp_path, path.name)
    approvals = tuple(
        approve_edit(tmp_path, resource, evidence)
        for resource in (path.name, alias.name)
    )
    assert approvals[0].snapshot.matches(approvals[1].snapshot)
    edits = edit_fragment(path.name, "FIRST", "ONE") + edit_fragment(
        alias.name, "SECOND", "TWO"
    )
    with file_edit_context(approvals, evidence):
        result = edit_files(tmp_path, edits)
    assert result["ok"]
    assert len(result["files"]) == 1
    assert path.read_text() == alias.read_text() == "ONE\nTWO\n"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_new_delete_use_normal_hooks_without_inspect_or_read(
    tmp_path, monkeypatch, streaming, missing
):
    hook_calls = []

    class Before(BeforeTool):
        def handle(self, context):
            hook_calls.append(context.arguments)
            return None

    hooks = {name: ToolHooks(before=Before()) for name in ("new-file", "delete-file")}
    node, server, provider = make_sdk_node(
        tmp_path,
        monkeypatch,
        [
            enter(),
            response(
                (
                    "new",
                    "new-file",
                    {"document": '<new-exmekh-file path="notes.txt"/>created\r\n'},
                ),
                continuation="cont-new",
            ),
            response(
                (
                    "delete",
                    "delete-file",
                    {
                        "paths": ["missing<&>.txt", "notes.txt"]
                        if missing
                        else ["notes.txt"]
                    },
                ),
                continuation="cont-delete",
            ),
            exit_skill(),
            response(text="done"),
        ],
        streaming=streaming,
        hooks=hooks,
    )
    try:
        result = node.run("create then delete")
        assert len(hook_calls) == 2
        assert not (tmp_path / "notes.txt").exists()
        assert not server.steps
        history = result.skill_executions[0].history
        results = [
            parse_tool_result(item.get_message())
            for item in history
            if item.get_role_name() == "tool" and item.get_name() in hooks
        ]
        assert len(results) == 2 and results[0]["ok"] is True
        assert results[1]["ok"] is (not missing)
        if missing:
            failed, deleted = results[1]["results"]
            assert failed["path"] == "missing<&>.txt"
            assert failed["error_code"] == "file_not_found"
            assert deleted["ok"] is True
        assert not any(
            item.get_name() in {"inspect-file", "read-file"} for item in history
        )
    finally:
        provider.get_session().close()
        server.client.close()


def test_edit_empty_file_retains_evidence_checks_and_existing_identity(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    with pytest.raises(FileInspectionError):
        approve_edit(tmp_path, path.name, ())
    evidence = evidence_for(tmp_path, path.name)
    approval = approve_edit(tmp_path, path.name, evidence)
    with file_edit_context((approval,), evidence):
        edit_files(tmp_path, edit_fragment(path.name, "", "FILLED"))
    assert path.read_bytes() == b"FILLED"


def test_cancelled_execution_does_not_commit_any_edit(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("KEEP")
    evidence = evidence_for(tmp_path, path.name)
    approval = approve_edit(tmp_path, path.name, evidence)

    def cancelled():
        raise FileInspectionError("invalidated by cancellation")

    with (
        file_edit_context((approval,), cancelled),
        pytest.raises(FileInspectionError, match="invalidated"),
    ):
        edit_files(tmp_path, edit_fragment(path.name, "KEEP", "BAD"))
    assert path.read_text() == "KEEP"


def test_snapshot_disambiguates_a_missing_separator_id(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("value\nvalue<exmekh_new_text/>unique")
    evidence = evidence_for(tmp_path, path.name)
    approval = approve_edit(tmp_path, path.name, evidence)
    source = edit_fragment(path.name, "value<exmekh_new_text/>unique", "CHANGED")
    with file_edit_context((approval,), evidence):
        edit_files(tmp_path, source)
    assert path.read_text() == "value\nCHANGED"


def test_explicit_separator_keeps_bare_marker_as_literal_content():
    document = EditDocument.parse(
        edit_fragment("x", "OLD", "<exmekh_new_text/>", "427")
    )
    assert document.elements[0].replacement == "<exmekh_new_text/>"
