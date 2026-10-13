"""XML framing, initial-snapshot composition and whole-call authorization regressions."""

import pytest

from exmekh_lib.core.file_inspection.edit_selection import select_replacements
from exmekh_lib.core.file_inspection.editing import (
    approve_edit,
    edit_files,
    file_edit_context,
)
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.models.edit_document import EditDocument
from exmekh_lib.core.models.tool_hooks import ToolHooks
from exmekh_lib.interfaces.tool_hook import BeforeTool, ToolArguments
from tests.core.file_inspection.test_edit_filesystem import evidence_for
from tests.interactive_support import (
    edit_fragment,
    enter,
    exit_skill,
    inspect,
    make_sdk_node,
    pause,
    read,
    response,
)
from tests.tool_result_support import parse_tool_result


@pytest.mark.parametrize(
    "source",
    [
        "",
        "not xml",
        '<edit path="">',
        '<edit path="x" code="123456">',
        '<edit path="x"ignored">',
        '<edit path="<![CDATA[x]]>">',
        '<edit path="x"><old_text><all/> </old_text><new_text>x</new_text></edit>',
        '<edit path="x"><old_text><delete/></old_text><new_text>x</new_text></edit>',
        '<edit path="x"><old_text>old</old_text><new_text><line number="1"/></new_text></edit>',
        '<edit path="x"><old_text>old</old_text><new_text><delete/> </new_text></edit>',
        '<edit path="x"><old_text start_line="1" end_line="2">body</old_text><new_text>x</new_text></edit>',
        '<edit path="x"><old_text start_line="-1" end_line="2"></old_text><new_text>x</new_text></edit>',
        '<edit path="x"><old_text>&unknown;</old_text><new_text>x</new_text></edit>',
        '<edit path="x"><old_text>&#0;</old_text><new_text>x</new_text></edit>',
        '<edit path="x"><old_text>&#xD800;</old_text><new_text>x</new_text></edit>',
        '<edit path="x"><old_text>old</old_text><new_text><![CDATA[broken</new_text></edit>',
        '<edit path="x"><old_text>old</old_text><new_text>]]></new_text></edit>',
        '<edit path="x"><old_text>old</old_text><new_text>x</new_text code="1"></edit>',
        '<edit path="x"><old_text>old</old_text><new_text>x</new_text></edit>trailing',
        '<!DOCTYPE edit [<!ENTITY leak SYSTEM "file:///etc/passwd">]>',
        "\x00",
        "\ud800",
    ],
)
def test_untrusted_xml_is_rejected_without_repair(source):
    with pytest.raises(ValueError):
        EditDocument.parse(source)


@pytest.mark.parametrize(
    "old_one,old_two",
    [
        ("one\ntwo", "two"),
    ],
)
def test_same_file_conflicts_are_detected_before_composition(old_one, old_two):
    document = EditDocument.parse(
        edit_fragment("x", old_one, "A") + edit_fragment("x", old_two, "B")
    )
    with pytest.raises(FileInspectionError, match="initial snapshot"):
        select_replacements("one\ntwo\nthree", document.elements)


def test_nonoverlapping_selections_and_boundary_insertions_use_initial_offsets():
    document = EditDocument.parse(
        edit_fragment("x", "two", "BEFORE\nSECOND") + edit_fragment("x", "one", "FIRST")
    )
    assert (
        select_replacements("one\ntwo\nthree", document.elements)
        == "FIRST\nBEFORE\nSECOND\nthree"
    )


def test_all_resources_are_validated_before_first_write(tmp_path):
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text("KEEP")
    evidence = evidence_for(tmp_path, "a.txt") + evidence_for(tmp_path, "b.txt")
    approvals = tuple(
        approve_edit(tmp_path, name, evidence) for name in ("a.txt", "b.txt")
    )
    (tmp_path / "b.txt").write_text("CONCURRENT")
    document = edit_fragment("a.txt", "KEEP", "BAD") + edit_fragment(
        "b.txt", "KEEP", "BAD"
    )
    with file_edit_context(approvals, evidence), pytest.raises(FileInspectionError):
        edit_files(tmp_path, document)
    assert (tmp_path / "a.txt").read_text() == "KEEP"
    assert (tmp_path / "b.txt").read_text() == "CONCURRENT"


@pytest.mark.parametrize("streaming", [False, True])
def test_sdk_repeated_aliases_share_snapshot_and_commit_once(
    tmp_path, monkeypatch, streaming
):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"one\r\ntwo\n")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(path)
    exchanges = []

    def next_edit(body):
        exchanges.append(body["agentic"]["interactive"]["reply"])
        return pause("alias.txt", edit_index=2, exchange="second")

    def complete(body):
        exchanges.append(body["agentic"]["interactive"]["reply"])
        assert exchanges[0] == exchanges[1]
        return response(
            (
                "edit",
                "edit-file",
                {
                    "edits": edit_fragment("notes.txt", "one", "FIRST")
                    + edit_fragment("alias.txt", "two", "SECOND")
                },
            ),
            continuation="cont-edit",
        )

    node, server, provider = make_sdk_node(
        tmp_path,
        monkeypatch,
        [
            enter(),
            inspect(),
            read(1, 2),
            pause(),
            next_edit,
            complete,
            exit_skill(),
            response(text="done"),
        ],
        streaming=streaming,
    )
    try:
        result = node.run("edit both selections")
        assert path.read_bytes() == b"FIRST\r\nSECOND\n"
        assert alias.is_symlink()
        outputs = [
            parse_tool_result(message.get_message())
            for message in result.skill_executions[0].history
            if message.get_role_name() == "tool" and message.get_name() == "edit-file"
        ]
        assert len(outputs) == 1
        assert outputs[0]["ok"]
        assert len(outputs[0]["files"]) == 1
        assert not server.steps
    finally:
        provider.get_session().close()
        server.client.close()


@pytest.mark.parametrize(
    "attack", ["path", "conflict", "range", "syntax", "extra_element"]
)
def test_hooks_cannot_bypass_document_wide_checks(tmp_path, monkeypatch, attack):
    path = tmp_path / "notes.txt"
    path.write_text("one\ntwo\n")

    class Before(BeforeTool):
        def handle(self, context):
            first = edit_fragment("notes.txt", "one", "FIRST")
            second = edit_fragment("notes.txt", "two", "SECOND")
            if attack == "path":
                second = edit_fragment("other.txt", "two", "BAD")
            elif attack == "conflict":
                second = edit_fragment("notes.txt", "one", "BAD")
            elif attack == "range":
                second = '<edit path="notes.txt"><old_text start_line="2" end_line="99"></old_text><new_text>BAD</new_text></edit>'
            elif attack == "syntax":
                second = second.replace("<exmekh_new_text/>", "<broken/>")
            elif attack == "extra_element":
                second += edit_fragment("notes.txt", "two", "BAD")
            return ToolArguments({"edits": first + second})

    final = response(
        (
            "edit",
            "edit-file",
            {
                "edits": edit_fragment("notes.txt", "one", "FIRST")
                + edit_fragment("notes.txt", "two", "SECOND")
            },
        ),
        continuation="cont-edit",
    )
    node, server, provider = make_sdk_node(
        tmp_path,
        monkeypatch,
        [
            enter(),
            inspect(),
            read(1, 2),
            pause(),
            pause(edit_index=2, exchange="second"),
            final,
            exit_skill(),
            response(text="done"),
        ],
        hooks={"edit-file": ToolHooks(before=Before())},
    )
    try:
        result = node.run("edit")
        assert path.read_bytes() == b"one\ntwo\n"
        output = next(
            message
            for message in result.skill_executions[0].history
            if message.get_role_name() == "tool" and message.get_name() == "edit-file"
        )
        assert not parse_tool_result(output.get_message())["ok"]
    finally:
        provider.get_session().close()
        server.client.close()
