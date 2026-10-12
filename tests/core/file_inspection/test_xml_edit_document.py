"""XML framing, initial-snapshot composition and whole-call authorization regressions."""

import pytest

from exmekh_lib.core.file_inspection import editing
from exmekh_lib.core.file_inspection.edit_selection import select_replacements
from exmekh_lib.core.file_inspection.editing import (
    approve_edit,
    edit_files,
    file_edit_context,
)
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.file_inspection.interactive import FileInteractiveCalls
from exmekh_lib.core.file_inspection.runtime import FileToolRuntime
from exmekh_lib.core.models.edit_document import EditDocument
from exmekh_lib.core.models.tool_hooks import ToolHooks
from exmekh_lib.interfaces.tool_hook import BeforeTool, ToolArguments
from tests.core.file_inspection.test_edit_filesystem import evidence_for
from tests.interactive_support import (
    edit_arguments,
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


@pytest.mark.parametrize("earlier_call", [False, True])
def test_second_resource_rejection_discards_its_entire_call_but_preserves_prior_calls(
    tmp_path, monkeypatch, earlier_call
):
    (tmp_path / "notes.txt").write_text("KEEP")

    def reject(body):
        assert body["agentic"]["interactive"]["reply"]["status"] == "reject"
        return {
            "type": "interactive.rejected",
            "version": 2,
            "continuation_id": "cont-edit",
            "call_id": "edit",
            "tool_name": "edit-file",
            "edit_index": 2,
            "exchange_id": "second",
            "values": {"path": "notes.txt"},
            "code": "uninspected",
            "message": "inspect first",
            "output": response(
                (
                    "earlier",
                    "edit-file",
                    edit_arguments("earlier.txt", "<all/>", "COMMITTED"),
                )
            )["output"]
            if earlier_call
            else [],
        }

    steps = [enter()]
    if earlier_call:
        steps += [pause("earlier.txt", call_id="earlier", exchange="earlier")]
    steps += [
        pause("new.txt"),
        pause(edit_index=2, exchange="second"),
        reject,
        exit_skill(),
        response(text="done"),
    ]
    node, server, provider = make_sdk_node(tmp_path, monkeypatch, steps)
    try:
        result = node.run("create then reject")
        assert not (tmp_path / "new.txt").exists()
        assert (tmp_path / "notes.txt").read_text() == "KEEP"
        assert (tmp_path / "earlier.txt").exists() == earlier_call
        if earlier_call:
            assert (tmp_path / "earlier.txt").read_text() == "COMMITTED"
        assert (
            len(
                [
                    message
                    for message in result.skill_executions[0].history
                    if message.get_role_name() == "tool"
                    and message.get_name() == "inspect-file"
                ]
            )
            == 1
        )
    finally:
        provider.get_session().close()
        server.client.close()


def test_each_edit_uses_its_own_final_close_and_preserves_exact_bodies():
    document = EditDocument.parse(
        ' \r\n<edit path="a&amp;&quot;.txt"><old_text><![CDATA[  a\r\n]]></old_text>'
        "<new_text> \r</new_text>literal&#10;&amp;<![CDATA[</new_text></edit>]]></new_text>\t</edit>\n"
        '<edit path="b.txt"><old_text>&lt;all/&gt;</old_text><new_text><![CDATA[<delete/>]]></new_text></edit>\t'
    )
    first, second = document.elements
    assert first.path == 'a&".txt'
    assert first.selection.text == "  a\r\n"
    assert first.replacement == " \r</new_text>literal\n&</new_text></edit>"
    assert second.selection.kind == "literal"
    assert second.selection.text == "<all/>"
    assert second.replacement == "<delete/>"


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
    "start,end,expected",
    [
        ("1", "2", "NEWthree\n"),
        ("2", "3", "one\r\nNEW"),
        ("0002", "0002", "one\r\nNEWthree\n"),
    ],
)
def test_range_uses_inclusive_physical_lines_and_original_newlines(
    start, end, expected
):
    document = EditDocument.parse(
        f'<edit path="x"><old_text start_line="{start}" end_line="{end}"></old_text><new_text>NEW</new_text></edit>'
    )
    assert (
        select_replacements("one\r\ntwo\rthree\n", document.elements, exists=True)
        == expected
    )


@pytest.mark.parametrize(
    "start,end", [("0", "1"), ("1", "0"), ("3", "2"), ("1", "4"), ("9" * 5000, "2")]
)
def test_range_lexical_acceptance_does_not_authorize_invalid_numeric_ranges(start, end):
    document = EditDocument.parse(
        f'<edit path="x"><old_text start_line="{start}" end_line="{end}"></old_text><new_text>x</new_text></edit>'
    )
    with pytest.raises(FileInspectionError):
        select_replacements("one\ntwo\nthree", document.elements, exists=True)


@pytest.mark.parametrize(
    "old_one,old_two",
    [
        ("one\ntwo", "two"),
        ("one\ntwo", '<line number="2"/>'),
        ('<line number="2"/>', '<line number="2"/>'),
        ("<all/>", "two"),
    ],
)
def test_same_file_conflicts_are_detected_before_composition(old_one, old_two):
    document = EditDocument.parse(
        edit_fragment("x", old_one, "A") + edit_fragment("x", old_two, "B")
    )
    with pytest.raises(FileInspectionError, match="initial snapshot"):
        select_replacements("one\ntwo\nthree", document.elements, exists=True)


def test_nonoverlapping_selections_and_boundary_insertions_use_initial_offsets():
    document = EditDocument.parse(
        edit_fragment("x", "two", "SECOND")
        + edit_fragment("x", '<line number="2"/>', "BEFORE\n")
        + edit_fragment("x", "one", "FIRST")
    )
    assert (
        select_replacements("one\ntwo\nthree", document.elements, exists=True)
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
    document = edit_fragment("a.txt", "<all/>", "BAD") + edit_fragment(
        "b.txt", "<all/>", "BAD"
    )
    with file_edit_context(approvals, evidence), pytest.raises(FileInspectionError):
        edit_files(tmp_path, document)
    assert (tmp_path / "a.txt").read_text() == "KEEP"
    assert (tmp_path / "b.txt").read_text() == "CONCURRENT"


@pytest.mark.parametrize("change", ["cancel", "scope"])
def test_execution_rechecks_cancellation_and_scope_after_hooks(tmp_path, change):
    cancelled = False
    scope = ("original",)
    handler = FileInteractiveCalls(
        FileToolRuntime(), tmp_path, lambda: (), lambda: cancelled, lambda: scope
    )
    handler.begin()
    assert handler.prepare(pause("new.txt"), {"behavior": "edit_file.v2"}) == {
        "status": "create"
    }
    if change == "cancel":
        cancelled = True
    else:
        scope = ("different",)
    with (
        handler.execution_context("edit"),
        pytest.raises(FileInspectionError, match="invalidated"),
    ):
        edit_files(tmp_path, edit_fragment("new.txt", "<all/>", "BAD"))
    assert not (tmp_path / "new.txt").exists()


def test_cancellation_between_distinct_file_commits_reports_partial_progress(
    tmp_path, monkeypatch
):
    cancelled = False
    handler = FileInteractiveCalls(
        FileToolRuntime(), tmp_path, lambda: (), lambda: cancelled, lambda: ()
    )
    handler.begin()
    for index, path in enumerate(("a.txt", "b.txt"), 1):
        assert handler.prepare(
            pause(path, edit_index=index), {"behavior": "edit_file.v2"}
        ) == {"status": "create"}
    commit = editing.commit_edit

    def cancel_after_first(*args, **kwargs):
        nonlocal cancelled
        commit(*args, **kwargs)
        cancelled = True

    monkeypatch.setattr(editing, "commit_edit", cancel_after_first)
    with handler.execution_context("edit"):
        result = edit_files(
            tmp_path,
            edit_fragment("a.txt", "<all/>", "A")
            + edit_fragment("b.txt", "<all/>", "B"),
        )
    assert not result["ok"]
    assert result["error_kind"] == "partial_file_edit"
    assert result["files"] == [
        {"path": "a.txt", "operation": "create", "size_bytes": 1}
    ]
    assert result["failed_path"] == "b.txt"
    assert (tmp_path / "a.txt").read_bytes() == b"A"
    assert not (tmp_path / "b.txt").exists()


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
                second += "broken"
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
