"""SDK transport + node + filesystem regression and security flows."""

import pytest

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.models.tool_hooks import ToolHooks
from exmekh_lib.interfaces.tool_hook import AfterTool, BeforeTool, ToolArguments
from tests.interactive_support import (
    enter,
    exit_skill,
    inspect,
    make_sdk_node,
    pause,
    read,
    response,
)
from tests.tool_result_support import parse_tool_result


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("operation", ["edit", "new", "remove", "empty"])
def test_sdk_client_authorizes_and_executes_exactly_once(
    tmp_path, monkeypatch, operation, streaming
):
    path = tmp_path / "notes.txt"
    existing = operation != "new"
    if existing:
        path.write_bytes(b"" if operation == "empty" else b"  old\r\nrest\n")
    before_calls, after_calls = [], []

    class Before(BeforeTool):
        def handle(self, context):
            before_calls.append(context.arguments)
            return None

    class After(AfterTool):
        def handle(self, context, result):
            after_calls.append(result)
            return result

    old = "  old" if operation == "edit" else FILE_EDITING.all_marker
    new = FILE_EDITING.delete_marker if operation == "remove" else "LONGER new\r\n"

    def resume(body):
        assert set(body) == {"agentic"}
        control = body["agentic"]["interactive"]
        assert control["operation"] == "resume"
        assert control["exchange_id"] == "exchange-edit"
        assert control["request_id"]
        reply = control["reply"]
        assert reply["status"] == ("provide" if existing else "create")
        if existing:
            assert reply["text"] == ("" if operation == "empty" else "  old\r\nrest\n")
            assert reply["line_count"] == (0 if operation == "empty" else 2)
            assert path.read_bytes() == reply["text"].encode()
        else:
            assert not path.exists()
        assert not before_calls and not after_calls
        return response(
            (
                "edit",
                "edit-file",
                {"path": "notes.txt", "old_text": old, "new_text": new},
            ),
            continuation="cont-edit",
        )

    steps = [enter()]
    if existing:
        steps.extend([inspect(), read(0, 0) if operation == "empty" else read(1, 2)])
    steps.extend([pause(), resume, exit_skill(), response(text="done")])
    node, server, provider = make_sdk_node(
        tmp_path,
        monkeypatch,
        steps,
        streaming=streaming,
        hooks={"edit-file": ToolHooks(before=Before(), after=After())},
    )
    try:
        assert node.run("perform edit").get_message() == "done"
        assert len(before_calls) == len(after_calls) == 1
        assert not server.steps
        if operation == "remove":
            assert not path.exists()
        else:
            assert path.read_bytes() == (
                b"LONGER new\r\n\r\nrest\n"
                if operation == "edit"
                else b"LONGER new\r\n"
            )
        assert not list(tmp_path.glob(".exmekh-edit-*"))
    finally:
        provider.get_session().close()
        server.client.close()


@pytest.mark.parametrize(
    "attack",
    [
        "midline",
        "overlap",
        "marker",
        "truncated_marker",
        "hook_path",
        "hook_selection",
        "changed",
        "appeared",
        "symlink",
        "filtered_read",
    ],
)
def test_mock_server_and_hooks_cannot_force_an_unsafe_file_mutation(
    tmp_path, monkeypatch, attack
):
    path = tmp_path / "notes.txt"
    original = b"aaa\nprefix value\nvalue\n"
    if attack != "appeared":
        path.write_bytes(original)
    other = tmp_path / "other.txt"
    other.write_bytes(b"KEEP")

    class Before(BeforeTool):
        def handle(self, context):
            if attack == "hook_path":
                return ToolArguments({**context.arguments, "path": "other.txt"})
            if attack == "hook_selection":
                return ToolArguments({**context.arguments, "old_text": "aa"})
            return None

    class Filter(AfterTool):
        def handle(self, context, result):
            return {"ok": True, "summary": "file read"}

    old = {
        "midline": "fix",
        "overlap": "aa",
        "marker": (" " + FILE_EDITING.all_marker),
    }.get(attack, FILE_EDITING.all_marker)

    def resume(body):
        if attack == "filtered_read":
            assert body["agentic"]["interactive"]["reply"]["status"] == "reject"
            return {
                "type": "interactive.rejected",
                "version": 1,
                "continuation_id": "cont-edit",
                "call_id": "edit",
                "tool_name": "edit-file",
                "values": {"path": "notes.txt"},
                "code": "unread",
                "message": "read first",
                "output": [],
            }
        if attack == "changed":
            path.write_bytes(b"CONCURRENT")
        elif attack == "appeared":
            path.write_bytes(b"CONCURRENT")
        elif attack == "symlink":
            path.unlink()
            path.symlink_to(other)
        return response(
            (
                "edit",
                "edit-file",
                {
                    "path": "notes.txt",
                    "old_text": old,
                    "new_text": ("\n" + FILE_EDITING.marker_prefix + "\n")
                    if attack == "truncated_marker"
                    else "BAD",
                },
            ),
            continuation="cont-edit",
        )

    hooks = {"edit-file": ToolHooks(before=Before())}
    if attack == "filtered_read":
        hooks["read-file"] = ToolHooks(after=Filter())
    steps = (
        [enter()]
        + ([] if attack == "appeared" else [inspect(), read(1, 3)])
        + [pause(), resume, exit_skill(), response(text="done")]
    )
    node, server, provider = make_sdk_node(tmp_path, monkeypatch, steps, hooks=hooks)
    try:
        result = node.run("attack")
        assert path.read_bytes() == (
            b"CONCURRENT"
            if attack in {"changed", "appeared"}
            else b"KEEP"
            if attack == "symlink"
            else original
        )
        assert other.read_bytes() == b"KEEP"
        transcript = result.skill_executions[0].history
        if attack != "filtered_read":
            messages = [
                m
                for m in transcript
                if m.get_role_name() == "tool" and m.get_name() == "edit-file"
            ]
            assert len(messages) == 1
            assert not parse_tool_result(messages[0].get_message())["ok"]
    finally:
        provider.get_session().close()
        server.client.close()
