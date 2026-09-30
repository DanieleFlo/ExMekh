from __future__ import annotations

import base64
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from qs.core.agents.exit_tool.items import ExitItemError, ExitItemResolver
from qs.core.cache.node_cache import NodeCache
from qs.core.cache.session_cache import SessionCache
from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.tool_calls.execution_history import ExecutionHistory


def _save(
    history: ExecutionHistory,
    cache: NodeCache,
    *,
    name: str,
    call_id: str,
    output_id: str,
    result: object,
    skill_path: str,
    source: str = "Tool",
) -> None:
    entry = history.record(name, call_id)
    cache.store(
        ToolCallRecord(
            name=name,
            id=output_id,
            tool_call_id=call_id,
            args={},
            result=result,
            skill_path=skill_path,
            source=source,
        )
    )
    entry.attach_output(output_id, cache.session_dir)


def test_empty_items_return_text_without_opening_cache() -> None:
    resolver = ExitItemResolver(ExecutionHistory(), lambda: pytest.fail("cache opened"))

    assert resolver.compose("only text", [], "/parent", 0) == "only text"


def test_appends_selected_outputs_in_order_as_escaped_xml(tmp_path: Path) -> None:
    with SessionCache(tmp_path) as session:
        cache = session.node("main")
        history = ExecutionHistory()
        opening = history.record("parent", "open")
        _save(
            history,
            cache,
            name="first",
            call_id="call-1",
            output_id="out-1",
            result="A < B & C",
            skill_path="/parent",
        )
        _save(
            history,
            cache,
            name="child",
            call_id="call-2",
            output_id="out-2",
            result={"text": "child <result>", "items": []},
            skill_path="/parent/child",
            source="DeepSkill",
        )

        output = ExitItemResolver(history, lambda: cache).compose(
            "summary", ["out-2", "out-1"], "/parent", opening.position
        )

        assert output.startswith("summary\n\n")
        first, second = output.split("\n\n")[1:]
        child = ET.fromstring(first)
        tool = ET.fromstring(second)
        assert (child.attrib, child.findtext("content")) == (
            {"id": "out-2", "name": "child", "type": "DeepSkill"},
            "child <result>",
        )
        assert tool.findtext("content") == "A < B & C"


@pytest.mark.parametrize(
    ("listed", "placeholder"),
    [("out", "call"), ("call", "out")],
)
def test_crossed_ids_and_repeated_placeholder_resolve_one_output(
    tmp_path: Path, listed: str, placeholder: str
) -> None:
    with SessionCache(tmp_path) as session:
        cache = session.node("main")
        history = ExecutionHistory()
        opening = history.record("parent", "open")
        _save(
            history,
            cache,
            name="tool",
            call_id="call",
            output_id="out",
            result="payload",
            skill_path="/parent",
        )

        output = ExitItemResolver(history, lambda: cache).compose(
            f"before {{{{{placeholder}}}}} between {{{{{placeholder}}}}} after",
            [listed],
            "/parent",
            opening.position,
        )

        assert output.count('<tool-output id="out"') == 2
        assert output.endswith("</tool-output> after")


def test_scope_requires_current_invocation_and_same_node(tmp_path: Path) -> None:
    with SessionCache(tmp_path) as session:
        cache = session.node("main")
        other_node = session.node("other")
        history = ExecutionHistory()
        _save(
            history,
            cache,
            name="old",
            call_id="old-call",
            output_id="old-out",
            result="old",
            skill_path="/parent",
        )
        opening = history.record("parent", "open")
        _save(
            history,
            cache,
            name="sibling",
            call_id="sibling-call",
            output_id="sibling-out",
            result="other",
            skill_path="/sibling",
        )
        _save(
            history,
            other_node,
            name="foreign",
            call_id="foreign-call",
            output_id="foreign-out",
            result="other node",
            skill_path="/parent",
        )
        _save(
            history,
            cache,
            name="nested",
            call_id="nested-call",
            output_id="nested-out",
            result="allowed",
            skill_path="/parent/nested",
        )
        resolver = ExitItemResolver(history, lambda: cache)

        assert "allowed" in resolver.compose(
            "done", ["nested-out"], "/parent", opening.position
        )
        for identifier in ("old-out", "sibling-out"):
            with pytest.raises(ExitItemError, match="another skill or invocation"):
                resolver.compose("done", [identifier], "/parent", opening.position)
        with pytest.raises(ExitItemError, match="has no saved"):
            resolver.compose("done", ["foreign-out"], "/parent", opening.position)


def test_rejects_unsaved_unknown_duplicate_ambiguous_and_unlisted_ids(
    tmp_path: Path,
) -> None:
    with SessionCache(tmp_path) as session:
        cache = session.node("main")
        history = ExecutionHistory()
        opening = history.record("parent", "open")
        history.record("memory", "unsaved")
        history.record("stale", "stale-call").attach_output(
            "stale-out", cache.session_dir
        )
        _save(
            history,
            cache,
            name="one",
            call_id="reused",
            output_id="out-1",
            result="one",
            skill_path="/parent",
        )
        _save(
            history,
            cache,
            name="two",
            call_id="reused",
            output_id="out-2",
            result="two",
            skill_path="/parent",
        )
        _save(
            history,
            cache,
            name="three",
            call_id="unique",
            output_id="out-3",
            result="three",
            skill_path="/parent",
        )
        resolver = ExitItemResolver(history, lambda: cache)

        cases = [
            ("done", ["missing"], "is not present"),
            ("done", ["unsaved"], "has no saved"),
            ("done", ["stale-out"], "has no saved"),
            ("done", ["reused"], "multiple outputs"),
            ("done", ["out-1", "reused"], "multiple outputs"),
            ("done", ["out-1", "out-1"], "more than once"),
            ("done", ["out-3", "unique"], "more than once"),
            ("see {{out-2}}", ["out-1"], "is not listed"),
            ("see {{out-1}}", [], "requires its ID in items"),
            ("see {{}}", ["out-1"], "Empty placeholder"),
            ("see {{}}", [], "Empty placeholder"),
            ("see {{out-1}", ["out-1"], "Incomplete placeholder"),
        ]
        for text, items, message in cases:
            with pytest.raises(ExitItemError, match=message):
                resolver.compose(text, items, "/parent", opening.position)


def test_control_character_is_preserved_as_base64_xml(tmp_path: Path) -> None:
    with SessionCache(tmp_path) as session:
        cache = session.node("main")
        history = ExecutionHistory()
        opening = history.record("parent", "open")
        _save(
            history,
            cache,
            name="tool",
            call_id="call",
            output_id="out",
            result="a\x00b",
            skill_path="/parent",
        )

        output = ExitItemResolver(history, lambda: cache).compose(
            "done", ["out"], "/parent", opening.position
        )

        block = ET.fromstring(output.split("\n\n", 1)[1])
        content = block.find("content")
        assert content is not None
        assert content.attrib == {"encoding": "base64"}
        assert base64.b64decode(content.text or "").decode("utf-8") == "a\x00b"


def test_positive_depth_root_scope_accepts_nested_path(tmp_path: Path) -> None:
    with SessionCache(tmp_path) as session:
        cache = session.node("main")
        history = ExecutionHistory()
        _save(
            history,
            cache,
            name="nested",
            call_id="call",
            output_id="out",
            result="nested output",
            skill_path="/nested",
        )

        output = ExitItemResolver(history, lambda: cache).compose(
            "done", ["out"], "/", 0
        )

        assert "nested output" in output


@pytest.mark.parametrize("result", ["first\r\nsecond\rthird", "a\x00b", "a\ud800b"])
def test_embedded_outputs_preserve_xml_unsafe_metadata_and_content(tmp_path, result):
    name = "work\x00"
    output_id = "out\x01"
    with SessionCache(tmp_path) as session:
        cache = session.node("main")
        history = ExecutionHistory()
        _save(
            history,
            cache,
            name=name,
            call_id="call",
            output_id=output_id,
            result=result,
            skill_path="/parent",
        )

        output = ExitItemResolver(history, lambda: cache).compose(
            "done", [output_id], "/parent", 0
        )

    root = ET.fromstring(output.split("\n\n", 1)[1])
    for attribute, expected in (("name", name), ("id", output_id)):
        assert root.get(f"{attribute}-encoding") == "base64"
        assert (
            base64.b64decode(root.get(attribute)).decode("utf-8", "surrogatepass")
            == expected
        )
    content = root.find("content")
    actual = content.text
    if content.get("encoding") == "base64":
        actual = base64.b64decode(actual).decode("utf-8", "surrogatepass")
    assert actual == result
