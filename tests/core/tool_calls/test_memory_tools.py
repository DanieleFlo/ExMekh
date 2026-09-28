from __future__ import annotations

import base64
from xml.etree import ElementTree as ET

import pytest

from qs.core.cache import SessionCache, ToolCallRecord
from qs.core.models.tool import PreformattedToolResult
from qs.core.tool_calls.memory_tools import ExecutionHistory, HarnessMemoryTools


def _content(result: PreformattedToolResult) -> str:
    return result.content


def test_history_preserves_control_characters_in_call_identifiers(tmp_path) -> None:
    call_id = "call\x00\ud800"
    history = ExecutionHistory()
    history.record("work", call_id)
    with SessionCache(tmp_path) as session:
        memory = HarnessMemoryTools(history, lambda: session.node("node"))
        root = ET.fromstring(memory.get_execution_history().content)

    identifier = root.find("call/id")
    assert identifier.get("encoding") == "base64"
    assert base64.b64decode(identifier.text).decode("utf-8", "surrogatepass") == call_id


def test_saved_unfinished_execution_is_not_reported_as_a_final_result(tmp_path) -> None:
    with SessionCache(tmp_path) as session:
        cache = session.node("node")
        cache.store(
            ToolCallRecord(
                "work",
                "output",
                {},
                {"status": "running"},
                execution={
                    "status": "running",
                    "tool_call_id": "call",
                    "started_at": "then",
                },
            )
        )
        memory = HarnessMemoryTools(ExecutionHistory(), lambda: cache)
        output = memory.read_tool_output("output").content
        assert "<status>unavailable</status>" in output
        assert "<response-type>failed</response-type>" in output
        assert "not resumed" in output


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_saved_terminal_state_retains_its_xml_type(tmp_path, status) -> None:
    with SessionCache(tmp_path) as session:
        cache = session.node("node")
        record = ToolCallRecord(
            "work", "output", {}, "diagnostic", execution={"status": status}
        )
        cache.store(record)
        memory = HarnessMemoryTools(ExecutionHistory(), lambda: cache)
        assert (
            f"<response-type>{status}</response-type>"
            in memory.read_tool_output("output").content
        )
        assert ToolCallRecord.from_dict(record.to_dict()).execution == record.execution


def test_valid_result_with_negative_status_remains_final(tmp_path) -> None:
    with SessionCache(tmp_path) as session:
        cache = session.node("node")
        cache.store(
            ToolCallRecord("work", "output", {}, {"status": "failed", "ok": False})
        )
        memory = HarnessMemoryTools(ExecutionHistory(), lambda: cache)
        assert (
            "<response-type>final</response-type>"
            in memory.read_tool_output("output").content
        )


def test_history_reports_only_live_recoverable_output_ids(tmp_path) -> None:
    with SessionCache(tmp_path) as session:
        cache = session.node("node")
        cache.store(ToolCallRecord("read_file", "output-1", {}, "content"))
        history = ExecutionHistory()
        history.record("read_file", "call-1")
        history.attach_output("call-1", "output-1", cache.session_dir)
        history.record("analysis-skill", "call-2")
        history.attach_output("call-2", "missing", cache.session_dir)
        history.attach_output("unknown-call", "ignored", cache.session_dir)
        memory = HarnessMemoryTools(history, lambda: cache)

        xml = _content(memory.get_execution_history())

        assert xml == (
            "<execution-history>\n"
            "  <total>2</total>\n"
            "  <view>overview</view>\n"
            "  <call>\n"
            "    <position>1</position>\n"
            "    <name>read_file</name>\n"
            "    <id>call-1</id>\n"
            "    <saved-output-id>output-1</saved-output-id>\n"
            "  </call>\n"
            "  <call>\n"
            "    <position>2</position>\n"
            "    <name>analysis-skill</name>\n"
            "    <id>call-2</id>\n"
            "  </call>\n"
            "</execution-history>"
        )


def test_history_overview_prioritizes_the_first_five_and_last_fifteen(
    tmp_path,
) -> None:
    history = ExecutionHistory()
    for position in range(1, 101):
        history.record(f"tool-{position}", f"call-{position}")

    with SessionCache(tmp_path) as session:
        memory = HarnessMemoryTools(history, lambda: session.node("node"))

        root = ET.fromstring(_content(memory.get_execution_history()))

    assert root.findtext("total") == "100"
    assert root.findtext("view") == "overview"
    assert [int(position.text) for position in root.findall("./call/position")] == [
        *range(1, 6),
        *range(86, 101),
    ]
    assert root.findtext("./omitted/from-position") == "6"
    assert root.findtext("./omitted/to-position") == "85"
    assert root.findtext("./omitted/count") == "80"
    assert [child.tag for child in root] == [
        "total",
        "view",
        *(["call"] * 5),
        "omitted",
        *(["call"] * 15),
    ]


@pytest.mark.parametrize(
    ("limit", "head_count", "tail_count"),
    [(2, 1, 1), (8, 2, 6), (21, 5, 16), (100, 25, 75)],
)
def test_history_overview_uses_a_quarter_of_the_limit_for_the_head(
    limit: int,
    head_count: int,
    tail_count: int,
) -> None:
    history = ExecutionHistory()
    for position in range(1, 102):
        history.record("tool", f"call-{position}")

    view = history.select(limit=limit)

    assert [entry.position for entry in view.entries] == [
        *range(1, head_count + 1),
        *range(102 - tail_count, 102),
    ]
    assert view.omitted is not None
    assert view.omitted.start_position == head_count + 1
    assert view.omitted.end_position == 101 - tail_count
    assert view.omitted.count == 101 - limit


@pytest.mark.parametrize("total", [0, 19, 20])
def test_history_overview_returns_all_entries_without_an_omission(
    total: int,
) -> None:
    history = ExecutionHistory()
    for position in range(1, total + 1):
        history.record("tool", f"call-{position}")

    view = history.select()

    assert view.mode == "overview"
    assert view.total == total
    assert [entry.position for entry in view.entries] == list(range(1, total + 1))
    assert view.omitted is None


@pytest.mark.parametrize(
    (
        "start_position",
        "positions",
        "effective_range",
        "previous_position",
        "next_position",
    ),
    [
        (1, [1, 2, 3, 4], (1, 4), None, 5),
        (5, [5, 6, 7, 8], (5, 8), 1, 9),
        (9, [9, 10], (9, 10), 5, None),
        (10, [10], (10, 10), 6, None),
        (11, [], (None, None), 9, None),
        (100, [], (None, None), 9, None),
    ],
)
def test_history_pages_are_contiguous_and_navigable(
    start_position: int,
    positions: list[int],
    effective_range: tuple[int | None, int | None],
    previous_position: int | None,
    next_position: int | None,
) -> None:
    history = ExecutionHistory()
    for position in range(1, 11):
        history.record("tool", f"call-{position}")

    view = history.select(start_position=start_position, limit=4)

    assert view.mode == "page"
    assert view.total == 10
    assert [entry.position for entry in view.entries] == positions
    assert (view.start_position, view.end_position) == effective_range
    assert view.previous_position == previous_position
    assert view.next_position == next_position


def test_empty_history_page_has_no_range_or_cursor() -> None:
    view = ExecutionHistory().select(start_position=1)

    assert view.total == 0
    assert view.entries == ()
    assert view.start_position is None
    assert view.end_position is None
    assert view.previous_position is None
    assert view.next_position is None


def test_empty_history_xml_has_no_range_or_cursor(tmp_path) -> None:
    with SessionCache(tmp_path) as session:
        memory = HarnessMemoryTools(ExecutionHistory(), lambda: session.node("node"))

        xml = _content(memory.get_execution_history(start_position=1))

    assert xml == (
        "<execution-history>\n"
        "  <total>0</total>\n"
        "  <view>page</view>\n"
        "</execution-history>"
    )


@pytest.mark.parametrize("start_position", [True, False, 0, -1, 1.5, "1"])
def test_history_rejects_invalid_start_positions(start_position: object) -> None:
    with pytest.raises(
        ValueError,
        match="start_position must be a positive integer or None",
    ):
        ExecutionHistory().select(start_position=start_position)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", [True, False, 1, 101, 2.5, "20"])
def test_history_rejects_invalid_limits(limit: object) -> None:
    with pytest.raises(ValueError, match="limit must be between 2 and 100"):
        ExecutionHistory().select(limit=limit)  # type: ignore[arg-type]


def test_history_page_xml_exposes_the_effective_range_and_cursors(
    tmp_path,
) -> None:
    history = ExecutionHistory()
    for position in range(1, 11):
        history.record("tool", f"call-{position}")

    with SessionCache(tmp_path) as session:
        memory = HarnessMemoryTools(history, lambda: session.node("node"))

        xml = _content(memory.get_execution_history(start_position=5, limit=4))

    assert xml.startswith(
        "<execution-history>\n"
        "  <total>10</total>\n"
        "  <view>page</view>\n"
        "  <start-position>5</start-position>\n"
        "  <end-position>8</end-position>\n"
        "  <previous-position>1</previous-position>\n"
        "  <next-position>9</next-position>\n"
    )
    root = ET.fromstring(xml)
    assert [int(position.text) for position in root.findall("./call/position")] == [
        5,
        6,
        7,
        8,
    ]


def test_search_returns_filtered_paginated_xml_previews(tmp_path) -> None:
    with SessionCache(tmp_path) as session:
        cache = session.node("node")
        cache.store(
            ToolCallRecord(
                "fetch",
                "one",
                {},
                "hello " + "x" * 300,
                skill_path="/analysis",
            )
        )
        cache.store(
            ToolCallRecord("fetch", "two", {}, "hello again", skill_path="/analysis")
        )
        cache.store(ToolCallRecord("other", "three", {}, "hello"))
        memory = HarnessMemoryTools(ExecutionHistory(), lambda: cache)

        first = ET.fromstring(
            _content(
                memory.search_tool_outputs(
                    query="hello",
                    tool_name="fetch",
                    skill_path="/analysis",
                    limit=1,
                )
            )
        )

        assert first.findtext("total") == "2"
        assert first.findtext("next-offset") == "1"
        record = first.find("./records/record")
        assert record is not None
        assert record.findtext("name") == "fetch"
        assert record.findtext("type") == "Tool"
        assert record.findtext("id") == "one"
        preview = record.findtext("preview")
        assert preview is not None
        assert len(preview) == 240
        assert preview.endswith("…")

        second = ET.fromstring(
            _content(
                memory.search_tool_outputs(
                    query="hello",
                    tool_name="fetch",
                    skill_path="/analysis",
                    offset=1,
                    limit=1,
                )
            )
        )
        assert second.find("next-offset") is None
        assert second.findtext("./records/record/id") == "two"


def test_read_returns_header_arguments_and_display_content(tmp_path) -> None:
    with SessionCache(tmp_path) as session:
        cache = session.node("node")
        cache.store(
            ToolCallRecord(
                "analysis-skill",
                "skill-output",
                {"topic": "Rome & Lazio", "options": {"depth": 2}},
                {"text": "complete result", "items": ["done"]},
                source="DeepSkill",
            )
        )
        memory = HarnessMemoryTools(ExecutionHistory(), lambda: cache)

        content = _content(memory.read_tool_output("skill-output"))

        header, result = content.split("</header>\n", maxsplit=1)
        parsed = ET.fromstring(header + "</header>")
        assert parsed.findtext("name") == "analysis-skill"
        assert parsed.findtext("type") == "DeepSkill"
        assert parsed.findtext("id") == "skill-output"
        assert parsed.findtext("./args/topic") == "Rome & Lazio"
        assert parsed.findtext("./args/options") == '{"depth":2}'
        assert result == "complete result"


def test_read_missing_id_is_non_destructive_xml(tmp_path) -> None:
    with SessionCache(tmp_path) as session:
        memory = HarnessMemoryTools(ExecutionHistory(), lambda: session.node("node"))

        assert _content(memory.read_tool_output("missing")) == (
            "<saved-tool-output>\n"
            "  <id>missing</id>\n"
            "  <found>false</found>\n"
            "</saved-tool-output>"
        )


def test_memory_tools_are_transient_and_have_reserved_names(tmp_path) -> None:
    with SessionCache(tmp_path) as session:
        memory = HarnessMemoryTools(ExecutionHistory(), lambda: session.node("node"))

        assert memory.tool_names == (
            "get_execution_history",
            "search_tool_outputs",
            "read_tool_output",
        )
        assert all(tool.cache_result is False for tool in memory.tools)


def test_history_does_not_link_reused_id_from_a_new_session(tmp_path) -> None:
    first = SessionCache(tmp_path / "first")
    first_cache = first.node("node")
    first_cache.store(ToolCallRecord("old", "reused", {}, "old result"))
    history = ExecutionHistory()
    history.record("old", "call-old")
    history.attach_output("call-old", "reused", first_cache.session_dir)
    first.close()

    with SessionCache(tmp_path / "second") as second:
        second_cache = second.node("node")
        second_cache.store(ToolCallRecord("new", "reused", {}, "new result"))
        memory = HarnessMemoryTools(history, lambda: second_cache)

        assert "saved-output-id" not in _content(memory.get_execution_history())


def test_history_attaches_output_to_exact_position_when_call_id_repeats(
    tmp_path,
) -> None:
    with SessionCache(tmp_path) as session:
        history = ExecutionHistory()
        first = history.record("skill", "same-call")
        second = history.record("skill", "same-call")

        history.attach_output_at_position(
            first.position, "first-output", session.session_dir
        )

        assert first.saved_output_id == "first-output"
        assert second.saved_output_id is None
        with pytest.raises(ValueError, match="out of range"):
            history.attach_output_at_position(0, "invalid", session.session_dir)
