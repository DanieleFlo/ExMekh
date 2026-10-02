import base64
from threading import Event
from xml.etree import ElementTree as ET

import pytest

from qs.capabilities.tools.read_tool_output.tool_output_reader import ToolOutputReader
from qs.core.cache import SessionCache
from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.models.tool import (
    ExecutionOptions,
    PreformattedToolResult,
    ToolExecutionOperation,
)
from qs.core.tool_calls.execution_manager import ToolExecutionManager
from qs.utils.xml import stringify_tool_result


def unavailable_cache():
    raise AssertionError("Transient output must not depend on the cache")


@pytest.mark.parametrize("cursor", [-1, True, 0.5, "0"])
def test_invalid_cursors_are_rejected_before_accessing_storage(cursor):
    with pytest.raises(ValueError, match="cursor must be a non-negative integer"):
        ToolOutputReader(unavailable_cache).read("output", cursor)


def test_missing_output_identifier_with_control_characters_remains_readable(tmp_path):
    output_id = "missing\x00"
    with SessionCache(tmp_path) as session:
        reader = ToolOutputReader(lambda: session.node("node"))
        root = ET.fromstring(reader.read(output_id).content)

    assert root.findtext("found") == "false"
    identifier = root.find("id")
    assert identifier.get("encoding") == "base64"
    assert (
        base64.b64decode(identifier.text).decode("utf-8", "surrogatepass") == output_id
    )


def test_saved_structured_result_uses_the_same_xml_as_immediate_output(tmp_path):
    result = {"title": "Wikipedia", "page_dates": {"published": None}}
    record = ToolCallRecord("read-web-page", "output", {}, result)
    with SessionCache(tmp_path) as session:
        cache = session.node("node")
        cache.store(record)
        content = ToolOutputReader(lambda: cache).read("output").content
        body = content.split("</header>\n", 1)[1]
        assert body == stringify_tool_result(result)
        assert cache.get("output").result == result


def test_runtime_output_is_read_without_opening_the_cache_and_acknowledges_delivery():
    release = Event()
    manager = ToolExecutionManager()
    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(lambda: release.wait() and "done"),
        ("owner",),
        ExecutionOptions(mode="background"),
        output_id="output",
    )
    reader = ToolOutputReader(unavailable_cache, manager)
    try:
        assert "<response-type>running</response-type>" in reader.read("output").content
    finally:
        release.set()
    assert manager.wait("call", 1).is_finished
    assert manager.next_delivery(("owner",)) is not None
    assert reader.read("output").content.endswith("done")
    assert manager.next_delivery(("owner",)) is None


def test_unsuccessful_final_read_does_not_acknowledge_delivery():
    release = Event()
    manager = ToolExecutionManager()
    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(release.wait),
        ("owner",),
        ExecutionOptions(mode="background"),
        output_id="output",
    )
    release.set()
    assert manager.wait("call", 1).is_finished
    reader = ToolOutputReader(unavailable_cache, manager)
    with pytest.raises(ValueError, match="does not support cursors"):
        reader.read("output", cursor=0)
    assert manager.next_delivery(("owner",)) is not None


def test_saved_running_read_keeps_warning_outside_character_cursor(tmp_path):
    original = ToolCallRecord(
        "work",
        "output",
        {"label": "saved"},
        "prima\nrisultato è parziale",
        tool_call_id="call",
        execution={"status": "running", "tool_call_id": "call", "started_at": "then"},
    )
    cursor = len("prima\n")
    with SessionCache(tmp_path) as session:
        cache = session.node("owner")
        cache.store(original)
        reader = ToolOutputReader(lambda: cache)
        prefix, _, content = reader.read("output", cursor).content.partition(
            "</header>"
        )
        header = ET.fromstring(prefix + "</header>")
        assert header.findtext("status") == "unavailable"
        assert header.findtext("response-type") == "failed"
        assert header.findtext("tool-call-id") == "call"
        assert header.findtext("started-at") == "then"
        assert int(header.findtext("next-cursor")) == len(original.result)
        assert "not resumed" in content
        assert content.endswith(original.result[cursor:])
        assert cache.get("output") == original


def test_final_cursor_read_includes_new_progress_and_preformatted_result():
    release = Event()
    progress = "first\nsecond\n"
    manager = ToolExecutionManager()
    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(
            lambda: release.wait() and PreformattedToolResult("public final receipt"),
            read_partial=lambda cursor: (progress[cursor:], len(progress)),
        ),
        ("owner",),
        ExecutionOptions(mode="background"),
        output_id="output",
    )
    release.set()
    assert manager.wait("call", 1).is_finished
    reader = ToolOutputReader(unavailable_cache, manager)
    prefix, _, content = reader.read("output", len("first\n")).content.partition(
        "</header>"
    )
    header = ET.fromstring(prefix + "</header>")
    assert header.findtext("response-type") == "final"
    assert int(header.findtext("next-cursor")) == len(progress)
    assert content.strip() == "second\n\npublic final receipt"
    assert manager.next_delivery(("owner",)) is None


@pytest.mark.parametrize("status", ["running", "completed", "failed"])
@pytest.mark.parametrize("cursor", [None, 0])
def test_progress_failure_preserves_current_state_and_final_output(status, cursor):
    release = Event()
    manager = ToolExecutionManager()

    def work():
        release.wait()
        if status == "failed":
            raise RuntimeError("work failed")
        return "final result"

    def read_progress(cursor):
        raise OSError("progress <&> unavailable")

    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(work, read_partial=read_progress),
        ("owner",),
        ExecutionOptions(mode="background"),
        output_id="output",
    )
    try:
        if status != "running":
            release.set()
            assert manager.wait("call", 1).is_finished
        result = ToolOutputReader(unavailable_cache, manager).read("output", cursor)
        prefix, _, body = result.content.partition("</header>\n")
        header = ET.fromstring(prefix + "</header>")
        assert header.findtext("status") == status
        assert header.findtext("response-type") == (
            "final" if status == "completed" else status
        )
        # Final reads without a cursor do not depend on the progress channel.
        expects_diagnostic = cursor is not None or status == "running"
        assert header.findtext("partial-output-error") == (
            "progress <&> unavailable" if expects_diagnostic else None
        )
        assert header.find("next-cursor") is None
        if status == "completed":
            assert body == "final result"
        elif status == "failed":
            assert "RuntimeError: work failed" in body
        else:
            assert "still running" in body
        assert manager.by_output_id("output").status == status
        if status != "running":
            assert manager.next_delivery(("owner",)) is None
    finally:
        release.set()
        assert manager.wait("call", 1).is_finished
