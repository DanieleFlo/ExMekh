from contextlib import contextmanager
from threading import Event
from xml.etree import ElementTree as ET

import pytest

from qs.core.cache import SessionCache
from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.models.tool import (
    ExecutionOptions,
    PreformattedToolResult,
    Tool,
    ToolExecutionOperation,
)
from qs.core.tool_calls.execution_history import ExecutionHistory
from qs.core.tool_calls.execution_manager import ToolExecutionManager
from qs.core.tool_calls.memory_tools import HarnessMemoryTools
from qs.interfaces.tool_hook import AfterTool


@pytest.fixture
def output_rig(tmp_path):
    with SessionCache(tmp_path) as session:
        cache = session.node("owner")
        manager = ToolExecutionManager()
        memory = HarnessMemoryTools(ExecutionHistory(), lambda: cache, manager)
        yield cache, manager, memory, session


def search_xml(memory, **arguments):
    return ET.fromstring(memory.search_tool_outputs(**arguments).content)


@contextmanager
def pending_job(
    manager, call_id="call", *, result="finished receipt", partial=None, skill_path="/"
):
    release = Event()

    def run():
        release.wait()
        if isinstance(result, Exception):
            raise result
        return result

    def cancel():
        release.set()
        return True

    manager.start(
        call_id,
        "work",
        {"label": "needle"},
        ToolExecutionOperation(run, cancel, partial),
        ("owner",),
        ExecutionOptions(mode="background"),
        output_id=f"output-{call_id}",
        skill_path=skill_path,
    )
    try:
        yield release
    finally:
        release.set()
        assert manager.wait(call_id, 1).is_finished


def test_search_discovers_uncached_background_work_and_its_origin(output_rig):
    cache, manager, memory, _ = output_rig
    with pending_job(manager, skill_path="/analysis"):
        root = search_xml(memory, tool_name="work", skill_path="/analysis")
        assert root.tag == "tool-outputs"
        assert root.findtext("total") == "1"
        record = root.find("./records/record")
        assert record.findtext("id") == "output-call"
        assert record.findtext("tool-call-id") == "call"
        assert record.findtext("type") == "Tool"
        assert record.findtext("response-type") == "running"
        assert record.findtext("status") == "running"
        assert record.findtext("storage") == "runtime"
        assert record.findtext("started-at")
        assert float(record.findtext("elapsed-seconds")) >= 0
        assert record.find("finished-at") is None
        assert (
            search_xml(memory, skill_path="/analysis/nested").findtext("total") == "0"
        )
        assert cache.get_all() == []


def test_search_lists_an_execution_only_after_its_output_id_is_published(output_rig):
    _, manager, memory, _ = output_rig
    release = Event()
    manager.start(
        "unpublished",
        "work",
        {},
        ToolExecutionOperation(release.wait),
        ("owner",),
        ExecutionOptions(mode="background"),
    )
    try:
        assert search_xml(memory).findtext("total") == "0"
        manager.attach_output("unpublished", "published", None)
        assert search_xml(memory).findtext("./records/record/id") == "published"
    finally:
        release.set()
        assert manager.wait("unpublished", 1).is_finished


@pytest.mark.parametrize("query", ["WORK", "CALL", "OUTPUT-CALL", "NEEDLE", "PROGRESS"])
def test_search_matches_runtime_metadata_and_available_partial_output(
    output_rig, query
):
    _, manager, memory, _ = output_rig
    with pending_job(manager, partial=lambda cursor: ("filtered progress", 17)):
        record = search_xml(memory, query=query).find("./records/record")
        assert record is not None
        assert record.findtext("response-type") == "partial"
        assert record.findtext("status") == "running"
        assert record.findtext("preview") == "filtered progress"


def test_runtime_final_replaces_cached_running_view_without_duplicates_or_acknowledgement(
    output_rig,
):
    cache, manager, memory, _ = output_rig
    with pending_job(manager) as release:
        cache.store(
            ToolCallRecord(
                "work",
                "output-call",
                {"label": "needle"},
                "old running preview",
                tool_call_id="call",
                execution=manager.by_output_id("output-call").status_info(),
            )
        )
        release.set()
        assert manager.wait("call", 1).is_finished
        root = search_xml(memory)
        assert root.findtext("total") == "1"
        record = root.find("./records/record")
        assert record.findtext("response-type") == "final"
        assert record.findtext("status") == "completed"
        assert record.findtext("storage") == "saved"
        assert record.findtext("preview") == "finished receipt"
        assert record.findtext("finished-at")
        assert manager.next_delivery(("owner",)) is not None
        assert cache.get("output-call").result == "old running preview"


def test_runtime_output_records_preserve_saved_annotations_without_mutation(output_rig):
    _, manager, _, _ = output_rig
    stored = ToolCallRecord(
        "work", "output-call", {}, "old preview", thinking="saved annotation"
    )
    records = {stored.id: stored}
    with pending_job(manager) as release:
        release.set()
        assert manager.wait("call", 1).is_finished

        record = manager.output_records(records)[0]

        assert record.result == "finished receipt"
        assert record.thinking == "saved annotation"
        assert record.execution["storage"] == "saved"
        assert stored.result == "old preview"
        assert records == {stored.id: stored}
        assert manager.next_delivery(("owner",)) is not None


@pytest.mark.parametrize(
    ("result", "status", "response_type", "preview"),
    [
        ({"exit_code": 7}, "completed", "final", "7"),
        (
            PreformattedToolResult("public receipt"),
            "completed",
            "final",
            "public receipt",
        ),
        (RuntimeError("worker failed"), "failed", "failed", "worker failed"),
    ],
)
def test_search_distinguishes_valid_negative_results_from_failed_work(
    output_rig, result, status, response_type, preview
):
    _, manager, memory, _ = output_rig
    with pending_job(manager, result=result) as release:
        release.set()
        assert manager.wait("call", 1).is_finished
        record = search_xml(memory).find("./records/record")
        assert record.findtext("status") == status
        assert record.findtext("response-type") == response_type
        assert preview in record.findtext("preview")


def test_search_shows_confirmed_cancellation(output_rig):
    _, manager, memory, _ = output_rig
    with pending_job(manager):
        assert manager.cancel("call")[1]
        record = search_xml(memory).find("./records/record")
        assert record.findtext("status") == "cancelled"
        assert record.findtext("response-type") == "cancelled"
        assert "cancelled" in record.findtext("preview")


def test_search_paginates_combined_sources_and_keeps_node_isolation(output_rig):
    cache, manager, memory, session = output_rig
    cache.store(ToolCallRecord("work", "saved", {}, "needle saved"))
    session.node("other").store(ToolCallRecord("work", "foreign", {}, "needle private"))
    with pending_job(manager, "first"), pending_job(manager, "second"):
        pages = [
            search_xml(memory, query="needle", offset=index, limit=1)
            for index in range(3)
        ]
        assert all(page.findtext("total") == "3" for page in pages)
        assert [page.findtext("./records/record/id") for page in pages] == [
            "saved",
            "output-first",
            "output-second",
        ]
        assert [page.findtext("next-offset") for page in pages] == ["1", "2", None]
        assert search_xml(memory, query="private").findtext("total") == "0"
        isolated = HarnessMemoryTools(
            ExecutionHistory(), lambda: session.node("other"), ToolExecutionManager()
        )
        assert search_xml(isolated, query="output-first").findtext("total") == "0"


def test_old_running_record_is_unavailable_in_both_search_and_read(output_rig):
    cache, _, memory, _ = output_rig
    cache.store(
        ToolCallRecord(
            "work", "old", {}, "old progress", execution={"status": "running"}
        )
    )
    record = search_xml(memory).find("./records/record")
    assert record.findtext("status") == "unavailable"
    assert record.findtext("response-type") == "failed"
    assert "not resumed" in record.findtext("preview")
    header = (
        memory.read_tool_output("old").content.split("</header>", 1)[0] + "</header>"
    )
    assert ET.fromstring(header).findtext("status") == record.findtext("status")
    assert ET.fromstring(header).findtext("response-type") == record.findtext(
        "response-type"
    )


@pytest.mark.parametrize(
    ("status", "response_type"),
    [("completed", "final"), ("failed", "failed"), ("cancelled", "cancelled")],
)
def test_saved_terminal_records_keep_their_state_in_search_and_read(
    output_rig, status, response_type
):
    cache, _, memory, _ = output_rig
    cache.store(
        ToolCallRecord(
            "work", "saved", {}, "terminal output", execution={"status": status}
        )
    )
    record = search_xml(memory).find("./records/record")
    assert record.findtext("response-type") == response_type
    assert record.findtext("status") == status
    assert record.findtext("storage") == "saved"
    assert record.findtext("preview") == "terminal output"
    header = (
        memory.read_tool_output("saved").content.split("</header>", 1)[0] + "</header>"
    )
    assert ET.fromstring(header).findtext("response-type") == response_type


@pytest.mark.parametrize("error", [{"reason": "worker failed"}, ["worker failed"]])
def test_search_accepts_structured_diagnostics_in_saved_failed_outputs(
    output_rig, error
):
    cache, _, memory, _ = output_rig
    cache.store(
        ToolCallRecord(
            "work",
            "saved",
            {},
            {"error": error},
            execution={"status": "failed"},
        )
    )
    record = search_xml(memory, query="worker failed").find("./records/record")
    assert record.findtext("status") == "failed"
    assert record.findtext("response-type") == "failed"
    assert "worker failed" in record.findtext("preview")


def test_search_keeps_legacy_skill_outputs_final_with_their_text_preview(output_rig):
    cache, _, memory, _ = output_rig
    cache.store(
        ToolCallRecord(
            "analysis",
            "skill",
            {},
            {"text": "skill summary", "items": []},
            source="DeepSkill",
        )
    )
    record = search_xml(memory).find("./records/record")
    assert record.findtext("type") == "DeepSkill"
    assert record.findtext("response-type") == "final"
    assert record.findtext("status") == "completed"
    assert record.findtext("preview") == "skill summary"


def test_partial_read_error_does_not_hide_other_outputs_or_fail_the_work(output_rig):
    cache, manager, memory, _ = output_rig
    cache.store(ToolCallRecord("saved", "saved", {}, "valid result"))

    def unavailable_partial(cursor):
        raise OSError("progress is unavailable")

    with pending_job(manager, partial=unavailable_partial):
        root = search_xml(memory)
        assert root.findtext("total") == "2"
        record = root.findall("./records/record")[1]
        assert record.findtext("status") == "running"
        assert record.findtext("response-type") == "running"
        assert "progress is unavailable" in record.findtext("preview")


def test_after_hook_filters_searchable_results_and_withholds_raw_progress(
    output_rig, tmp_path
):
    cache, manager, memory, _ = output_rig
    entered, release = Event(), Event()
    hook_calls = []

    class OutputFilter(AfterTool):
        def handle(self, context, result):
            hook_calls.append(result)
            return "public receipt"

    class FilteredWork(Tool):
        def create_execution(self, arguments, **context):
            def run():
                entered.set()
                release.wait()
                return "private receipt"

            return ToolExecutionOperation(
                run, read_partial=lambda cursor: ("private progress", 16)
            )

    tool = FilteredWork(lambda: None, name="filtered", after=OutputFilter())
    manager.start(
        "filtered-call",
        tool.name,
        {},
        tool.create_execution_with_hooks(
            {}, call_id="filtered-call", work_root=tmp_path, history=()
        ),
        ("owner",),
        ExecutionOptions(mode="background"),
        output_id="filtered-output",
    )
    try:
        assert entered.wait(1)
        assert search_xml(memory, query="private").findtext("total") == "0"
        assert (
            search_xml(memory).findtext("./records/record/response-type") == "running"
        )
        release.set()
        assert manager.wait("filtered-call", 1).is_finished
        assert search_xml(memory, query="private").findtext("total") == "0"
        assert search_xml(memory, query="public receipt").findtext("total") == "1"
        assert len(hook_calls) == 1
        assert cache.get_all() == []
    finally:
        release.set()
        assert manager.wait("filtered-call", 1).is_finished
