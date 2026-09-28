from __future__ import annotations

import hashlib
from contextlib import ExitStack
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from dotenv import load_dotenv

from qs.core.models.tool import Tool
from qs.interfaces.tool_hook import AfterTool, BeforeTool, ToolArguments, ToolContext
from tests.live.tool_execution_support import (
    FailedProcessTool,
    LiveExecutionRig,
    NegativeProcessTool,
    ProcessFixture,
    ProcessTool,
    QuickProcessTool,
    RecordingExecutionProvider,
    UncancellableProcessTool,
    split_output,
)

pytestmark = pytest.mark.live


@pytest.fixture
def live_rig(tmp_path: Path):
    load_dotenv()
    rigs = []
    with ExitStack() as cleanup:

        def build(
            tool_factory=lambda fixture: [ProcessTool(fixture)],
            *,
            streaming=True,
            children=(),
        ):
            directory = tmp_path / f"run-{len(rigs)}"
            directory.mkdir()
            fixture = ProcessFixture(directory)
            provider = RecordingExecutionProvider(
                directory / "cache", streaming=streaming
            )
            cleanup.callback(provider.client.close)
            cleanup.callback(provider.get_session().close)
            rig = LiveExecutionRig(provider, fixture)
            cleanup.callback(rig.close)
            rigs.append(rig)
            rig.initialize(tool_factory(fixture), children=children)
            return rig

        yield build


def assert_saved_final(rig: LiveExecutionRig, call, output_id: str) -> None:
    record = rig.saved_output(output_id)
    assert record is not None, "CACHE: missing canonical output"
    assert record.tool_call_id == call.id
    assert record.execution["status"] == "completed"
    assert record.result["receipt"] == rig.fixture.job(call.id).receipt["receipt"]
    assert record.args == call.get_arguments(), "CACHE: original call arguments changed"
    assert (
        len([item for item in rig.saved_outputs() if item.tool_call_id == call.id]) == 1
    )


@pytest.mark.parametrize("cache_result", [True, False], ids=["saved", "runtime"])
def test_live_search_tracks_background_work_without_consuming_final_delivery(
    live_rig, cache_result
):
    rig = live_rig(lambda fixture: [ProcessTool(fixture, cache_result=cache_result)])
    searches = []

    def observe(event):
        if not event.is_tool_running and event.tool_name == "search_tool_outputs":
            searches.append(ET.fromstring(rig.outputs[event.tool_call_id]))
            if len(searches) == 1:
                rig.finish_all()

    rig.observer = observe
    rig.node.run(
        "At root call live_process_job once with label='search-background' and "
        "execution_mode='background', wait_seconds=60. Then open work-scope. "
        "Inside the skill call search_tool_outputs with query='search-background' "
        "and tool_name='live_process_job'. The job finishes after that search. "
        "Call search_tool_outputs again with the same filters to see the finished job. "
        "Search returns previews, so then exit_agent(text='search complete', items=[]). "
        "Back at root use the automatic final output delivery and summarize the outcome."
    )
    if len(searches) < 2:
        pytest.skip(
            "MODEL: searches before and after completion were not both exercised"
        )
    call = rig.one_call("live_process_job")
    origin = next(
        event
        for event in rig.events
        if event.tool_call_id == call.id and event.is_tool_running
    )
    search_events = [
        event
        for event in rig.events
        if event.tool_name == "search_tool_outputs" and not event.is_tool_running
    ]
    scoped_searches = [
        search
        for search, event in zip(searches, search_events, strict=True)
        if event.skill_path == "/work-scope"
    ]
    if origin.skill_path != "/" or len(scoped_searches) < 2:
        pytest.skip(
            "MODEL: root work and searches inside the child skill were not exercised"
        )
    searches = scoped_searches[:2]
    output_id = rig.result(call)[0]["id"]
    for search in searches:
        records = [
            record
            for record in search.findall("./records/record")
            if record.findtext("id") == output_id
        ]
        assert len(records) == 1, (
            "CLIENT: background output missing or duplicated in search"
        )
        record = records[0]
        assert record.findtext("tool-call-id") == call.id
        assert record.findtext("storage") == ("saved" if cache_result else "runtime")
        assert record.findtext("skill-path") == "/"
    assert searches[0].findtext("./records/record/response-type") in {
        "running",
        "partial",
    }
    assert searches[-1].findtext("./records/record/response-type") == "final"
    assert len(rig.provider.synthetic_reads()) == 1, (
        "CLIENT: search consumed or duplicated final delivery"
    )
    synthetic_ids = {read.id for read in rig.provider.synthetic_reads()}
    assert all(
        event.skill_path == "/"
        for event in rig.events
        if event.tool_call_id in synthetic_ids
    ), "CLIENT: final search result was automatically delivered in a foreign scope"
    rig.assert_completed_job(call)
    if cache_result:
        assert_saved_final(rig, call, output_id)
    else:
        assert rig.saved_output(output_id) is None, (
            "CACHE: search persisted an uncached result"
        )


@pytest.mark.parametrize("streaming", [False, True], ids=["ordinary", "streaming"])
def test_live_default_wait_returns_real_file_result(live_rig, streaming):
    def tools(fixture):
        def read_fixture_hash() -> str:
            return hashlib.sha256(
                (fixture.directory / "fixture-input.txt").read_bytes()
            ).hexdigest()

        return [QuickProcessTool(fixture), Tool(read_fixture_hash)]

    rig = live_rig(tools, streaming=streaming)
    rig.node.run(
        "Call live_process_job exactly once with label='quick-default'. "
        "Use default execution settings (null execution_mode and null wait_seconds if required). "
        "Then call read_fixture_hash once with default settings. "
        "Finally summarize the outcome in your own words. Do not use exit_agent at root."
    )
    call = rig.one_call("live_process_job")
    header, result = rig.result(call)
    assert header["response-type"] == "final"
    assert result["exit_code"] == 0
    assert rig.durations[call.id] < 5, (
        "CLIENT: short work waited for the entire deadline"
    )
    rig.assert_completed_job(call)
    hash_call = rig.one_call("read_fixture_hash")
    assert split_output(rig.provider.output(hash_call.id))[1] == rig.fixture.sha256
    assert not rig.provider.synthetic_reads()
    assert_saved_final(rig, call, header["id"])


@pytest.mark.parametrize("streaming", [False, True], ids=["ordinary", "streaming"])
def test_live_background_wait_and_automatic_delivery(live_rig, streaming):
    rig = live_rig(streaming=streaming)
    rig.node.run(
        "Call live_process_job once with label='background', execution_mode='background', "
        "wait_seconds=60. Then call control_tool_execution with the original tool_call_id, "
        "action='wait', wait_seconds=2. The fixture completes during that wait. "
        "The harness will supply the final read automatically; do not call read_tool_output yourself. "
        "Finally summarize the completed work in your own words."
    )
    call = rig.one_call("live_process_job")
    header, pending = rig.result(call)
    assert header["response-type"] == "running"
    assert pending["status"] == "running"
    assert pending["tool_call_id"] == call.id
    assert pending["output_id"] == header["id"]
    assert rig.durations[call.id] < 1, "CLIENT: background invocation blocked"
    control = rig.one_call("control_tool_execution")
    _, status = rig.result(control)
    assert status["status"] == "completed"
    assert status["tool_call_id"] == call.id != control.id
    assert "result" not in status and "receipt" not in status
    assert status["output_id"] == header["id"]
    assert status["started_at"] == pending["started_at"]
    assert status["elapsed_seconds"] >= pending["elapsed_seconds"]
    assert "finished_at" in status
    assert 0.15 <= rig.durations[control.id] < 3
    reads = rig.provider.synthetic_reads()
    assert len(reads) == 1 and reads[0].id != call.id
    assert reads[0].get_arguments()["output_id"] == header["id"]
    assert split_output(rig.provider.output(reads[0].id))[0]["response-type"] == "final"
    rig.assert_completed_job(call)
    assert_saved_final(rig, call, header["id"])


def test_live_two_background_tools_keep_independent_output_ids(live_rig):
    rig = live_rig(
        lambda fixture: [
            ProcessTool(fixture, name="first_job"),
            ProcessTool(fixture, name="second_job"),
        ]
    )
    overlapping = []

    def observe(event):
        if event.is_tool_running and event.tool_name == "control_tool_execution":
            if len(rig.fixture.jobs) == 2:
                overlapping.append(
                    all(job.process.poll() is None for job in rig.fixture.jobs.values())
                )

    rig.observer = observe
    rig.node.run(
        "Start first_job and second_job once each with distinct labels, both execution_mode='background' "
        "and wait_seconds=60. Start both before waiting for either. "
        "Then call control_tool_execution for each original tool_call_id with action='wait', "
        "wait_seconds=2. Let the harness deliver both final results. Summarize the outcome."
    )
    first, second = rig.one_call("first_job"), rig.one_call("second_job")
    if not any(overlapping):
        pytest.skip("MODEL: concurrent work case was not exercised")
    output_ids = {rig.result(call)[0]["id"] for call in (first, second)}
    assert len(output_ids) == 2, "CLIENT: different tools share one output ID"
    reads = rig.provider.synthetic_reads()
    assert len(reads) == 2
    assert {call.get_arguments()["output_id"] for call in reads} == output_ids
    for call in (first, second):
        rig.assert_completed_job(call)
        assert_saved_final(rig, call, rig.result(call)[0]["id"])


def test_live_sub_agent_waits_in_its_own_execution_context(live_rig):
    child = live_rig()
    parent = live_rig(lambda fixture: [], children=(child.node,))
    parent.node.run(
        "Use call_sub_agent once with sub_agent='main' and text requesting this workflow: "
        "start live_process_job once with label='delegated', execution_mode='background', wait_seconds=60; "
        "call control_tool_execution for its original tool_call_id with action='wait', wait_seconds=2; "
        "let the harness deliver the final result, then summarize it. After delegation, reply briefly."
    )
    delegated = parent.one_call("call_sub_agent")
    child_call = child.one_call("live_process_job")
    assert not {"execution_mode", "wait_seconds"} & delegated.get_arguments().keys()
    assert parent.node.agent_context_id != child.node.agent_context_id
    assert not parent.provider.synthetic_reads(), (
        "CLIENT: child completion was injected into parent history"
    )
    assert len(child.provider.synthetic_reads()) == 1
    child.assert_completed_job(child_call)
    output_id = child.result(child_call)[0]["id"]
    with pytest.raises(ValueError, match="Unknown tool call ID"):
        parent.node.control_tool_execution(child_call.id, wait_seconds=0.01)
    assert all(
        record.tool_call_id != child_call.id for record in parent.saved_outputs()
    ), "CACHE: child process output was stored in the parent node"
    # Independent tokenizer sessions may reuse an output ID; reads stay node-scoped.
    if (record := parent.saved_output(output_id)) is not None:
        assert record.tool_call_id == delegated.id
    assert (
        parent.durations[delegated.id]
        >= child.durations[child.one_call("control_tool_execution").id]
    )


def test_live_default_minute_yields_without_killing_work(live_rig):
    rig = live_rig()
    alive_after_deadline = []

    def observe(event):
        if event.tool_name == "live_process_job" and not event.is_tool_running:
            job = rig.fixture.job(event.tool_call_id)
            alive_after_deadline.append(job.process.poll() is None)

    rig.observer = observe
    rig.node.run(
        "Call live_process_job once with label='minute-default', execution_mode=null and "
        "wait_seconds=null to exercise the default one-minute wait. It will remain running. "
        "Then call control_tool_execution with its original tool_call_id, action='wait', "
        "wait_seconds=2. Wait for the automatic final read and summarize the outcome."
    )
    call = rig.one_call("live_process_job")
    if (
        call.get_arguments().get("wait_seconds", 60) != 60
        or call.get_arguments().get("execution_mode", "wait") != "wait"
    ):
        pytest.skip("MODEL: default one-minute wait was not exercised")
    assert 59 <= rig.durations[call.id] < 65, (
        "CLIENT: default deadline is not one minute"
    )
    assert alive_after_deadline == [True], "PROCESS: initial wait terminated the child"
    assert rig.result(call)[0]["response-type"] == "running"
    rig.assert_completed_job(call)


def test_live_expired_waits_and_incremental_output(live_rig):
    rig = live_rig()
    partials = []

    def observe(event):
        if event.is_tool_running:
            return
        if event.tool_name == "control_tool_execution":
            rig.fixture.job(rig.one_call("live_process_job").id)
        if event.tool_name == "read_tool_output":
            header, body = split_output(rig.outputs[event.tool_call_id])
            if header["response-type"] == "partial":
                partials.append((header, body))
                job = rig.fixture.job(rig.one_call("live_process_job").id)
                if len(partials) == 1:
                    job.send("progress")
                    assert job.advanced.wait(5), "PROCESS: no new progress"
                elif len(partials) == 2:
                    rig.finish_all()

    rig.observer = observe
    rig.node.run(
        "Execute these steps in separate model responses: "
        "1. Call live_process_job once with label='incremental', execution_mode='wait', wait_seconds=0.05. "
        "2. Call control_tool_execution with its original tool_call_id, action='wait', wait_seconds=0.05. "
        "The job will still be running; do not cancel or relaunch it. "
        "3. Call read_tool_output with its XML output ID and cursor=0. "
        "4. Call read_tool_output again with that same output ID and the returned next-cursor. "
        "The fixture then completes and an automatic final read arrives. "
        "5. Summarize the outcome in your own words."
    )
    call = rig.one_call("live_process_job")
    control = rig.one_call("control_tool_execution")
    assert rig.result(call)[0]["response-type"] == "running"
    assert rig.result(control)[1]["status"] == "running"
    assert 0.04 <= rig.durations[call.id] < 1
    assert 0.04 <= rig.durations[control.id] < 1
    assert len(partials) == 2, "MODEL: missing incremental reads"
    assert '"phase": "started"' in partials[0][1]
    assert '"phase": "advanced"' in partials[1][1]
    assert '"phase": "started"' not in partials[1][1], (
        "CLIENT: cursor repeated old output"
    )
    assert int(partials[1][0]["next-cursor"]) > int(partials[0][0]["next-cursor"])
    assert partials[0][0]["id"] == partials[1][0]["id"]
    assert len(rig.provider.calls("read_tool_output")) >= 2
    assert len(rig.provider.synthetic_reads()) == 1
    rig.assert_completed_job(call)


def test_live_real_cancellation_and_unsupported_cancellation(live_rig):
    rig = live_rig(
        lambda fixture: [
            ProcessTool(fixture),
            UncancellableProcessTool(fixture, name="uncancellable_job"),
        ]
    )
    rig.node.run(
        "Call live_process_job once with label='cancel-me', execution_mode='background', wait_seconds=60. "
        "Call control_tool_execution for that original tool_call_id with action='cancel', wait_seconds=1. "
        "Then call uncancellable_job once with label='keep-running', execution_mode='background', wait_seconds=60. "
        "Call control_tool_execution for its original tool_call_id with action='cancel', wait_seconds=1. "
        "Cancellation is unsupported for this second job. Call control_tool_execution for the second "
        "job with action='wait', wait_seconds=2 to finish it. Reply briefly after both terminal results."
    )
    first = rig.one_call("live_process_job")
    second = rig.one_call("uncancellable_job")
    controls = rig.provider.calls("control_tool_execution")
    assert len(controls) == 3, "MODEL: expected cancel, unsupported cancel, wait"
    cancelled, unsupported, completed = [rig.result(call)[1] for call in controls]
    assert cancelled["cancellation_supported"] is True
    assert cancelled["cancelled"] is True
    assert cancelled["status"] == "cancelled", (
        "CLIENT: confirmed cancellation lost to worker completion"
    )
    assert rig.fixture.job(first.id).process.poll() is not None
    assert not rig.fixture.job(first.id).receipt
    assert unsupported["cancellation_supported"] is False
    assert unsupported["cancelled"] is False
    assert unsupported["status"] == "running"
    assert completed["status"] == "completed"
    reads = [
        split_output(rig.provider.output(call.id))[0]
        for call in rig.provider.synthetic_reads()
    ]
    assert [header["response-type"] for header in reads] == ["cancelled", "final"]
    assert rig.fixture.job(second.id).receipt


def test_live_execution_failure_differs_from_negative_tool_result(live_rig):
    rig = live_rig(
        lambda fixture: [
            FailedProcessTool(fixture, name="failed_job"),
            NegativeProcessTool(fixture, name="negative_job"),
        ]
    )
    rig.node.run(
        "Call failed_job once with label='failure', execution_mode='wait', wait_seconds=5. "
        "Then call negative_job once with label='negative', execution_mode='wait', wait_seconds=5. "
        "Both real children exit with code 7, but only the first raises an execution exception. "
        "Read each XML output ID once with read_tool_output(cursor=null), then summarize their statuses."
    )
    failed = rig.one_call("failed_job")
    negative = rig.one_call("negative_job")
    assert rig.result(failed)[0]["response-type"] == "failed"
    assert "RuntimeError" in rig.result(failed)[1]["error"]
    assert rig.result(negative)[0]["response-type"] == "final"
    assert rig.result(negative)[1]["exit_code"] == 7
    read_ids = {
        call.get_arguments()["output_id"]
        for call in rig.provider.calls("read_tool_output")
    }
    assert {rig.result(failed)[0]["id"], rig.result(negative)[0]["id"]} <= read_ids
    assert not rig.provider.synthetic_reads()
    for call in (failed, negative):
        assert rig.fixture.job(call.id).process.poll() == 7


def test_live_manual_final_read_in_same_batch_suppresses_auto_delivery(live_rig):
    rig = live_rig()
    rig.node.run(
        "Call live_process_job once with label='manual-final', execution_mode='background', wait_seconds=60. "
        "In your NEXT SINGLE RESPONSE emit TWO tool calls in this exact order: "
        "control_tool_execution(original tool_call_id, action='wait', wait_seconds=2), "
        "then read_tool_output(XML output ID, cursor=null). The wait completes the fixture. "
        "Then summarize the completed work in your own words."
    )
    batches = [
        [call.get_name() for call in item.get_tool_calls() or ()]
        for item in rig.provider.responses
    ]
    rig.assert_completed_job(rig.one_call("live_process_job"))
    if not any(
        "control_tool_execution" in batch
        and "read_tool_output" in batch
        and batch.index("control_tool_execution") < batch.index("read_tool_output")
        for batch in batches
    ):
        pytest.skip(
            f"MODEL: same-response wait/read case not exercised; batches={batches}"
        )
    assert rig.provider.calls("read_tool_output")
    assert not rig.provider.synthetic_reads(), (
        "CLIENT: manually consumed final output delivered twice"
    )


@pytest.mark.parametrize("scope_change", ["nested", "reentry"])
def test_live_skill_instance_controls_automatic_delivery(live_rig, scope_change):
    rig = live_rig()
    openings = []

    def observe(event):
        if not event.is_tool_running and event.tool_name in {
            "work-scope",
            "inner-scope",
        }:
            openings.append(event.tool_name)
            if event.tool_name == "inner-scope" or openings.count("work-scope") == 2:
                rig.finish_all()

    rig.observer = observe
    if scope_change == "nested":
        response = rig.node.run(
            "Open work-scope. Inside it call live_process_job once with label='scope', "
            "execution_mode='background', wait_seconds=60. "
            "Then open inner-scope. The process completes while you are nested. "
            "Inside inner-scope call exit_agent(text='inner done', items=[]). "
            "Back in the original work-scope use the automatic final read, "
            "then exit_agent with a summary in text and items=[]. "
            "Finally at root summarize the outcome in your own words."
        )
    else:
        rig.node.run(
            "Open work-scope. Inside it call live_process_job once with label='scope', "
            "execution_mode='background', wait_seconds=60. "
            "Then exit_agent(text='work pending', items=[]), leaving the job in background. "
            "At root briefly acknowledge that work continues. Do not wait or read the job."
        )
        output_id = rig.result(rig.one_call("live_process_job"))[0]["id"]
        response = rig.node.run(
            "Open work-scope again. The previous job will finish as you enter this new instance. "
            f"Inside the skill read the previous output with read_tool_output(output_id='{output_id}', "
            "cursor=null). Then exit_agent with a summary in text and items=[]. "
            "At root briefly summarize the outcome."
        )
    call = rig.one_call("live_process_job")
    origin = next(
        event
        for event in rig.events
        if event.tool_call_id == call.id and event.is_tool_running
    )
    if origin.skill_path != "/work-scope":
        pytest.skip("MODEL: job was not started in the requested owner skill")
    synthetic_ids = {read.id for read in rig.provider.synthetic_reads()}
    auto_events = [
        event
        for event in rig.events
        if event.tool_call_id in synthetic_ids and not event.is_tool_running
    ]
    assert all(event.skill_path == "/work-scope" for event in auto_events), (
        "CLIENT: final delivered inside nested scope"
    )
    if scope_change == "nested":
        returned_from_inner = any(
            event.tool_name == "exit_agent"
            and not event.is_tool_running
            and event.skill_path == "/work-scope"
            for event in rig.events
        )
        if "inner-scope" not in openings or not returned_from_inner:
            pytest.skip("MODEL: nested skill return was not exercised")
        assert len(auto_events) == 1, (
            "CLIENT: expected one automatic final delivery after nested return"
        )
    else:
        if openings.count("work-scope") < 2:
            pytest.skip("MODEL: exit and skill reentry were not exercised")
        assert not auto_events, (
            "CLIENT: old output automatically delivered to a new skill instance"
        )
        output_id = rig.result(call)[0]["id"]
        if not any(
            read.get_arguments()["output_id"] == output_id
            for read in rig.provider.calls("read_tool_output")
        ):
            pytest.skip("MODEL: manual read of the old job was not exercised")
        assert len(response.skill_executions) >= 2
    rig.assert_completed_job(call)


def test_live_hooks_filter_progress_and_run_once(live_rig):
    contexts = []
    results = []

    class NormalizeLabel(BeforeTool):
        def handle(self, context: ToolContext):
            contexts.append(context)
            return ToolArguments({"label": context.arguments["label"].lower()})

    class FilterResult(AfterTool):
        def handle(self, context: ToolContext, result):
            results.append(result)
            return {key: value for key, value in result.items() if key != "raw"}

    rig = live_rig(
        lambda fixture: [
            ProcessTool(fixture, before=NormalizeLabel(), after=FilterResult())
        ]
    )
    rig.node.run(
        "Call live_process_job once with label='HOOKS', execution_mode='background', wait_seconds=60. "
        "Read its XML output ID with read_tool_output(cursor=null) while running. "
        "Then call control_tool_execution(original tool_call_id, action='wait', wait_seconds=2). "
        "Use the automatic final read and summarize the outcome in your own words."
    )
    call = rig.one_call("live_process_job")
    assert len(contexts) == len(results) == 1, (
        "CLIENT: hooks repeated during control or output reads"
    )
    assert contexts[0].arguments == {"label": call.get_arguments()["label"]}
    assert results[0]["label"] == contexts[0].arguments["label"].lower()
    assert not {"execution_mode", "wait_seconds"} & contexts[0].arguments.keys()
    read = rig.one_call("read_tool_output")
    assert split_output(rig.provider.output(read.id))[0]["response-type"] == "running"
    visible = [
        message.get_message()
        for request in rig.provider.requests
        for message in request.get_messages()
    ]
    assert all(rig.fixture.raw not in (message or "") for message in visible), (
        "CLIENT: raw progress bypassed AfterTool filtering"
    )
    rig.assert_completed_job(call)
    record = rig.saved_output(rig.result(call)[0]["id"])
    assert "raw" not in record.result


def test_live_uncached_output_survives_to_next_persistent_turn(live_rig):
    rig = live_rig(lambda fixture: [ProcessTool(fixture, cache_result=False)])
    first = rig.node.run(
        "Call live_process_job once with label='next-turn', execution_mode='background', wait_seconds=60. "
        "Do not wait or read it. Briefly acknowledge that work continues."
    )
    call = rig.one_call("live_process_job")
    assert first.get_message()
    assert not rig.provider.synthetic_reads()
    rig.finish_all()
    rig.node.run(
        "Use the completed result provided by the harness and summarize the outcome in your own words."
    )
    rig.assert_completed_job(call)
    output_id = rig.result(call)[0]["id"]
    assert len(rig.provider.synthetic_reads()) == 1
    assert rig.saved_output(output_id) is None
    assert not rig.saved_outputs(), "CACHE: transient output persisted"
