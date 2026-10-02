from __future__ import annotations

import hashlib
from contextlib import ExitStack
from pathlib import Path

import pytest
from dotenv import load_dotenv

from exmekh_lib.core.models.tool import Tool
from tests.live.tool_execution_support import (
    LiveExecutionRig,
    ProcessFixture,
    ProcessTool,
    QuickProcessTool,
    RecordingExecutionProvider,
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


def executed_call(rig: LiveExecutionRig, *, label: str, response_type: str):
    calls = [
        call
        for call in rig.provider.calls("live_process_job")
        if call.id in rig.fixture.jobs
        and call.get_arguments().get("label") == label
        and rig.result(call)[0]["response-type"] == response_type
    ]
    assert calls, f"MODEL: no executed job for {label!r} with {response_type} output"
    return calls[0]


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
        "Finally reply only DONE. Do not use exit_agent at root."
    )
    call = executed_call(rig, label="quick-default", response_type="final")
    header, result = rig.result(call)
    assert header["response-type"] == "final"
    assert result["exit_code"] == 0
    assert rig.durations[call.id] < 5, (
        "CLIENT: short work waited for the entire deadline"
    )
    rig.assert_completed_job(call)
    hash_calls = rig.provider.calls("read_fixture_hash")
    assert hash_calls, "MODEL: fixture hash was not requested"
    hash_call = hash_calls[-1]
    assert split_output(rig.provider.output(hash_call.id))[1] == rig.fixture.sha256
    assert not any(
        read.get_arguments().get("output_id") == header["id"]
        for read in rig.provider.synthetic_reads()
    )
    assert_saved_final(rig, call, header["id"])


@pytest.mark.parametrize("streaming", [False, True], ids=["ordinary", "streaming"])
def test_live_background_wait_and_automatic_delivery(live_rig, streaming):
    rig = live_rig(streaming=streaming)
    rig.node.run(
        "Call live_process_job once with label='background', execution_mode='background', "
        "wait_seconds=60. Then call control_tool_execution with the original tool_call_id, "
        "action='wait', wait_seconds=2. The fixture completes during that wait. "
        "The harness will supply the final read automatically; do not call read_tool_output yourself. "
        "Finally reply only DONE."
    )
    call = executed_call(rig, label="background", response_type="running")
    header, pending = rig.result(call)
    assert header["response-type"] == "running"
    assert pending["status"] == "running"
    assert pending["tool_call_id"] == call.id
    assert pending["output_id"] == header["id"]
    assert rig.durations[call.id] < 1, "CLIENT: background invocation blocked"
    controls = [
        control
        for control in rig.provider.calls("control_tool_execution")
        if control.get_arguments().get("tool_call_id") == call.id
        and rig.result(control)[1].get("status") == "completed"
    ]
    assert controls, "MODEL: no wait completed the original job"
    control = controls[0]
    _, status = rig.result(control)
    assert status["status"] == "completed"
    assert status["tool_call_id"] == call.id != control.id
    assert "result" not in status and "receipt" not in status
    assert status["output_id"] == header["id"]
    assert status["started_at"] == pending["started_at"]
    assert status["elapsed_seconds"] >= pending["elapsed_seconds"]
    assert "finished_at" in status
    assert rig.durations[control.id] < 3
    reads = [
        read
        for read in rig.provider.synthetic_reads()
        if read.get_arguments().get("output_id") == header["id"]
    ]
    assert len(reads) == 1 and reads[0].id != call.id
    assert reads[0].get_arguments()["output_id"] == header["id"]
    assert split_output(rig.provider.output(reads[0].id))[0]["response-type"] == "final"
    rig.assert_completed_job(call)
    assert_saved_final(rig, call, header["id"])


def test_live_expired_waits_and_incremental_output(live_rig):
    rig = live_rig()
    partials = []

    def observe(event):
        if event.is_tool_running:
            return
        if event.tool_name == "control_tool_execution":
            rig.fixture.job(
                executed_call(rig, label="incremental", response_type="running").id
            )
        if event.tool_name == "read_tool_output":
            header, body = split_output(rig.outputs[event.tool_call_id])
            if header["response-type"] == "partial":
                partials.append((header, body))
                job = rig.fixture.job(
                    executed_call(rig, label="incremental", response_type="running").id
                )
                if len(partials) == 1:
                    job.send("progress")
                    assert job.advanced.wait(5), "PROCESS: no new progress"
                elif '"phase": "advanced"' in body:
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
        "5. Reply only DONE."
    )
    call = executed_call(rig, label="incremental", response_type="running")
    controls = [
        control
        for control in rig.provider.calls("control_tool_execution")
        if control.get_arguments().get("tool_call_id") == call.id
        and rig.result(control)[1].get("status") == "running"
    ]
    assert controls, "MODEL: bounded wait while running was not exercised"
    control = controls[0]
    assert rig.result(call)[0]["response-type"] == "running"
    assert rig.result(control)[1]["status"] == "running"
    assert 0.04 <= rig.durations[call.id] < 1
    assert 0.04 <= rig.durations[control.id] < 1
    assert len(partials) >= 2, "MODEL: missing incremental reads"
    first = next((page for page in partials if '"phase": "started"' in page[1]), None)
    advanced = next(
        (page for page in partials if '"phase": "advanced"' in page[1]), None
    )
    assert first is not None and advanced is not None, "MODEL: missing process progress"
    assert '"phase": "started"' not in advanced[1], "CLIENT: cursor repeated old output"
    assert int(advanced[0]["next-cursor"]) > int(first[0]["next-cursor"])
    assert first[0]["id"] == advanced[0]["id"]
    assert len(rig.provider.calls("read_tool_output")) >= 2
    output_id = rig.result(call)[0]["id"]
    assert (
        len(
            [
                read
                for read in rig.provider.synthetic_reads()
                if read.get_arguments().get("output_id") == output_id
            ]
        )
        == 1
    )
    rig.assert_completed_job(call)
