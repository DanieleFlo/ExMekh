from datetime import datetime, timezone
from threading import Event
from unittest.mock import MagicMock

import pytest
from PIL import Image

from exmekh_lib.core.agents.node.tool_result_recorder import ToolResultRecorder
from exmekh_lib.core.agents.node.types import CacheIdentity, ToolExecutionResult
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.cache.session_cache import SessionCache
from exmekh_lib.core.models.tool import (
    ExecutionOptions,
    PreformattedToolResult,
    ToolExecutionOperation,
)
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import (
    ImagesUserMessage,
    TextsUserMessage,
    UserMessage,
)
from exmekh_lib.core.tool_calls.execution_history import ExecutionHistory
from exmekh_lib.core.tool_calls.execution_manager import (
    ExecutionSnapshot,
    ToolExecutionManager,
)
from exmekh_lib.core.tool_calls.harness_memory_tools import HarnessMemoryTools


def test_managed_result_without_output_id_fails_before_persistence():
    cache = MagicMock(spec=NodeCache)
    manager = MagicMock(spec=ToolExecutionManager)
    recorder = ToolResultRecorder(ExecutionHistory(), lambda: cache, manager)
    result = ToolExecutionResult(
        response=ToolMessage("done", "call", "work"),
        save_result=True,
        execution=ExecutionSnapshot(
            call_id="call",
            name="work",
            arguments={},
            status="completed",
            started_at=datetime.now(timezone.utc),
            finished_at=None,
            output_id=None,
            elapsed_seconds=0,
        ),
    )
    identity = CacheIdentity("work", "Tool", "call", {}, 1)

    with pytest.raises(RuntimeError, match="missing its output ID"):
        recorder.persist_result(result, identity, "/")

    cache.store.assert_not_called()
    manager.attach_output.assert_not_called()


def test_saved_steering_is_listed_searched_and_read_from_harness_memory(tmp_path):
    with SessionCache(tmp_path) as session:
        cache = session.node("node")
        history = ExecutionHistory()
        recorder = ToolResultRecorder(history, lambda: cache, ToolExecutionManager())
        recorder.persist_steering(
            UserMessage(TextsUserMessage(["first", "second"])), "guidance", "/skill"
        )
        memory = HarnessMemoryTools(history, lambda: cache)
        assert "<name>steering_message</name>" in memory.get_execution_history().content
        assert (
            "<saved-output-id>guidance</saved-output-id>"
            in memory.get_execution_history().content
        )
        found = memory.search_tool_outputs(query="second", skill_path="/skill").content
        assert "<id>guidance</id>" in found
        assert "<type>UserMessage</type>" in found
        read = memory.read_tool_output("guidance")
        assert "first\nsecond" in read.content
        assert read.attachments is None


@pytest.mark.parametrize("mode", ["direct", "wait", "background"])
@pytest.mark.parametrize("multimodal", [False, True])
def test_tool_attachments_survive_runtime_reads_and_cached_reads(
    tmp_path, mode, multimodal
):
    release = Event()
    companion = UserMessage(
        ["tool context", ImagesUserMessage([Image.new("RGB", (2, 2), "red")])]
        if multimodal
        else "tool context",
        name="work",
    )
    manager = ToolExecutionManager()
    history = ExecutionHistory()
    entry = history.record("work", "call")
    identity = CacheIdentity("work", "Tool", "call", {}, entry.position)
    with SessionCache(tmp_path) as session:
        cache = session.node("node")
        recorder = ToolResultRecorder(history, lambda: cache, manager)
        snapshot = None
        initial = PreformattedToolResult("result", attachments=companion)
        try:
            if mode != "direct":
                snapshot = manager.start(
                    "call",
                    "work",
                    {},
                    ToolExecutionOperation(
                        lambda: (
                            release.wait(3) and initial
                            if mode == "background"
                            else initial
                        )
                    ),
                    ("owner",),
                    ExecutionOptions(mode=mode),
                    output_id="output",
                )
                assert snapshot.status == (
                    "running" if mode == "background" else "completed"
                )
            recorder.persist_result(
                ToolExecutionResult(
                    response=ToolMessage(
                        "pending" if mode == "background" else "result",
                        "call",
                        "work",
                        attachments=None if mode == "background" else companion,
                    ),
                    raw_result=snapshot.record_result() if snapshot else "result",
                    output_id="output",
                    save_result=True,
                    execution=snapshot,
                ),
                identity,
                "/",
            )
            if mode == "background":
                release.set()
                assert manager.wait("call", 1).status == "completed"
                recorder.sync_completed()

            memory = HarnessMemoryTools(history, lambda: cache, manager)
            runtime_read = memory.read_tool_output("output")
            saved_read = HarnessMemoryTools(history, lambda: cache).read_tool_output(
                "output"
            )
            for read in (runtime_read, saved_read):
                assert read.attachments is not None
                parts = list(read.attachments.iter_content_parts())
                assert "tool context" in parts
                assert read.attachments.get_name() == "work"
                if multimodal:
                    image = next(
                        part for part in parts if isinstance(part, Image.Image)
                    )
                    assert image.getpixel((0, 0)) == (255, 0, 0)
            assert cache.get("output").result == "result"
        finally:
            release.set()
