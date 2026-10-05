from unittest.mock import MagicMock
from xml.etree import ElementTree as ET

import pytest

from exmekh_lib.core.agents.exit_tool.exit_tool import ExitTool
from exmekh_lib.core.agents.node import Node, tool_call_executor
from exmekh_lib.core.agents.node.agentic_scope import AgenticScope
from exmekh_lib.core.agents.node.tool_call_executor import ToolCallExecutor
from exmekh_lib.core.agents.node.tool_call_resolver import ToolCallResolver
from exmekh_lib.core.agents.node.tool_result_recorder import ToolResultRecorder
from exmekh_lib.core.cache import SessionCache
from exmekh_lib.core.models.assistant_message import FunctionCall, ToolCall
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import CalledTool, Tool
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.tool_calls.execution_history import ExecutionHistory
from exmekh_lib.core.tool_calls.execution_manager import ToolExecutionManager
from exmekh_lib.core.tool_calls.harness_memory_tools import HarnessMemoryTools
from tests.core.agents.node.test_node import (
    SequenceProvider,
    _write_runnable_capabilities,
)
from tests.tool_result_support import parse_used_tools


@pytest.fixture
def make_executor():
    def build(manager=None):
        client = MagicMock(spec=Client)
        client.generate_id.return_value = "output"
        client.dump_history.return_value = []
        executor = ToolCallExecutor(
            client=client,
            exit_tool=MagicMock(spec=ExitTool),
            initial_depth=0,
            agentic_scope=MagicMock(spec=AgenticScope),
            exit_content_resolver=MagicMock(),
            execution_manager=manager,
        )
        return executor

    return build


def unavailable_annotation():
    raise RuntimeError("Thinking annotation unavailable")


@pytest.mark.parametrize("managed", [False, True])
def test_annotation_failure_preserves_the_tool_result(make_executor, managed):
    executor = make_executor(ToolExecutionManager() if managed else None)
    called = CalledTool(
        Tool(lambda: "done", name="work", background_enabled=managed),
        ToolCall("call", FunctionCall("work", {})),
    )

    result = executor.execute(
        called, thinking_annotation_provider=unavailable_annotation
    )

    assert result.raw_result == "done"
    assert result.response.get_tool_call_id() == "call"
    header, body = result.response.get_message().split("</header>\n", 1)
    root = ET.fromstring(header + "</header>")
    assert root.findtext("id") == result.output_id == "output"
    assert root.findtext("response-type") == "final"
    assert root.find("thinking") is None
    assert body == "done"


def test_skill_return_without_execution_manager_has_a_complete_header(tmp_path):
    capabilities_path = _write_runnable_capabilities(tmp_path)
    (capabilities_path / "deep-skill" / "mock-skill.md").write_text(
        "---\ndescription: Mock skill\ninherits-parent-tools: true\n---\nMock skill body"
    )
    capabilities = Node([capabilities_path], name_hds="main").load_capabilities(
        work_root=tmp_path
    )
    client = Client(SequenceProvider([]), SystemMessage("rules"))
    scope = AgenticScope(capabilities.get_hds("main"), "exit_agent")
    executor = ToolCallExecutor(
        client=client,
        exit_tool=capabilities.get_exit_tool("exit_agent"),
        initial_depth=0,
        agentic_scope=scope,
        exit_content_resolver=lambda content, items, path, position: content,
        work_root=tmp_path,
    )
    resolver = ToolCallResolver(capabilities)
    client.add_user_message(UserMessage("start"))
    opening = ToolCall("open", FunctionCall("mock-skill", {}))
    client.add_tool_calls([opening])
    client.add_tool_response(executor.execute(resolver.resolve(opening)).response)
    work = ToolCall("work", FunctionCall("mock-tool", {"text": "hello"}))
    client.add_tool_calls([work])
    work_result = executor.execute(resolver.resolve(work))
    client.add_tool_response(work_result.response)
    closing = ToolCall(
        "exit", FunctionCall("exit_agent", {"text": "done", "items": []})
    )
    client.add_tool_calls([closing])

    result = executor.execute(resolver.resolve(closing))

    header, body = result.response.get_message().split("</header>\n", 1)
    root = ET.fromstring(header + "</header>")
    assert root.findtext("type") == "DeepSkill"
    assert root.findtext("name") == "mock-skill"
    assert root.findtext("id") == result.output_id
    assert root.find("background-tools") is None
    assert parse_used_tools(result.response.get_message()) == [
        {"name": "mock-tool", "id": work_result.output_id}
    ]
    assert body == "done"
    assert result.raw_result == {"text": "done", "items": []}
    assert result.response.get_tool_call_id() == "open"
    assert executor.pending_return_call_id == "open"
    assert executor.depth == 0


def test_synchronous_duration_is_measured_and_preserved_in_saved_reads(
    make_executor, tmp_path, monkeypatch
):

    clocks = iter([10, 12.5])
    monkeypatch.setattr(tool_call_executor, "monotonic", lambda: next(clocks))
    called = CalledTool(
        Tool(lambda: "done", name="work", background_enabled=False),
        ToolCall("call", FunctionCall("work", {})),
    )
    result = make_executor().execute(called)
    assert result.elapsed_seconds == 2.5
    immediate_header = ET.fromstring(
        result.response.get_message().split("</header>")[0] + "</header>"
    )
    assert immediate_header.findtext("elapsed-seconds") == "2.5"
    with SessionCache(tmp_path) as session:
        cache = session.node("node")
        history = ExecutionHistory()
        recorder = ToolResultRecorder(history, lambda: cache, ToolExecutionManager())
        identity = recorder.record_call(called, None)
        recorder.persist_result(result, identity, "/")
        record = cache.get(result.output_id)
        assert record.execution == {"status": "completed", "elapsed_seconds": 2.5}
        read = HarnessMemoryTools(history, lambda: cache).read_tool_output(
            result.output_id
        )
        header = ET.fromstring(read.content.split("</header>")[0] + "</header>")
        assert header.findtext("elapsed-seconds") == "2.5"
        assert read.content.endswith("done")
