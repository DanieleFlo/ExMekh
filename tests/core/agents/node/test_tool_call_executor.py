from unittest.mock import MagicMock
from xml.etree import ElementTree as ET

import pytest

from exmekh_lib.core.agents.exit_tool.exit_tool import CalledExitTool, ExitTool
from exmekh_lib.core.agents.node import Node, tool_call_executor
from exmekh_lib.core.agents.node.agentic_scope import AgenticScope
from exmekh_lib.core.agents.node.tool_call_executor import ToolCallExecutor
from exmekh_lib.core.agents.node.tool_call_resolver import ToolCallResolver
from exmekh_lib.core.agents.node.tool_result_recorder import ToolResultRecorder
from exmekh_lib.core.agents.node.types import ToolExecutionResult
from exmekh_lib.core.cache import SessionCache
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import CalledTool, Tool
from exmekh_lib.core.models.tool_hooks import ToolHooks
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.tool_calls.execution_history import ExecutionHistory
from exmekh_lib.core.tool_calls.execution_manager import ToolExecutionManager
from exmekh_lib.core.tool_calls.harness_memory_tools import HarnessMemoryTools
from exmekh_lib.interfaces.tool_hook import AfterTool, BeforeTool, ToolContext
from tests.core.agents.node.test_node import (
    SequenceProvider,
    _write_runnable_capabilities,
)
from tests.tool_result_support import parse_tool_result, parse_used_tools


@pytest.fixture
def make_executor():
    def build(manager=None, *, output_ids=None):
        client = MagicMock(spec=Client)
        client.generate_id.return_value = "output"
        client.generate_id.side_effect = output_ids
        client.get_history.return_value = []
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


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize(
    "failure", ["arguments", "type", "value", "runtime", "negative"]
)
def test_work_failures_have_consistent_status_and_readable_results(
    make_executor, tmp_path, managed, failure
):
    def work(text):
        if failure == "type":
            raise TypeError("internal type error")
        if failure == "value":
            raise ValueError("internal value error")
        if failure == "runtime":
            raise RuntimeError("internal execution error")
        return {"ok": False}

    manager = ToolExecutionManager()
    called = CalledTool(
        Tool(work, name="work", background_enabled=managed),
        ToolCall(
            "origin",
            FunctionCall("work", {} if failure == "arguments" else {"text": "x"}),
        ),
    )
    result = make_executor(manager).execute(called)
    immediate = parse_tool_result(result.response.get_message())
    failed = failure != "negative"
    assert result.is_error is failed
    assert result.save_result
    assert (
        f"<response-type>{'failed' if failed else 'final'}</response-type>"
        in result.response.get_message()
    )
    assert immediate["ok"] is False
    if failed:
        assert immediate["error_kind"] == (
            "invalid_tool_arguments"
            if failure == "arguments"
            else "tool_execution_failed"
        )
        assert immediate["error_code"] == (
            "missing_tool_arguments"
            if failure == "arguments"
            else "tool_execution_failed"
        )
        assert immediate["hint"]
    with SessionCache(tmp_path / "cache") as session:
        cache = session.node("node")
        history = ExecutionHistory()
        recorder = ToolResultRecorder(history, lambda: cache, manager)
        identity = recorder.record_call(called, None)
        recorder.persist_result(result, identity, "/")
        for reader in (
            HarnessMemoryTools(history, lambda: cache, manager),
            HarnessMemoryTools(history, lambda: cache),
        ):
            saved = reader.read_tool_output(result.output_id).content
            assert parse_tool_result(saved) == immediate
            assert (
                f"<response-type>{'failed' if failed else 'final'}</response-type>"
                in saved
            )


def test_failed_inline_result_without_duration_has_failure_metadata():
    result = ToolExecutionResult(ToolMessage("failed", "origin"), failed=True)
    assert result.execution_metadata() == {"status": "failed"}


@pytest.mark.parametrize(
    "options", [{"execution_mode": "unknown"}, {"wait_seconds": 0}]
)
def test_invalid_execution_options_return_failure_without_starting_work(
    make_executor, options
):
    calls = []
    called = CalledTool(
        Tool(lambda: calls.append("ran"), name="work"),
        ToolCall("origin", FunctionCall("work", options)),
    )
    result = make_executor(ToolExecutionManager()).execute(called)
    assert result.is_error
    assert (
        parse_tool_result(result.response.get_message())["error_kind"]
        == "invalid_tool_arguments"
    )
    assert calls == []
    assert (
        parse_tool_result(result.response.get_message())["error_code"]
        == "invalid_execution_options"
    )


@pytest.mark.parametrize("error", [TypeError("manager bug"), ValueError("manager bug")])
def test_launch_infrastructure_failures_are_not_argument_errors(
    make_executor, monkeypatch, error
):
    manager = ToolExecutionManager()

    def fail(request):
        raise error

    monkeypatch.setattr(manager, "launch", fail)
    called = CalledTool(
        Tool(lambda: "done", name="work"), ToolCall("call", FunctionCall("work", {}))
    )
    result = make_executor(manager).execute(called)
    body = parse_tool_result(result.response.get_message())
    assert result.is_error
    assert body["error_kind"] == "tool_execution_failed"
    assert body["error_code"] == "tool_execution_failed"


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize(
    "arguments,code",
    [
        ({}, "missing_tool_arguments"),
        ({"text": "x", "typo": "y"}, "unexpected_tool_arguments"),
    ],
)
def test_argument_diagnostics_expose_required_names_before_work(
    make_executor, managed, arguments, code
):
    calls = []

    def work(text):
        calls.append(text)
        return "done"

    tool = Tool(work, name="work", background_enabled=managed)
    executor = make_executor(
        ToolExecutionManager(), output_ids=["error-output", "final-output"]
    )
    rejected = executor.execute(
        CalledTool(tool, ToolCall("bad", FunctionCall("work", arguments)))
    )
    result = parse_tool_result(rejected.response.get_message())
    assert result["error_kind"] == "invalid_tool_arguments"
    assert result["error_code"] == code
    assert result["details"]["required"] == ["text"]
    assert result["details"]["accepted"] == ["text"]
    assert calls == []
    corrected = executor.execute(
        CalledTool(tool, ToolCall("good", FunctionCall("work", {"text": "x"})))
    )
    assert corrected.is_error is False
    assert calls == ["x"]


@pytest.mark.parametrize("mode", ["inline", "managed", "exit"])
def test_runtime_hooks_receive_messages_without_mutating_client_history(tmp_path, mode):
    seen = []

    class InspectHistory(BeforeTool):
        def handle(self, context: ToolContext) -> None:
            assert isinstance(context.history, tuple)
            assert isinstance(context.history[0], SystemMessage)
            user = context.history[1]
            assert isinstance(user, UserMessage)
            assert user.get_message() == ["start"]
            opening = context.history[-1]
            assert isinstance(opening, AssistantMessage)
            assert opening.get_tool_calls()[0].get_id() == "origin"
            assert user.get_metadata() == {"local": {"value": "original"}}
            user.get_message().append("hook change")
            user.set_metadata("local", {"value": "hook"})
            seen.append(context.history)

    class CheckHistory(AfterTool):
        def handle(self, context: ToolContext, result):
            assert context.history is seen[0]
            return result

    capabilities = Node(
        [_write_runnable_capabilities(tmp_path)], name_hds="main"
    ).load_capabilities(work_root=tmp_path)
    client = Client(SequenceProvider([]), SystemMessage("rules"))
    user = UserMessage(["start"])
    user.set_metadata("local", {"value": "original"})
    client.add_user_message(user)
    hooks = ToolHooks(before=InspectHistory(), after=CheckHistory())
    exit_tool = capabilities.get_exit_tool("exit_agent").with_hooks(hooks)
    if mode == "exit":
        tool_call = ToolCall(
            "origin", FunctionCall("exit_agent", {"text": "done", "items": []})
        )
        called = CalledExitTool(exit_tool, tool_call)
    else:
        tool_call = ToolCall("origin", FunctionCall("work", {}))
        called = CalledTool(
            Tool(
                lambda: "done", name="work", background_enabled=mode == "managed"
            ).with_hooks(hooks),
            tool_call,
        )
    client.add_tool_calls([tool_call])
    executor = ToolCallExecutor(
        client=client,
        exit_tool=exit_tool,
        initial_depth=0,
        agentic_scope=AgenticScope(capabilities.get_hds("main"), "exit_agent"),
        exit_content_resolver=lambda content, items, path, position: content,
        work_root=tmp_path,
        execution_manager=ToolExecutionManager(),
    )

    result = executor.execute(called)

    assert not result.is_error
    assert len(seen) == 1
    assert client.get_history()[1].get_message() == ["start"]
    assert user.get_metadata() == {"local": {"value": "original"}}
    assert client.dump_history()[1] == {"role": "user", "content": "start"}
