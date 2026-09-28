from pathlib import Path
from threading import Event
from xml.etree import ElementTree as ET

import pytest

from qs.core.agents.capabilities.defaults import DefaultCapabilities
from qs.core.agents.node.node import Node
from qs.core.cache import SessionCache
from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.tool import PreformattedToolResult, Tool, ToolExecutionOperation
from qs.core.models.tool_message import ToolMessage
from qs.interfaces.tool_hook import AfterTool, BeforeTool, ToolArguments, ToolContext
from tests.core.agents.node.test_node import (
    SequenceProvider,
    _function_call,
    _write_nested_capabilities,
)


def tool_message(name: str, arguments: dict, call_id: str) -> AssistantMessage:
    return AssistantMessage(
        None, tool_calls=[_function_call(name, arguments, call_id=call_id)]
    )


@pytest.fixture
def make_node(tmp_path: Path):
    capabilities = _write_nested_capabilities(tmp_path)
    with SessionCache(tmp_path / "cache") as session:

        def build(provider, tool, *, persistent=True):
            provider.get_session = lambda: session
            counter = iter(range(1, 100))
            provider.generate_id = lambda: f"output-{next(counter)}"
            node = Node(
                [capabilities],
                name_hds="main",
                persistent_history=persistent,
                default_capabilities=DefaultCapabilities(
                    harness=False,
                    include={
                        "read_tool_output",
                        "control_tool_execution",
                        "get_execution_history",
                        "search_tool_outputs",
                    },
                    overrides={tool.name: tool},
                ),
            )
            node.initialize(provider, work_root=tmp_path)
            return node

        yield build


def read_calls(messages) -> list:
    return [
        call
        for message in messages
        if isinstance(message, AssistantMessage)
        for call in message.get_tool_calls() or ()
        if call.get_name() == "read_tool_output"
    ]


def test_background_status_control_and_saved_result_use_xml(make_node):
    release = Event()
    page = {"title": "Wikipedia", "markdown": "# Wikipedia\nFree <&> encyclopedia"}
    provider = SequenceProvider(
        [
            tool_message("mock-tool", {"execution_mode": "background"}, "origin"),
            tool_message(
                "control_tool_execution",
                {"tool_call_id": "origin", "wait_seconds": 0.01},
                "control",
            ),
            AssistantMessage("pending"),
        ]
    )
    node = make_node(provider, Tool(lambda: release.wait() and page, name="mock-tool"))
    try:
        node.run("start")
        messages = provider.requests[-1].get_messages()
        outputs = {
            message.get_name(): ET.fromstring(
                message.get_message().split("</header>\n", 1)[1]
            )
            for message in messages
            if isinstance(message, ToolMessage)
        }
        for name in ("mock-tool", "control_tool_execution"):
            assert outputs[name].tag == "tool-result"
            assert outputs[name].findtext("status") == "running"
            assert outputs[name].findtext("tool_call_id") == "origin"
            assert outputs[name].findtext("output_id") == "output-1"
    finally:
        release.set()
        node.control_tool_execution("origin", wait_seconds=1)
    body = node.read_tool_output("output-1").content.split("</header>\n", 1)[1]
    assert ET.fromstring(body).findtext("markdown") == page["markdown"]


def test_search_finds_uncached_work_after_its_owner_skill_was_compacted(make_node):
    release = Event()
    provider = SequenceProvider(
        [
            tool_message("mock-skill", {}, "open"),
            tool_message(
                "mock-sub-tool",
                {"text": "work", "execution_mode": "background"},
                "origin",
            ),
            tool_message("exit_agent", {"text": "pending", "items": []}, "exit"),
            tool_message(
                "search_tool_outputs",
                {"tool_name": "mock-sub-tool", "skill_path": "/mock-skill"},
                "search",
            ),
            AssistantMessage("done"),
        ]
    )
    node = make_node(
        provider,
        Tool(
            lambda text: release.wait() and text,
            name="mock-sub-tool",
            cache_result=False,
        ),
    )
    try:
        assert node.run("start").get_message() == "done"
        from_search = provider.requests[-1].get_messages()[-1].get_message()
        assert "<skill-path>/mock-skill</skill-path>" in from_search
        assert "<response-type>running</response-type>" in from_search
        assert "<storage>runtime</storage>" in from_search
        assert "<id>output-1</id>" in from_search
        assert "<tool-call-id>origin</tool-call-id>" in from_search
        assert read_calls(provider.requests[-1].get_messages()) == []
    finally:
        release.set()
        node.control_tool_execution("origin", wait_seconds=1)


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        ("control_tool_execution", {"tool_call_id": "call"}),
        ("read_tool_output", {"output_id": "output"}),
    ],
)
def test_tool_execution_api_requires_an_initialized_node(method, arguments):
    node = Node(name_hds="main")
    with pytest.raises(RuntimeError, match="Node must be initialized"):
        getattr(node, method)(**arguments)


@pytest.mark.parametrize("persistent", [True, False])
def test_pending_result_is_delivered_on_next_turn_only_in_same_session(
    make_node, persistent
) -> None:
    release = Event()
    provider = SequenceProvider(
        [
            tool_message(
                "mock-tool", {"text": "work", "execution_mode": "background"}, "origin"
            ),
            AssistantMessage("first"),
            AssistantMessage("second"),
        ]
    )
    node = make_node(
        provider,
        Tool(lambda text: release.wait() and text, name="mock-tool"),
        persistent=persistent,
    )
    try:
        assert node.run("start").get_message() == "first"
    finally:
        release.set()
    node.control_tool_execution("origin", wait_seconds=1)
    assert node.run("continue").get_message() == "second"
    assert len(read_calls(provider.requests[-1].get_messages())) == int(persistent)
    assert "work" in node.read_tool_output("output-1").content


@pytest.mark.parametrize("reused_opening_id", [False, True])
def test_exit_and_reentry_do_not_deliver_old_skill_output(
    make_node, reused_opening_id
) -> None:
    release = Event()
    provider = SequenceProvider(
        [
            tool_message("mock-skill", {}, "open-1"),
            tool_message(
                "mock-sub-tool",
                {"text": "work", "execution_mode": "background"},
                "origin",
            ),
            tool_message(
                "exit_agent", {"text": "pending output-1", "items": []}, "exit-1"
            ),
            tool_message("mock-skill", {}, "open-1" if reused_opening_id else "open-2"),
            tool_message("exit_agent", {"text": "second skill", "items": []}, "exit-2"),
            tool_message("read_tool_output", {"output_id": "output-1"}, "manual-read"),
            AssistantMessage("done"),
        ]
    )
    node = make_node(
        provider, Tool(lambda text: release.wait() and text, name="mock-sub-tool")
    )
    original_call = provider.call

    def call(request):
        if len(provider.requests) == 3:
            release.set()
            node.control_tool_execution("origin", wait_seconds=1)
        return original_call(request)

    provider.call = call
    try:
        response = node.run("start")
    finally:
        release.set()
    assert response.get_message() == "done"
    assert len(response.skill_executions) == 2
    assert read_calls(provider.requests[4].get_messages()) == []
    reads = read_calls(provider.requests[-1].get_messages())
    assert [item.get_id() for item in reads] == ["manual-read"]
    assert provider.requests[-1].get_messages()[-1].get_message().endswith("work")


@pytest.mark.parametrize("completed_count", [1, 2])
def test_result_waits_for_original_parent_skill_after_nested_return(
    make_node, completed_count
) -> None:
    release = Event()
    provider = SequenceProvider(
        [
            tool_message("mock-skill", {}, "parent"),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "mock-sub-tool",
                        {"text": "work", "execution_mode": "background"},
                        call_id=f"origin-{index}",
                    )
                    for index in range(completed_count)
                ],
            ),
            tool_message("nested-skill", {}, "nested"),
            tool_message(
                "exit_agent", {"text": "nested done", "items": []}, "nested-exit"
            ),
            tool_message(
                "exit_agent", {"text": "parent done", "items": []}, "parent-exit"
            ),
            AssistantMessage("done"),
        ]
    )
    node = make_node(
        provider, Tool(lambda text: release.wait() and text, name="mock-sub-tool")
    )
    original_call = provider.call

    def call(request):
        if len(provider.requests) == 3:
            release.set()
            for index in range(completed_count):
                node.control_tool_execution(f"origin-{index}", wait_seconds=1)
        return original_call(request)

    provider.call = call
    try:
        response = node.run("start")
    finally:
        release.set()
    assert read_calls(provider.requests[3].get_messages()) == []
    assert len(read_calls(provider.requests[4].get_messages())) == completed_count
    parent_messages = provider.requests[4].get_messages()
    last_assistant = next(
        message
        for message in reversed(parent_messages)
        if isinstance(message, AssistantMessage)
    )
    assert {call.id for call in last_assistant.get_tool_calls()} >= {
        "nested",
        *(call.id for call in read_calls(parent_messages)),
    }, "Native skill return and automatic reads must share the latest batch"
    assert parent_messages[-1].get_tool_call_id() == "nested"
    assert any(
        isinstance(message, ToolMessage)
        and message.get_name() == "read_tool_output"
        and message.get_message().endswith("work")
        for message in parent_messages
    )
    calls = [
        call.get_id()
        for message in parent_messages
        if isinstance(message, AssistantMessage)
        for call in message.get_tool_calls() or ()
    ]
    results = [
        message.get_tool_call_id()
        for message in parent_messages
        if isinstance(message, ToolMessage)
    ]
    assert sorted(calls) == sorted(results)
    assert response.skill_executions[0].children[0].skill.name == "nested-skill"


def test_manual_final_read_suppresses_automatic_delivery(make_node) -> None:
    release = Event()
    provider = SequenceProvider(
        [
            tool_message(
                "mock-tool", {"text": "work", "execution_mode": "background"}, "origin"
            ),
            tool_message("read_tool_output", {"output_id": "output-1"}, "manual-read"),
            AssistantMessage("done"),
        ]
    )
    node = make_node(
        provider, Tool(lambda text: release.wait() and text, name="mock-tool")
    )
    original_call = provider.call

    def call(request):
        if len(provider.requests) == 1:
            release.set()
            node.control_tool_execution("origin", wait_seconds=1)
        return original_call(request)

    provider.call = call
    try:
        node.run("start")
    finally:
        release.set()
    assert [
        item.get_id() for item in read_calls(provider.requests[-1].get_messages())
    ] == ["manual-read"]
    assert node._runtime._get_tool_cache().get("output-1").result == "work"


def test_transient_background_result_is_readable_and_never_cached(make_node) -> None:
    release = Event()
    provider = SequenceProvider(
        [
            tool_message(
                "mock-tool",
                {"text": "private", "execution_mode": "background"},
                "origin",
            ),
            AssistantMessage("continuing"),
        ]
    )
    node = make_node(
        provider,
        Tool(
            lambda text: release.wait() and text, name="mock-tool", cache_result=False
        ),
    )
    try:
        node.run("start")
        assert node._runtime._get_tool_cache().get_all() == []
    finally:
        release.set()
    node.control_tool_execution("origin", wait_seconds=1)

    assert node.read_tool_output("output-1").content.endswith("private")
    node._runtime._sync_completed_tool_outputs()
    assert node._runtime._get_tool_cache().get_all() == []


def test_control_wait_has_its_own_call_id_and_no_output_content(make_node) -> None:
    release = Event()
    provider = SequenceProvider(
        [
            tool_message(
                "mock-tool", {"text": "work", "execution_mode": "background"}, "origin"
            ),
            tool_message(
                "control_tool_execution",
                {"tool_call_id": "origin", "wait_seconds": 0.01},
                "wait-call",
            ),
            AssistantMessage("continuing"),
        ]
    )
    node = make_node(
        provider, Tool(lambda text: release.wait() and text, name="mock-tool")
    )
    try:
        node.run("start")
        messages = provider.requests[-1].get_messages()
        results = [message for message in messages if isinstance(message, ToolMessage)]
        assert [message.get_tool_call_id() for message in results] == [
            "origin",
            "wait-call",
        ]
        content = results[-1].get_message()
        body = ET.fromstring(content.split("</header>\n", 1)[1])
        assert body.findtext("status") == "running"
        assert body.findtext("output_id") == "output-1"
        assert body.find("result") is None
        assert (
            node.control_tool_execution("origin", "cancel")["cancellation_supported"]
            is False
        )
    finally:
        release.set()


def test_completion_during_final_inference_is_delivered_before_return(
    make_node,
) -> None:
    release = Event()
    provider = SequenceProvider(
        [
            tool_message(
                "mock-tool", {"text": "work", "execution_mode": "background"}, "origin"
            ),
            AssistantMessage("draft"),
            AssistantMessage("done with result"),
        ]
    )
    node = make_node(
        provider, Tool(lambda text: release.wait() and text, name="mock-tool")
    )
    original_call = provider.call

    def call(request):
        if len(provider.requests) == 1:
            release.set()
            node.control_tool_execution("origin", wait_seconds=1)
        return original_call(request)

    provider.call = call
    try:
        assert node.run("start").get_message() == "done with result"
    finally:
        release.set()
    assert len(read_calls(provider.requests[-1].get_messages())) == 1


class ProgressTool(Tool):
    def __init__(self, release: Event, *, before=None, after=None):
        self.release = release
        self.started = Event()
        super().__init__(
            lambda text: text, name="mock-tool", before=before, after=after
        )

    def create_execution(self, arguments, **kwargs):
        def run():
            self.started.set()
            self.release.wait()
            return arguments["text"]

        return ToolExecutionOperation(
            run, read_partial=lambda cursor: ("raw-progress"[cursor:], 12)
        )


def test_hooked_custom_execution_filters_final_output_once_and_hides_raw_progress(
    make_node,
) -> None:
    release = Event()
    calls = []

    class Before(BeforeTool):
        def handle(self, context: ToolContext):
            calls.append(("before", context.arguments))
            return ToolArguments({"text": "raw-final"})

    class After(AfterTool):
        def handle(self, context: ToolContext, result):
            calls.append(("after", result))
            return "filtered"

    tool = ProgressTool(release, before=Before(), after=After())
    provider = SequenceProvider(
        [
            tool_message(
                "mock-tool", {"text": "work", "execution_mode": "background"}, "origin"
            ),
            AssistantMessage("continuing"),
        ]
    )
    node = make_node(provider, tool)
    try:
        node.run("start")
        assert tool.started.wait(1)
        pending = node.read_tool_output("output-1").content
        assert "raw-progress" not in pending
    finally:
        release.set()
    node.control_tool_execution("origin", wait_seconds=1)
    final = node.read_tool_output("output-1").content
    assert final.endswith("filtered")
    assert "raw-final" not in final
    assert calls == [("before", {"text": "work"}), ("after", "raw-final")]


def test_incremental_reads_keep_full_read_and_final_result_available(make_node) -> None:
    release = Event()
    tool = ProgressTool(release)
    provider = SequenceProvider(
        [
            tool_message(
                "mock-tool", {"text": "final", "execution_mode": "background"}, "origin"
            ),
            AssistantMessage("continuing"),
        ]
    )
    node = make_node(provider, tool)
    try:
        node.run("start")
        assert tool.started.wait(1)
        reader = node
        first = reader.read_tool_output("output-1", cursor=0).content
        assert first.endswith("raw-progress")
        assert "<response-type>partial</response-type>" in first
        second = reader.read_tool_output("output-1", cursor=12).content
        assert "raw-progress" not in second
        assert reader.read_tool_output("output-1").content.endswith("raw-progress")
    finally:
        release.set()
    node.control_tool_execution("origin", wait_seconds=1)
    final = reader.read_tool_output("output-1", cursor=12).content
    assert final.endswith("final")
    assert "<response-type>final</response-type>" in final


def test_canonical_output_id_is_exposed_even_without_a_tokenizer(make_node) -> None:
    release = Event()
    provider = SequenceProvider(
        [
            tool_message(
                "mock-tool", {"text": "work", "execution_mode": "background"}, "origin"
            ),
            AssistantMessage("continuing"),
        ]
    )
    node = make_node(
        provider, Tool(lambda text: release.wait() and text, name="mock-tool")
    )

    def unavailable():
        raise RuntimeError("Tokenizer unavailable")

    provider.generate_id = unavailable
    try:
        node.run("start")
        record = node._runtime._get_tool_cache().get_all()[0]
        assert (
            f"<id>{record.id}</id>"
            in provider.requests[-1].get_messages()[-1].get_message()
        )
        assert node.read_tool_output(record.id).content
    finally:
        release.set()


def test_preformatted_result_is_saved_as_content_after_completion(make_node) -> None:
    provider = SequenceProvider(
        [
            tool_message("mock-tool", {"text": "work"}, "origin"),
            AssistantMessage("done"),
        ]
    )
    node = make_node(
        provider,
        Tool(
            lambda text: PreformattedToolResult("<result>filtered</result>"),
            name="mock-tool",
        ),
    )
    node.run("start")
    record = node._runtime._get_tool_cache().get("output-1")
    assert record.result == "<result>filtered</result>"
    assert node.read_tool_output("output-1").content.endswith(record.result)


def test_supported_control_cancellation_is_readable_as_cancelled(make_node) -> None:
    release = Event()

    class CancellableTool(ProgressTool):
        def create_execution(self, arguments, **kwargs):
            work = super().create_execution(arguments, **kwargs)
            work.cancel = lambda: True
            return work

    tool = CancellableTool(release)
    provider = SequenceProvider(
        [
            tool_message(
                "mock-tool", {"text": "work", "execution_mode": "background"}, "origin"
            ),
            AssistantMessage("continuing"),
        ]
    )
    node = make_node(provider, tool)
    try:
        node.run("start")
        assert tool.started.wait(1)
        state = node.control_tool_execution("origin", "cancel")
        assert state["cancelled"] is True
        output = node.read_tool_output("output-1").content
        assert "<response-type>cancelled</response-type>" in output
        assert output.endswith("Tool execution was cancelled.")
    finally:
        release.set()


@pytest.mark.parametrize("fails", [True, False])
def test_tool_exception_is_distinct_from_negative_result(make_node, fails) -> None:
    def work(text):
        if fails:
            raise RuntimeError("work failed")
        return {"ok": False}

    provider = SequenceProvider(
        [
            tool_message("mock-tool", {"text": "work"}, "origin"),
            AssistantMessage("done"),
        ]
    )
    node = make_node(provider, Tool(work, name="mock-tool"))
    node.run("start")
    output = node.read_tool_output("output-1").content
    expected = "failed" if fails else "final"
    assert f"<response-type>{expected}</response-type>" in output


def test_background_output_is_isolated_from_another_node(make_node) -> None:
    release = Event()
    provider = SequenceProvider(
        [
            tool_message(
                "mock-tool",
                {"text": "private", "execution_mode": "background"},
                "origin",
            ),
            AssistantMessage("continuing"),
        ]
    )
    tool = Tool(lambda text: release.wait() and text, name="mock-tool")
    first = make_node(provider, tool)
    second = make_node(SequenceProvider([AssistantMessage("other")]), tool)
    try:
        first.run("start")
        assert "<found>false</found>" in second.read_tool_output("output-1").content
        with pytest.raises(ValueError, match="Unknown tool call ID"):
            second.control_tool_execution("origin", "wait", 0.01)
    finally:
        release.set()
