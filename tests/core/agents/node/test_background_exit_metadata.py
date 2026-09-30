from itertools import count
from pathlib import Path
from threading import Event
from xml.etree import ElementTree as ET

import pytest

from qs.core.agents.capabilities.defaults import DefaultCapabilities
from qs.core.agents.node.node import Node
from qs.core.cache import SessionCache
from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.tool import Tool, ToolExecutionOperation
from qs.core.models.tool_message import ToolMessage
from tests.core.agents.node.steering_support import call, tool_reply
from tests.core.agents.node.test_node import (
    SequenceProvider,
    _write_nested_capabilities,
)
from tests.tool_result_support import parse_tool_result


def exit_call(call_id):
    return call("exit_agent", {"text": "done", "items": []}, call_id)


def background_tools(content):
    header = ET.fromstring(content.split("</header>", 1)[0] + "</header>")
    tools = header.find("background-tools")
    if tools is None:
        return []
    assert tools.get("type") == "array"
    return parse_tool_result(ET.tostring(tools, encoding="unicode"))


def pending_tool(name, call_id, output_id, skill_path):
    return {
        "name": name,
        "tool-call-id": call_id,
        "output-id": output_id,
        "skill-path": skill_path,
        "status": "running",
    }


class ControlledTool(Tool):
    def __init__(self, name, release, *, fails=False):
        self.release = release
        self.started = Event()
        self.fails = fails
        super().__init__(lambda: {"ok": True}, name=name)

    def create_execution(self, arguments, **context):
        def run():
            self.started.set()
            self.release.wait()
            if self.fails:
                raise RuntimeError("work failed")
            return {"ok": True}

        return ToolExecutionOperation(
            run,
            cancel=lambda: True,
            read_partial=lambda cursor: ("progress"[cursor:], len("progress")),
        )


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("cache_result", [True, False])
@pytest.mark.parametrize("mode", ["background", "wait"])
def test_exit_metadata_keeps_running_work_from_closed_descendants(
    make_agent, depth, cache_result, mode
):
    release = Event()
    responses = [tool_reply(call("mock-skill", call_id="parent"))]
    if depth == 2:
        responses.append(tool_reply(call("nested-skill", call_id="child")))
    responses.extend(
        [
            tool_reply(
                call(
                    "mock-sub-tool",
                    {"execution_mode": mode, "wait_seconds": 0.01},
                    "work",
                )
            ),
            tool_reply(call("read_tool_output", {"output_id": "output-0"}, "read")),
            tool_reply(
                call("shared-tool", {"text": "inline result"}, "other"),
                exit_call("exit"),
            ),
        ]
    )
    if depth == 2:
        responses.append(tool_reply(exit_call("parent-exit")))
    responses.append(AssistantMessage("final"))
    provider = SequenceProvider(responses)
    agent = make_agent(
        provider,
        Tool(
            lambda: release.wait() and {"ok": True},
            name="mock-sub-tool",
            cache_result=cache_result,
        ),
        Tool(lambda text: text, name="shared-tool", background_enabled=False),
    )
    try:
        response = agent.run("start")
        path = "/mock-skill" + ("/nested-skill" if depth == 2 else "")
        expected = [pending_tool("mock-sub-tool", "work", "output-0", path)]
        outer = response.skill_executions[0]
        assert background_tools(outer.output) == expected
        if depth == 2:
            assert background_tools(outer.children[0].output) == expected
        result = provider.requests[-1].get_messages()[-1]
        assert result.get_name() == "mock-skill"
        assert background_tools(result.get_message()) == expected
        assert result.get_message().split("</header>\n", 1)[1] == "done"
        cache = agent.root_node._runtime._get_tool_cache()
        assert (cache.get("output-0") is not None) == cache_result
        assert cache.get(
            ET.fromstring(outer.output.split("</header>")[0] + "</header>").findtext(
                "id"
            )
        ).result == {"text": "done", "items": []}
    finally:
        release.set()
        agent.root_node.control_tool_execution("work", wait_seconds=1)


@pytest.mark.parametrize("reused_opening_id", [False, True])
def test_each_exit_lists_its_own_invocation_and_parent_collects_all_children(
    make_agent, reused_opening_id
):
    release = Event()
    second_id = "child" if reused_opening_id else "second-child"
    provider = SequenceProvider(
        [
            tool_reply(call("mock-skill", call_id="parent")),
            tool_reply(call("nested-skill", call_id="child")),
            tool_reply(call("shared-tool", {"execution_mode": "background"}, "first")),
            tool_reply(exit_call("first-exit")),
            tool_reply(call("nested-skill", call_id=second_id)),
            tool_reply(call("shared-tool", {"execution_mode": "background"}, "second")),
            tool_reply(exit_call("second-exit")),
            tool_reply(exit_call("parent-exit")),
            AssistantMessage("final"),
        ]
    )
    agent = make_agent(provider, ControlledTool("shared-tool", release))
    try:
        response = agent.run("start")
        outer = response.skill_executions[0]
        first = pending_tool(
            "shared-tool", "first", "output-0", "/mock-skill/nested-skill"
        )
        second = pending_tool(
            "shared-tool", "second", "output-2", "/mock-skill/nested-skill"
        )
        assert background_tools(outer.children[0].output) == [first]
        assert background_tools(outer.children[1].output) == [second]
        assert background_tools(outer.output) == [first, second]
    finally:
        release.set()
        for call_id in ("first", "second"):
            agent.root_node.control_tool_execution(call_id, wait_seconds=1)


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
@pytest.mark.parametrize("delivery", ["automatic", "manual", "unread"])
def test_exit_excludes_terminal_work_even_when_final_output_has_not_been_read(
    make_agent, monkeypatch, status, delivery
):
    release = Event()
    tool = ControlledTool("mock-sub-tool", release, fails=status == "failed")
    responses = [
        tool_reply(call("mock-skill", call_id="parent")),
        tool_reply(call("mock-sub-tool", {"execution_mode": "background"}, "work")),
    ]
    if delivery == "automatic":
        responses.append(tool_reply(call("shared-tool", {"text": "continue"}, "other")))
    elif delivery == "manual":
        responses.append(
            tool_reply(call("read_tool_output", {"output_id": "output-0"}, "read"))
        )
    responses.extend([tool_reply(exit_call("exit")), AssistantMessage("final")])
    provider = SequenceProvider(responses)
    agent = make_agent(provider, tool)
    original_call = provider.call

    def finish_before_next_response(request):
        if len(provider.requests) == 2:
            assert tool.started.wait(1)
            if status == "cancelled":
                assert agent.root_node.control_tool_execution("work", "cancel")[
                    "cancelled"
                ]
            release.set()
            assert (
                agent.root_node.control_tool_execution("work", wait_seconds=1)["status"]
                == status
            )
        return original_call(request)

    monkeypatch.setattr(provider, "call", finish_before_next_response)
    try:
        response = agent.run("start")
        execution = response.skill_executions[0]
        assert background_tools(execution.output) == []
        reads = [
            message
            for message in execution.history
            if isinstance(message, ToolMessage)
            and message.get_name() == "read_tool_output"
        ]
        assert len(reads) == int(delivery != "unread")
        if reads:
            expected_type = "final" if status == "completed" else status
            assert (
                f"<response-type>{expected_type}</response-type>"
                in reads[0].get_message()
            )
    finally:
        release.set()
        agent.root_node.control_tool_execution("work", wait_seconds=1)


def test_partial_reads_keep_work_listed_and_completed_child_is_removed_at_parent_exit(
    make_agent, monkeypatch
):
    release = Event()
    tool = ControlledTool("shared-tool", release)
    provider = SequenceProvider(
        [
            tool_reply(call("mock-skill", call_id="parent")),
            tool_reply(call("nested-skill", call_id="child")),
            tool_reply(call("shared-tool", {"execution_mode": "background"}, "work")),
            tool_reply(
                call("read_tool_output", {"output_id": "output-0", "cursor": 0}, "read")
            ),
            tool_reply(exit_call("child-exit")),
            tool_reply(exit_call("parent-exit")),
            AssistantMessage("final"),
        ]
    )
    agent = make_agent(provider, tool)
    original_call = provider.call

    def finish_after_child_return(request):
        if len(provider.requests) == 5:
            release.set()
            assert (
                agent.root_node.control_tool_execution("work", wait_seconds=1)["status"]
                == "completed"
            )
        return original_call(request)

    monkeypatch.setattr(provider, "call", finish_after_child_return)
    try:
        response = agent.run("start")
        outer = response.skill_executions[0]
        child = outer.children[0]
        assert background_tools(child.output) == [
            pending_tool("shared-tool", "work", "output-0", "/mock-skill/nested-skill")
        ]
        assert background_tools(outer.output) == []
        read = next(
            message
            for message in child.history
            if isinstance(message, ToolMessage)
            and message.get_name() == "read_tool_output"
        )
        assert "<response-type>partial</response-type>" in read.get_message()
    finally:
        release.set()
        agent.root_node.control_tool_execution("work", wait_seconds=1)


def test_exit_still_has_metadata_when_provider_cannot_generate_an_id(
    make_agent, monkeypatch
):
    release = Event()
    provider = SequenceProvider(
        [
            tool_reply(call("mock-skill", call_id="parent")),
            tool_reply(call("mock-sub-tool", {"execution_mode": "background"}, "work")),
            tool_reply(exit_call("exit")),
            AssistantMessage("final"),
        ]
    )
    agent = make_agent(provider, Tool(release.wait, name="mock-sub-tool"))

    def unavailable_id():
        raise RuntimeError("Tokenizer unavailable")

    monkeypatch.setattr(provider, "generate_id", unavailable_id)
    try:
        response = agent.run("start")
        (snapshot,) = agent.root_node._runtime._execution_manager.output_snapshots()
        assert background_tools(response.skill_executions[0].output) == [
            pending_tool("mock-sub-tool", "work", snapshot.output_id, "/mock-skill")
        ]
    finally:
        release.set()
        agent.root_node.control_tool_execution("work", wait_seconds=1)


def test_three_skill_levels_collect_all_tool_kinds_without_ancestor_work(
    tmp_path: Path,
):
    capabilities = _write_nested_capabilities(tmp_path)
    nested_path = capabilities / "deep-skill" / "nested-skill.md"
    nested_path.write_text(
        nested_path.read_text().replace(
            "sub-tools:\n", "sub-skills:\n  - leaf-skill\nsub-tools:\n"
        )
    )
    (capabilities / "deep-skill" / "leaf-skill.md").write_text(
        "---\ndescription: Leaf skill\ninherits-parent-tools: true\n---\nLeaf body"
    )
    release = Event()
    tools = (
        Tool(lambda: release.wait() and "text", name="mock-tool"),
        Tool(lambda: release.wait() and {"ok": True}, name="mock-sub-tool"),
        Tool(
            lambda: release.wait() and [1, 2],
            name="nested-sub-tool",
            cache_result=False,
        ),
        ControlledTool("shared-tool", release),
    )
    provider = SequenceProvider(
        [
            tool_reply(
                call("mock-tool", {"execution_mode": "background"}, "root-work")
            ),
            tool_reply(call("mock-skill", call_id="parent")),
            tool_reply(
                call("mock-sub-tool", {"execution_mode": "background"}, "parent-work")
            ),
            tool_reply(call("nested-skill", call_id="child")),
            tool_reply(
                call("nested-sub-tool", {"execution_mode": "background"}, "child-work")
            ),
            tool_reply(call("leaf-skill", call_id="leaf")),
            tool_reply(
                exit_call("leaf-exit"),
                call("shared-tool", {"execution_mode": "background"}, "leaf-work"),
            ),
            tool_reply(exit_call("child-exit")),
            tool_reply(exit_call("parent-exit")),
            AssistantMessage("final"),
        ]
    )
    with SessionCache(tmp_path / "cache") as session:
        provider.get_session = lambda: session
        ids = count()
        provider.generate_id = lambda: f"output-{next(ids)}"
        node = Node(
            [capabilities],
            name_hds="main",
            default_capabilities=DefaultCapabilities(
                harness=False,
                include={"read_tool_output", "control_tool_execution"},
                overrides={tool.name: tool for tool in tools},
            ),
        )
        node.initialize(provider, work_root=tmp_path)
        try:
            response = node.run("start")
            parent = response.skill_executions[0]
            child = parent.children[0]
            leaf = child.children[0]
            parent_work = pending_tool(
                "mock-sub-tool", "parent-work", "output-1", "/mock-skill"
            )
            child_work = pending_tool(
                "nested-sub-tool", "child-work", "output-2", "/mock-skill/nested-skill"
            )
            leaf_work = pending_tool(
                "shared-tool",
                "leaf-work",
                "output-3",
                "/mock-skill/nested-skill/leaf-skill",
            )
            assert background_tools(leaf.output) == [leaf_work]
            assert background_tools(child.output) == [child_work, leaf_work]
            assert background_tools(parent.output) == [
                parent_work,
                child_work,
                leaf_work,
            ]
            assert (
                node._runtime._execution_manager.by_output_id("output-0").status
                == "running"
            )
        finally:
            release.set()
            for call_id in ("root-work", "parent-work", "child-work", "leaf-work"):
                node.control_tool_execution(call_id, wait_seconds=1)
