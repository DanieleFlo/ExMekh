from threading import Event

import pytest

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION
from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.agents.deep_agent.stream_event import (
    DeepAgentExecutionEvent,
    DeepAgentExecutionEventType,
)
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import UserMessage
from tests.core.agents.node.steering_support import call, tool_reply
from tests.core.agents.node.test_node import _write_nested_capabilities
from tests.core.history_compaction.support import CompactionProvider, compacted_payload
from tests.event_handler import as_agent_handler
from tests.history_validation import validate_request_history


def runtime_history(agent):
    return agent.root_node._runtime._client.get_history()


def test_idle_compaction_is_used_by_the_next_turn_and_preserves_saved_outputs(
    make_agent,
):
    provider = CompactionProvider(
        [
            tool_reply(call()),
            AssistantMessage("first done"),
            AssistantMessage("next done"),
        ],
        score=9,
    )
    agent = make_agent(provider, Tool(lambda: "original body", name="mock-tool"))
    agent.run("first")
    agent.compact_history(agent.root_node)
    compacted = runtime_history(agent)
    assert [item.get_role_name() for item in compacted] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert compacted_payload(compacted[-1])["outputs"][0]["id"] == "output-0"
    assert agent.root_node.read_tool_output("output-0").content.endswith(
        "original body"
    )
    agent.run("next")
    assert provider.requests[-1].get_messages()[:-1] == compacted
    validate_request_history(provider.requests[-1])


def test_active_compaction_waits_for_inference_and_complete_batch(make_agent):
    provider = CompactionProvider(
        [tool_reply(call()), AssistantMessage("done")], score=9
    )
    agent = make_agent(provider, Tool(lambda: "body", name="mock-tool"))
    provider.on_call = lambda request: (
        agent.compact_history(agent.root_node) if not provider.requests else None
    )
    assert agent.run("task").get_message() == "done"
    assert len(provider.score_requests) == 1
    assert provider.catalogs[0]["outputs"][0]["output_id"] == "output-0"
    assert isinstance(provider.requests[-1].get_messages()[-1], ToolMessage)
    assert (
        provider.requests[-1].get_messages()[-1].get_name()
        == HISTORY_COMPACTION.tool_name
    )


def test_compaction_in_nested_skills_preserves_instructions_scopes_and_exit_anchors(
    make_agent,
):
    provider = CompactionProvider(
        [
            tool_reply(call()),
            tool_reply(call("mock-skill", call_id="open")),
            tool_reply(call("mock-sub-tool", {"text": "parent work"}, "parent-work")),
            tool_reply(call("nested-skill", call_id="nested-open")),
            tool_reply(call("nested-sub-tool", {"text": "nested work"}, "nested-work")),
            tool_reply(
                call("exit_agent", {"text": "nested done", "items": []}, "nested-exit")
            ),
            tool_reply(call("shared-tool", {"text": "parent continued"}, "continued")),
            tool_reply(
                call("exit_agent", {"text": "parent done", "items": []}, "parent-exit")
            ),
            AssistantMessage("done"),
        ],
        score=9,
    )
    agent = make_agent(provider, Tool(lambda: "root output", name="mock-tool"))
    before_scopes = []

    def request_compaction(request):
        if len(provider.requests) == 4:
            before_scopes.append(agent.root_node._runtime._executor.skill_scope_ids)
            agent.compact_history(agent.root_node)
        elif len(provider.requests) == 5:
            assert (
                agent.root_node._runtime._executor.skill_scope_ids == before_scopes[0]
            )

    provider.on_call = request_compaction
    result = agent.run("task")
    assert result.get_message() == "done"
    assert agent.root_node.depth == 0
    catalog = provider.catalogs[0]["outputs"]
    assert "mock-skill" not in [item["name"] for item in catalog]
    assert "nested-skill" not in [item["name"] for item in catalog]
    scored_messages = provider.score_requests[0].get_messages()
    assert any("Mock skill body" in str(item.get_message()) for item in scored_messages)
    assert any(
        "Nested skill body" in str(item.get_message()) for item in scored_messages
    )
    assert before_scopes[0][1:]
    executions = result.skill_executions
    assert executions[0].output is not None
    assert executions[0].children[0].output is not None
    assert executions[0].opening_tool_call_id == "open"
    assert executions[0].children[0].opening_tool_call_id == "nested-open"
    validate_request_history(ProviderRequest(runtime_history(agent)))


def test_steering_accepted_while_compacting_is_delivered_after_compaction(make_agent):
    provider = CompactionProvider(
        [tool_reply(call()), AssistantMessage("done")], score=9
    )
    agent = make_agent(provider, Tool(lambda: "body", name="mock-tool"))
    provider.on_call = lambda request: (
        agent.compact_history(agent.root_node) if not provider.requests else None
    )
    provider.on_score = lambda request: agent.steer("new constraint")
    result = agent.run("task")
    assert result.get_message() == "done"
    user_messages = [
        item.get_message()
        for item in provider.requests[-1].get_messages()
        if item.get_role_name() == "user"
    ]
    assert user_messages == ["task", "new constraint"]


@pytest.mark.parametrize("cached", [True, False])
def test_compaction_can_summarize_files_without_persisting_uncached_outputs(
    make_agent, monkeypatch, cached
):
    monkeypatch.setenv("EXMEKH_LIB_TOOL_OUTPUT_MAX_TOKENS", "10")
    original = "start\nimportant middle\nend"
    provider = CompactionProvider(
        [tool_reply(call()), AssistantMessage("done")], score=10
    )
    agent = make_agent(
        provider,
        Tool(
            lambda: original,
            name="mock-tool",
            cache_result=cached,
            background_enabled=False,
        ),
    )
    agent.run("task")
    agent.compact_history(agent.root_node)
    output = compacted_payload(runtime_history(agent)[-1])["outputs"][0]
    assert output["representation"] == "summary"
    assert output["files"]["files"]
    assert original in [
        request.get_messages()[1].get_message() for request in provider.summary_requests
    ]
    records = agent.root_node._runtime._get_tool_cache().get_all()
    assert bool(records) == cached


def test_queued_compaction_releases_managed_wait_and_keeps_work_running(
    make_agent, gates, launch_run
):
    started = Event()
    release = gates()
    launch_run.release_on_cleanup(release)

    def work():
        started.set()
        assert release.wait(5)
        return "completed"

    provider = CompactionProvider(
        [tool_reply(call()), AssistantMessage("done")], score=9
    )
    agent = make_agent(provider, Tool(work, name="mock-tool"))
    running = launch_run(agent)
    assert started.wait(2)
    agent.compact_history(agent.root_node)
    result = running.result(3)
    assert result.get_message() == "done"
    assert not release.is_set()
    assert len(provider.score_requests) == 1
    assert agent.root_node.read_tool_output("output-0").content
    release.set()
    state = agent.root_node.control_tool_execution("origin", wait_seconds=1)
    assert state["status"] == "completed"


def test_idle_failure_keeps_original_and_releases_run_admission(make_agent):
    provider = CompactionProvider(
        [tool_reply(call()), AssistantMessage("first"), AssistantMessage("next")],
        score=10,
    )
    agent = make_agent(provider, Tool(lambda: "body", name="mock-tool"))
    agent.run("task")
    before = [message.message_dump() for message in runtime_history(agent)]
    provider.summary_error = RuntimeError("cannot summarize")
    with pytest.raises(RuntimeError, match="cannot summarize"):
        agent.compact_history(agent.root_node)
    assert [message.message_dump() for message in runtime_history(agent)] == before
    assert agent.run("continue").get_message() == "next"


def test_active_failure_discards_request_without_compacting_the_original(make_agent):
    provider = CompactionProvider(
        [tool_reply(call()), AssistantMessage("next")], score=10
    )
    provider.summary_error = RuntimeError("cannot summarize")
    agent = make_agent(provider, Tool(lambda: "body", name="mock-tool"))
    provider.on_call = lambda request: (
        agent.compact_history(agent.root_node) if not provider.requests else None
    )
    with pytest.raises(RuntimeError, match="cannot summarize"):
        agent.run("task")
    assert runtime_history(agent)[-1].get_name() == "mock-tool"
    provider.on_call = None
    assert agent.run("continue").get_message() == "next"
    assert len(provider.score_requests) == 1


def test_compaction_rejects_uninitialized_node(tmp_path):
    node = Node([tmp_path], name_hds="main")
    with pytest.raises(RuntimeError, match="initialized before compacting"):
        node.compact_history()


def test_deep_agent_requires_explicit_node_and_rejects_foreign_nodes(
    make_agent, tmp_path
):
    provider = CompactionProvider()
    agent = make_agent(provider)
    with pytest.raises(TypeError):
        agent.compact_history()
    with pytest.raises(TypeError, match="Node"):
        agent.compact_history("main")
    foreign = Node([tmp_path], name_hds="foreign")
    with pytest.raises(ValueError, match="belong"):
        agent.compact_history(foreign)


def test_explicit_child_compaction_leaves_root_history_unchanged(make_agent, tmp_path):

    provider = CompactionProvider(
        [
            tool_reply(call()),
            AssistantMessage("root done"),
            tool_reply(call(call_id="child-call")),
            AssistantMessage("child done"),
        ],
        score=9,
    )
    child = Node(
        [_write_nested_capabilities(tmp_path / "child")],
        name_hds="main",
        agent_context_id="child",
        default_capabilities=DefaultCapabilities(
            harness=False,
            overrides={"mock-tool": Tool(lambda: "child body", name="mock-tool")},
        ),
    )
    agent = make_agent(
        provider, Tool(lambda: "root body", name="mock-tool"), children=(child,)
    )
    agent.run("root task")
    child.run("child task")
    root_before = [item.message_dump() for item in runtime_history(agent)]
    child_before = child._runtime._client.get_history()
    agent.compact_history(child)
    assert [item.message_dump() for item in runtime_history(agent)] == root_before
    child_after = child._runtime._client.get_history()
    assert child_after[-1].get_name() == HISTORY_COMPACTION.tool_name
    assert child_before[1].get_message() == child_after[1].get_message() == "child task"
    assert len(provider.score_requests) == 1


def test_history_changed_during_preparation_is_not_overwritten(make_agent):

    provider = CompactionProvider(
        [tool_reply(call()), AssistantMessage("done")], score=9
    )
    agent = make_agent(provider, Tool(lambda: "body", name="mock-tool"))
    agent.run("task")
    provider.on_score = lambda request: agent.root_node.restore_history(
        [UserMessage("replacement")]
    )
    with pytest.raises(RuntimeError, match="History changed"):
        agent.compact_history(agent.root_node)
    assert runtime_history(agent)[-1].get_message() == "replacement"


def test_compaction_queued_at_skill_exit_preserves_the_pending_native_return(
    make_agent,
):
    provider = CompactionProvider(
        [
            tool_reply(call()),
            tool_reply(call("mock-skill", call_id="skill-open")),
            tool_reply(call("exit_agent", {"text": "skill done", "items": []}, "exit")),
            AssistantMessage("done"),
        ],
        score=9,
    )
    captured = []

    def observe(event):
        if (
            isinstance(event, DeepAgentExecutionEvent)
            and event.type is DeepAgentExecutionEventType.TOOL_END
            and event.tool_name == "exit_agent"
        ):
            captured.extend(
                message.message_dump() for message in runtime_history(agent)[-2:]
            )
            agent.compact_history(agent.root_node)

    agent = make_agent(
        provider,
        Tool(lambda: "old root output", name="mock-tool"),
        on_stream=as_agent_handler(observe),
    )
    assert agent.run("task").get_message() == "done"
    assert captured
    continuation = provider.requests[-1]
    assert continuation.get_extra()["agentic"]["operation"] == "return"
    assert [
        message.message_dump() for message in continuation.get_messages()[-2:]
    ] == captured
    assert [item["name"] for item in provider.catalogs[0]["outputs"]] == ["mock-tool"]
    assert agent.root_node.depth == 0
