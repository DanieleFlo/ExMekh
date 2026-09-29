from threading import Event, get_ident

import pytest

from qs.core.agents.capabilities.defaults import DefaultCapabilities
from qs.core.agents.node import Node
from qs.core.agents.node.node_runtime_config import NodeRuntimeConfig
from qs.core.agents.node.turn_control import TurnControl
from qs.core.cache.node_cache import NodeCache
from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.tool import Tool
from qs.core.models.tool_message import ToolMessage
from qs.core.models.user_message import TextsUserMessage
from qs.core.tool_calls.execution_manager import ToolExecutionManager
from qs.interfaces.tool_hook import AfterTool, BeforeTool, ToolContext
from tests.core.agents.node.steering_support import (
    blocked_work,
    call,
    tool_reply,
    user_contents,
)
from tests.core.agents.node.test_node import (
    SequenceProvider,
    _write_nested_capabilities,
)
from tests.event_handler import as_agent_handler
from tests.tool_result_support import parse_tool_result


@pytest.mark.parametrize("persistent", [True, False])
@pytest.mark.parametrize("cache_result", [True, False])
def test_steer_releases_wait_and_only_original_run_returns_response(
    make_agent, gates, launch_run, persistent, cache_result, monkeypatch
):
    stored_records = []
    original_store = NodeCache.store

    def store(cache, record):
        stored_records.append(record)
        return original_store(cache, record)

    monkeypatch.setattr(NodeCache, "store", store)
    work, started, release = blocked_work(gates, launch_run)
    provider = SequenceProvider(
        [tool_reply(call()), AssistantMessage("steered answer")]
    )
    agent = make_agent(
        provider,
        Tool(work, name="mock-tool", cache_result=cache_result),
        persistent=persistent,
    )
    result = launch_run(agent)
    assert started.wait(3)
    assert agent.steer("Use the other file") is None
    response = result.result(3)
    assert response.get_message() == "steered answer"
    assert not release.is_set()
    request = provider.requests[-1]
    assert user_contents(request) == ["start", "Use the other file"]
    output = next(m for m in request.get_messages() if isinstance(m, ToolMessage))
    assert output.get_tool_call_id() == "origin"
    assert parse_tool_result(output.get_message())["status"] == "running"
    assert "<response-type>running</response-type>" in output.get_message()
    release.set()
    assert (
        agent.root_node.control_tool_execution("origin", wait_seconds=3)["status"]
        == "completed"
    )
    assert "finished" in agent.root_node.read_tool_output("output-0").content
    assert bool(stored_records) == cache_result


def test_inactive_and_concurrent_calls_are_rejected(make_agent, gates, launch_run):
    work, started, release = blocked_work(gates, launch_run)
    provider = SequenceProvider([tool_reply(call()), AssistantMessage("done")])
    agent = make_agent(provider, Tool(work, name="mock-tool"))
    with pytest.raises(RuntimeError, match="requires an active"):
        agent.steer("too early")
    result = launch_run(agent)
    assert started.wait(3)
    with pytest.raises(RuntimeError, match="already active"):
        agent.run("second run")
    with pytest.raises(TypeError, match="User content"):
        agent.steer(17)
    agent.steer("valid")
    assert result.result(3).get_message() == "done"
    with pytest.raises(RuntimeError, match="requires an active"):
        agent.steer("too late")
    release.set()


def test_uninitialized_node_rejects_steering():
    node = Node(name_hds="main")
    with pytest.raises(RuntimeError, match="requires an active"):
        node.steer("too early")


@pytest.mark.parametrize("initial", ["old answer", ""])
def test_messages_during_inference_are_ordered_and_copied(
    make_agent, gates, launch_run, initial, monkeypatch
):
    entered = Event()
    release = gates()
    launch_run.release_on_cleanup(release)
    provider = SequenceProvider(
        [AssistantMessage(initial), AssistantMessage("updated")]
    )
    original_call = provider.call
    inference_threads = []

    def paused_call(request):
        inference_threads.append(get_ident())
        response = original_call(request)
        if len(provider.requests) == 1:
            entered.set()
            assert release.wait(5)
        return response

    monkeypatch.setattr(provider, "call", paused_call)
    agent = make_agent(provider)
    result = launch_run(agent)
    assert entered.wait(3)
    message = TextsUserMessage(["first"])
    agent.steer(message)
    message.texts.append("later mutation")
    agent.steer("second")
    assert not result.done()
    release.set()
    assert result.result(3).get_message() == "updated"
    messages = user_contents(provider.requests[-1])
    assert messages[1] == TextsUserMessage(["first"])
    assert messages[2] == "second"
    assert len(set(inference_threads)) == 1
    assert inference_threads[0] != get_ident()


def test_steering_releases_all_managed_waits_in_the_batch(
    make_agent, gates, launch_run
):
    work, started, release = blocked_work(gates, launch_run)
    provider = SequenceProvider(
        [
            tool_reply(call(call_id="first"), call(call_id="second")),
            AssistantMessage("updated"),
        ]
    )
    agent = make_agent(provider, Tool(work, name="mock-tool"))
    result = launch_run(agent)
    assert started.wait(3)
    agent.steer("change direction")
    assert result.result(3).get_message() == "updated"
    messages = provider.requests[-1].get_messages()
    outputs = [m for m in messages if isinstance(m, ToolMessage)]
    assert [m.get_tool_call_id() for m in outputs] == ["first", "second"]
    assert all(
        parse_tool_result(m.get_message())["status"] == "running" for m in outputs
    )
    assert messages[-1].get_message() == "change direction"
    assert not release.is_set()


def test_steering_releases_global_control_wait(make_agent, gates, launch_run):
    work, _, release = blocked_work(gates, launch_run)
    control_started = Event()
    provider = SequenceProvider(
        [
            tool_reply(call(arguments={"execution_mode": "background"})),
            tool_reply(
                call("control_tool_execution", {"tool_call_id": "origin"}, "control")
            ),
            AssistantMessage("updated"),
        ]
    )

    def observe(event):
        if (
            getattr(event, "tool_name", None) == "control_tool_execution"
            and event.is_tool_running
        ):
            control_started.set()

    agent = make_agent(
        provider, Tool(work, name="mock-tool"), on_stream=as_agent_handler(observe)
    )
    result = launch_run(agent)
    assert control_started.wait(3)
    agent.steer("continue now")
    assert result.result(3).get_message() == "updated"
    outputs = [
        m for m in provider.requests[-1].get_messages() if isinstance(m, ToolMessage)
    ]
    assert [m.get_tool_call_id() for m in outputs] == ["origin", "control"]
    assert parse_tool_result(outputs[-1].get_message())["status"] == "running"
    assert not release.is_set()


def test_inline_tool_keeps_running_until_safe_boundary(make_agent, gates, launch_run):
    work, started, release = blocked_work(gates, launch_run)
    provider = SequenceProvider([tool_reply(call()), AssistantMessage("updated")])
    agent = make_agent(provider, Tool(work, name="mock-tool", background_enabled=False))
    result = launch_run(agent)
    assert started.wait(3)
    agent.steer("change direction")
    assert not result.done()
    assert len(provider.requests) == 1
    release.set()
    assert result.result(3).get_message() == "updated"
    assert user_contents(provider.requests[-1])[-1] == "change direction"


@pytest.mark.parametrize("nested", [False, True])
def test_steering_survives_skill_compaction(make_agent, gates, launch_run, nested):
    work, started, _ = blocked_work(gates, launch_run)
    responses = [tool_reply(call("mock-skill", call_id="open"))]
    if nested:
        responses.append(tool_reply(call("nested-skill", call_id="nested")))
    responses.extend(
        [
            tool_reply(call("nested-sub-tool" if nested else "mock-sub-tool")),
            tool_reply(call("exit_agent", {"text": "done", "items": []}, "exit")),
        ]
    )
    if nested:
        responses.append(
            tool_reply(call("exit_agent", {"text": "done", "items": []}, "outer-exit"))
        )
    responses.append(AssistantMessage("updated"))
    provider = SequenceProvider(responses)
    agent = make_agent(
        provider, Tool(work, name="nested-sub-tool" if nested else "mock-sub-tool")
    )
    result = launch_run(agent)
    assert started.wait(3)
    agent.steer("Remember this correction")
    response = result.result(3)
    assert response.get_message() == "updated"
    assert [m["content"] for m in response.history if m["role"] == "user"] == [
        "start",
        "Remember this correction",
    ]
    assert all(
        user_contents(r).count("Remember this correction") == 1
        for r in provider.requests[3 if nested else 2 :]
    )
    assert "Remember this correction" in str(response.get_history_tree())
    assert "<header>" in provider.requests[-1].get_messages()[-2].get_message()


def test_steering_queued_during_exit_is_inserted_after_compaction(make_agent):
    provider = SequenceProvider(
        [
            tool_reply(call("mock-skill", call_id="open")),
            tool_reply(call("exit_agent", {"text": "done", "items": []}, "exit")),
            AssistantMessage("updated"),
        ]
    )
    agent = None

    def observe(event):
        if getattr(event, "tool_name", None) == "exit_agent" and event.is_tool_running:
            agent.steer("change direction")

    agent = make_agent(provider, on_stream=as_agent_handler(observe))
    assert agent.run("start").get_message() == "updated"
    assert user_contents(provider.requests[-1]) == ["start", "change direction"]


def test_steering_accepted_at_turn_closure_is_processed(make_agent, monkeypatch):
    provider = SequenceProvider([AssistantMessage("old"), AssistantMessage("updated")])
    agent = make_agent(provider)
    original_close = TurnControl.try_close
    closures = []

    def close(control):
        if not closures:
            agent.steer("last moment")
        closures.append(True)
        return original_close(control)

    monkeypatch.setattr(TurnControl, "try_close", close)
    assert agent.run("start").get_message() == "updated"
    assert len(closures) == 2
    assert user_contents(provider.requests[-1]) == ["start", "last moment"]


def test_provider_failure_discards_pending_input_and_releases_run_guard(
    make_agent, monkeypatch
):
    provider = SequenceProvider([AssistantMessage("next turn")])
    agent = make_agent(provider)
    original_call = provider.call

    def fail(request):
        agent.steer("do not transfer this")
        raise OSError("provider failed")

    monkeypatch.setattr(provider, "call", fail)
    with pytest.raises(OSError, match="provider failed"):
        agent.run("first")
    with pytest.raises(RuntimeError, match="requires an active"):
        agent.steer("inactive")
    monkeypatch.setattr(provider, "call", original_call)
    assert agent.run("second").get_message() == "next turn"
    assert "do not transfer this" not in user_contents(provider.requests[-1])


def test_nodes_do_not_share_steering(make_agent, gates, launch_run):
    first_work, first_started, _ = blocked_work(gates, launch_run)
    second_work, second_started, second_release = blocked_work(gates, launch_run)
    first_provider = SequenceProvider([tool_reply(call()), AssistantMessage("first")])
    second_provider = SequenceProvider([tool_reply(call()), AssistantMessage("second")])
    first = make_agent(first_provider, Tool(first_work, name="mock-tool"))
    second = make_agent(second_provider, Tool(second_work, name="mock-tool"))
    first_result = launch_run(first)
    second_result = launch_run(second)
    assert first_started.wait(3) and second_started.wait(3)
    first.steer("only first")
    assert first_result.result(3).get_message() == "first"
    assert not second_result.done()
    second_release.set()
    assert second_result.result(3).get_message() == "second"
    assert "only first" not in user_contents(second_provider.requests[-1])


def test_steering_does_not_rerun_hooks_and_automatic_delivery_stays_unique(
    make_agent, gates, launch_run, monkeypatch
):
    work, started, release = blocked_work(gates, launch_run)
    hook_calls = []

    class Before(BeforeTool):
        def handle(self, context: ToolContext):
            hook_calls.append("before")

    class After(AfterTool):
        def handle(self, context: ToolContext, result):
            hook_calls.append("after")
            return f"filtered {result}"

    provider = SequenceProvider(
        [
            tool_reply(call()),
            AssistantMessage("intermediate"),
            AssistantMessage("final"),
        ]
    )
    original_call = provider.call

    def finish_during_inference(request):
        response = original_call(request)
        if len(provider.requests) == 2:
            release.set()
            assert (
                agent.root_node.control_tool_execution("origin", wait_seconds=3)[
                    "status"
                ]
                == "completed"
            )
        return response

    monkeypatch.setattr(provider, "call", finish_during_inference)
    agent = make_agent(
        provider, Tool(work, name="mock-tool", before=Before(), after=After())
    )
    result = launch_run(agent)
    assert started.wait(3)
    agent.steer("change direction")
    assert result.result(3).get_message() == "final"
    assert hook_calls == ["before", "after"]
    messages = provider.requests[-1].get_messages()
    outputs = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(outputs) == 2
    assert outputs[0].get_tool_call_id() == "origin"
    assert outputs[1].get_name() == "read_tool_output"
    assert outputs[1].get_tool_call_id() != "origin"
    assert "filtered finished" in outputs[1].get_message()
    assert user_contents(provider.requests[-1]) == ["start", "change direction"]


def test_steering_during_sub_agent_wait_stays_in_root_history(
    tmp_path, make_agent, gates, launch_run
):
    work, started, release = blocked_work(gates, launch_run)
    child = Node(
        [_write_nested_capabilities(tmp_path / "child")],
        name_hds="main",
        agent_context_id="child",
        default_capabilities=DefaultCapabilities(
            harness=False, overrides={"mock-tool": Tool(work, name="mock-tool")}
        ),
    )
    provider = SequenceProvider(
        [
            tool_reply(
                call("call_sub_agent", {"sub_agent": "main", "text": "child input"})
            ),
            tool_reply(call()),
            AssistantMessage("child finished"),
            AssistantMessage("root updated"),
        ]
    )
    agent = make_agent(provider, children=(child,))
    result = launch_run(agent)
    assert started.wait(3)
    agent.steer("root correction")
    assert not result.done()
    assert len(provider.requests) == 2
    release.set()
    assert result.result(3).get_message() == "root updated"
    assert "root correction" not in user_contents(provider.requests[2])
    assert user_contents(provider.requests[-1]) == ["start", "root correction"]


def test_next_turn_restores_ordinary_tool_waits(make_agent, gates, launch_run):
    work, started, release = blocked_work(gates, launch_run)
    provider = SequenceProvider(
        [
            tool_reply(call()),
            AssistantMessage("first"),
            tool_reply(call(call_id="next")),
            AssistantMessage("second"),
        ]
    )
    agent = make_agent(provider, Tool(work, name="mock-tool"))
    first = launch_run(agent)
    assert started.wait(3)
    agent.steer("correction")
    assert first.result(3).get_message() == "first"
    started.clear()
    second = launch_run(agent, "next turn")
    assert started.wait(3)
    assert not second.done()
    release.set()
    assert second.result(3).get_message() == "second"


def test_steering_before_worker_registration_does_not_lose_signal(make_agent):
    provider = SequenceProvider([tool_reply(call()), AssistantMessage("updated")])
    executed = Event()
    release = Event()
    agent = None

    def work():
        executed.set()
        release.wait(3)
        return "done"

    def observe(event):
        if getattr(event, "tool_name", None) == "mock-tool" and event.is_tool_running:
            agent.steer("before registration")

    agent = make_agent(
        provider, Tool(work, name="mock-tool"), on_stream=as_agent_handler(observe)
    )
    try:
        assert agent.run("start").get_message() == "updated"
        assert executed.wait(3)
        assert user_contents(provider.requests[-1]) == ["start", "before registration"]
        output = next(
            m
            for m in provider.requests[-1].get_messages()
            if isinstance(m, ToolMessage)
        )
        assert parse_tool_result(output.get_message())["status"] == "running"
    finally:
        release.set()


def test_completed_tool_keeps_final_result_when_steering_is_accepted(
    make_agent, monkeypatch
):
    provider = SequenceProvider([tool_reply(call()), AssistantMessage("updated")])
    agent = make_agent(provider, Tool(lambda: "completed output", name="mock-tool"))
    original_start = ToolExecutionManager.start

    def complete_before_return(manager, *args, **kwargs):
        snapshot = original_start(manager, *args, **kwargs)
        assert snapshot.is_finished
        agent.steer("completion correction")
        return snapshot

    monkeypatch.setattr(ToolExecutionManager, "start", complete_before_return)
    assert agent.run("start").get_message() == "updated"
    outputs = [
        m for m in provider.requests[-1].get_messages() if isinstance(m, ToolMessage)
    ]
    assert len(outputs) == 1
    assert "completed output" in outputs[0].get_message()
    assert "<response-type>final</response-type>" in outputs[0].get_message()
    assert user_contents(provider.requests[-1]) == ["start", "completion correction"]


def test_steering_during_initial_delivery_precedes_inference(
    make_agent, gates, launch_run
):
    work, _, release = blocked_work(gates, launch_run)
    provider = SequenceProvider(
        [
            tool_reply(call(arguments={"execution_mode": "background"})),
            AssistantMessage("first"),
            AssistantMessage("updated"),
        ]
    )
    agent = None

    def observe(event):
        if (
            getattr(event, "tool_name", None) == "read_tool_output"
            and event.is_tool_running
        ):
            agent.steer("delivery correction")

    agent = make_agent(
        provider, Tool(work, name="mock-tool"), on_stream=as_agent_handler(observe)
    )
    assert agent.run("start").get_message() == "first"
    release.set()
    assert (
        agent.root_node.control_tool_execution("origin", wait_seconds=3)["status"]
        == "completed"
    )
    assert agent.run("next input").get_message() == "updated"
    assert user_contents(provider.requests[-1]) == [
        "start",
        "next input",
        "delivery correction",
    ]


def test_steering_during_session_setup_precedes_first_inference(
    make_agent, monkeypatch
):
    provider = SequenceProvider([AssistantMessage("updated")])
    agent = make_agent(provider, persistent=False)
    original_build = NodeRuntimeConfig.build_system_message

    def build(config):
        message = original_build(config)
        agent.steer("early correction")
        return message

    monkeypatch.setattr(NodeRuntimeConfig, "build_system_message", build)
    assert agent.run("start").get_message() == "updated"
    assert user_contents(provider.requests[0]) == ["start", "early correction"]


def test_turn_closure_rejects_late_input_before_return(make_agent, monkeypatch):
    provider = SequenceProvider([AssistantMessage("done")])
    agent = make_agent(provider)
    original_close = TurnControl.try_close

    def close(control):
        closed = original_close(control)
        assert closed
        with pytest.raises(RuntimeError, match="requires an active"):
            agent.steer("too late")
        with pytest.raises(RuntimeError, match="already active"):
            agent.run("overlapping")
        return closed

    monkeypatch.setattr(TurnControl, "try_close", close)
    assert agent.run("start").get_message() == "done"
    assert len(provider.requests) == 1
