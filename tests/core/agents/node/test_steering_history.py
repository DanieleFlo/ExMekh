from collections import Counter
from concurrent.futures import Future
from threading import Barrier, Event, Thread

import pytest

from exmekh_lib.core.agents.deep_agent.stream_event import DeepAgentExecutionEvent
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import ErrorUserMessage, UserMessage
from tests.core.agents.node.steering_support import call, tool_reply
from tests.core.agents.node.test_node import SequenceProvider
from tests.event_handler import as_agent_handler


def assert_complete_tool_batches(messages, *, terminal_exit_id=None):
    """Check the protocol independently of MessageHistory's role warnings."""
    pending = {}
    for index, message in enumerate(messages):
        role = message.get_role_name()
        if index:
            previous_role = messages[index - 1].get_role_name()
            if previous_role == "system":
                assert role == "user", (
                    "The system prompt must be followed by user input"
                )
            assert (previous_role, role) != ("assistant", "assistant"), (
                "Consecutive assistant turns without input"
            )
        if isinstance(message, ToolMessage):
            call_id = message.get_tool_call_id()
            assert call_id in pending, (
                f"Orphan or duplicate result at {index}: {call_id}"
            )
            assert pending.pop(call_id) == message.get_name()
            continue

        assert not pending, f"{role} at {index} interrupts unanswered calls: {pending}"
        assert role in {"system", "user", "assistant"}
        if role == "system":
            assert index == 0
        if isinstance(message, AssistantMessage):
            calls = message.get_tool_calls() or ()
            pending = {tool.id: tool.get_name() for tool in calls}
            assert len(pending) == len(calls), "Duplicate IDs within an assistant batch"

    # A closed skill's archived transcript ends with its native exit call;
    # the corresponding result is the skill return in its caller's history.
    expected = {} if terminal_exit_id is None else {terminal_exit_id: "exit_agent"}
    assert pending == expected, f"Unexpected unanswered calls at end: {pending}"


def history_positions(messages):
    positions = []
    for message in messages:
        if isinstance(message, AssistantMessage) and message.get_tool_calls():
            calls = ",".join(
                f"{tool.get_name()}#{tool.id}" for tool in message.get_tool_calls()
            )
            positions.append(f"assistant:{calls}")
        elif isinstance(message, ToolMessage):
            positions.append(f"tool:{message.get_name()}#{message.get_tool_call_id()}")
        elif message.get_role_name() == "system":
            positions.append("system")
        else:
            positions.append(f"{message.get_role_name()}:{message.get_message()}")
    return positions


def steering_messages(messages):
    return [
        message.get_message()
        for message in messages
        if isinstance(message, UserMessage)
        and not isinstance(message, ErrorUserMessage)
        and message.get_message() != "start"
    ]


def _archived_steering_messages(execution):
    yield from steering_messages(execution.history)
    for child in execution.children:
        yield from _archived_steering_messages(child)


def assert_final_history(provider, response, expected_steering):
    messages = [*provider.requests[-1].get_messages(), response]
    assert [message.message_dump() for message in messages] == response.history
    assert_complete_tool_batches(messages)
    delivered = steering_messages(messages)
    for execution in response.skill_executions:
        delivered.extend(_archived_steering_messages(execution))
    assert Counter(delivered) == Counter(expected_steering)
    return messages


def assert_archived_skill(execution, exit_id):
    assert_complete_tool_batches(execution.history, terminal_exit_id=exit_id)


def inline_work(text: str) -> str:
    return f"done: {text}"


def work_call(name, call_id):
    return call(name, {"text": call_id}, call_id)


def exit_call(call_id):
    return call("exit_agent", {"text": call_id, "items": []}, call_id)


@pytest.mark.parametrize(
    "messages",
    [
        [
            tool_reply(call()),
            UserMessage("interleaved"),
            ToolMessage("done", "origin", "mock-tool"),
        ],
        [tool_reply(call()), ToolMessage("done", "wrong-id", "mock-tool")],
        [tool_reply(call()), ToolMessage("done", "origin", "wrong-name")],
        [
            tool_reply(call()),
            ToolMessage("done", "origin", "mock-tool"),
            ToolMessage("duplicate", "origin", "mock-tool"),
        ],
        [tool_reply(call()), AssistantMessage("premature inference")],
        [SystemMessage("rules"), UserMessage("start"), tool_reply(call())],
        [SystemMessage("rules"), AssistantMessage("without user input")],
        [AssistantMessage("first answer"), AssistantMessage("concurrent answer")],
    ],
    ids=[
        "user-between-call-and-result",
        "orphan-result",
        "wrong-tool-name",
        "duplicate-result",
        "assistant-before-result",
        "missing-result",
        "no-initial-user-message",
        "consecutive-assistant-turns",
    ],
)
def test_history_validator_rejects_broken_tool_chains(messages):
    with pytest.raises(AssertionError):
        assert_complete_tool_batches(messages)


@pytest.mark.parametrize("persistent", [True, False])
@pytest.mark.parametrize("burst_size", [1, 3])
def test_steering_bursts_at_every_boundary_preserve_nested_history(
    make_agent, monkeypatch, caplog, persistent, burst_size
):
    provider = SequenceProvider(
        [
            tool_reply(
                call("mock-skill", call_id="outer"), work_call("mock-tool", "root-work")
            ),
            tool_reply(
                call("nested-skill", call_id="nested"),
                work_call("shared-tool", "outer-work"),
            ),
            tool_reply(
                work_call("nested-sub-tool", "nested-a"),
                work_call("shared-tool", "nested-b"),
            ),
            tool_reply(
                exit_call("nested-exit"), work_call("nested-sub-tool", "nested-last")
            ),
            # Reuse the opening ID after closure to challenge instance identification.
            tool_reply(
                call("nested-skill", call_id="nested"),
                work_call("mock-sub-tool", "outer-after"),
            ),
            tool_reply(exit_call("nested-second-exit")),
            tool_reply(exit_call("outer-exit"), work_call("shared-tool", "outer-last")),
            AssistantMessage("superseded answer"),
            AssistantMessage("final answer"),
        ]
    )
    accepted = []
    request_inputs = []
    event_scopes = []
    agent = None

    def send_burst(label):
        for index in range(burst_size):
            text = f"{label}/{index}: <&> {{unchanged user text}}"
            agent.steer(text)
            accepted.append(text)

    def observe(event):
        if isinstance(event, DeepAgentExecutionEvent):
            phase = "start" if event.is_tool_running else "end"
            event_scopes.append(event.skill_path)
            send_burst(f"{len(event_scopes)}:{phase}:{event.tool_call_id}")

    original_call = provider.call

    def malicious_provider(request):
        assert_complete_tool_batches(request.get_messages())
        delivered = steering_messages(request.get_messages())
        assert set(delivered) <= set(accepted)
        assert len(delivered) == len(set(delivered))
        request_inputs.append(delivered)
        # Inject while inference is in flight, including the first final answer.
        if len(provider.requests) < 8:
            send_burst(f"inference-{len(provider.requests)}")
        return original_call(request)

    monkeypatch.setattr(provider, "call", malicious_provider)
    agent = make_agent(
        provider,
        *[
            Tool(inline_work, name=name, background_enabled=False)
            for name in ("mock-tool", "shared-tool", "mock-sub-tool", "nested-sub-tool")
        ],
        persistent=persistent,
        on_stream=as_agent_handler(observe),
    )
    response = agent.run("start")
    assert response.get_message() == "final answer"
    assert len(provider.requests) == 9
    assert {"/", "/mock-skill", "/mock-skill/nested-skill"} <= set(event_scopes)
    final_messages = assert_final_history(provider, response, accepted)
    assert history_positions(final_messages)[:5] == [
        "system",
        "user:start",
        "assistant:mock-skill#outer,mock-tool#root-work",
        "tool:mock-tool#root-work",
        "tool:mock-skill#outer",
    ]
    assert set(accepted) == set().union(*request_inputs)
    outer = response.skill_executions[0]
    assert len(outer.children) == 2
    assert_archived_skill(outer, "outer-exit")
    assert_archived_skill(outer.children[0], "nested-exit")
    assert_archived_skill(outer.children[1], "nested-second-exit")
    assert steering_messages(outer.history)
    assert steering_messages(outer.children[0].history)
    assert steering_messages(outer.children[1].history)
    assert "Wrong message sequence" not in caplog.text


@pytest.mark.parametrize("burst_size", [1, 4])
def test_steering_during_deferred_sibling_skills_keeps_batches_complete(
    make_agent, monkeypatch, caplog, burst_size
):
    provider = SequenceProvider(
        [
            tool_reply(call("mock-skill", call_id="outer")),
            tool_reply(
                call("nested-skill", call_id="first"),
                call("nested-skill", call_id="second"),
                work_call("shared-tool", "parent-work"),
            ),
            tool_reply(exit_call("first-exit")),
            AssistantMessage("intermediate return"),
            tool_reply(exit_call("second-exit")),
            tool_reply(exit_call("outer-exit")),
            AssistantMessage("before pending steering"),
            AssistantMessage("done"),
        ]
    )
    accepted = []
    seen_events = []
    agent = None

    def observe(event):
        if isinstance(event, DeepAgentExecutionEvent):
            seen_events.append(event)
            for index in range(burst_size):
                text = f"event-{len(seen_events)}:{index}"
                agent.steer(text)
                accepted.append(text)

    original_call = provider.call

    def check_request(request):
        assert_complete_tool_batches(request.get_messages())
        assert set(steering_messages(request.get_messages())) <= set(accepted)
        return original_call(request)

    monkeypatch.setattr(provider, "call", check_request)
    agent = make_agent(
        provider,
        Tool(inline_work, name="shared-tool", background_enabled=False),
        on_stream=as_agent_handler(observe),
    )
    response = agent.run("start")
    assert len(provider.requests) == 8
    assert_final_history(provider, response, accepted)
    assert set(accepted) == {
        text
        for request in provider.requests
        for text in steering_messages(request.get_messages())
    }
    outer = response.skill_executions[0]
    assert [child.opening_tool_call_id for child in outer.children] == [
        "first",
        "second",
    ]
    assert_archived_skill(outer, "outer-exit")
    assert_archived_skill(outer.children[0], "first-exit")
    assert_archived_skill(outer.children[1], "second-exit")
    second_inference = provider.requests[4].get_messages()
    positions = history_positions(second_inference)
    assert positions.index("tool:nested-skill#first") < positions.index(
        "assistant:nested-skill#second"
    )
    assert positions.index("assistant:nested-skill#second") < positions.index(
        "tool:nested-skill#second"
    )
    assert "Wrong message sequence" not in caplog.text


def test_concurrent_steering_flood_during_nested_background_batch_preserves_history(
    make_agent, gates, launch_run, monkeypatch, caplog
):
    provider = SequenceProvider(
        [
            tool_reply(call("mock-skill", call_id="outer")),
            tool_reply(call("nested-skill", call_id="nested")),
            tool_reply(
                call("nested-sub-tool", call_id="first-work"),
                call("nested-sub-tool", call_id="second-work"),
            ),
            tool_reply(exit_call("nested-exit")),
            tool_reply(exit_call("outer-exit")),
            AssistantMessage("before pending steering"),
            AssistantMessage("done"),
        ]
    )
    work_started = Event()
    work_release = gates()
    inference_entered = Event()
    sending_finished = gates()
    launch_run.release_on_cleanup(work_release)
    launch_run.release_on_cleanup(sending_finished)

    def background_work():
        work_started.set()
        assert work_release.wait(5)
        return "finished"

    original_call = provider.call

    def paused_provider(request):
        assert_complete_tool_batches(request.get_messages())
        if len(provider.requests) == 3:
            inference_entered.set()
            assert sending_finished.wait(5)
        return original_call(request)

    monkeypatch.setattr(provider, "call", paused_provider)
    agent = make_agent(provider, Tool(background_work, name="nested-sub-tool"))
    result = launch_run(agent)
    assert work_started.wait(3)
    producer_count, messages_per_producer = 4, 12
    ready = Barrier(producer_count + 1)
    producers = []
    producer_results = []

    def send(producer, outcome):
        try:
            ready.wait(3)
            for index in range(messages_per_producer):
                if index == 1:
                    assert inference_entered.wait(3)
                agent.steer(f"producer-{producer}/{index}")
            outcome.set_result(None)
        except BaseException as error:
            outcome.set_exception(error)

    try:
        for producer in range(producer_count):
            outcome = Future()
            thread = Thread(target=send, args=(producer, outcome), daemon=True)
            producers.append(thread)
            producer_results.append(outcome)
            thread.start()
        ready.wait(3)
        for outcome in producer_results:
            outcome.result(3)
        sending_finished.set()
        response = result.result(3)
        assert response.get_message() == "done"
        final_messages = [*provider.requests[-1].get_messages(), response]
        assert_complete_tool_batches(final_messages)
        assert [
            message.message_dump() for message in final_messages
        ] == response.history
        expected = [
            f"producer-{producer}/{index}"
            for producer in range(producer_count)
            for index in range(messages_per_producer)
        ]
        observed = [
            text
            for request in provider.requests
            for text in steering_messages(request.get_messages())
        ]
        assert set(observed) == set(expected)
        first_delivery_order = list(dict.fromkeys(observed))
        for producer in range(producer_count):
            assert [
                text
                for text in first_delivery_order
                if text.startswith(f"producer-{producer}/")
            ] == [
                f"producer-{producer}/{index}" for index in range(messages_per_producer)
            ]
        assert_final_history(provider, response, expected)
        outer = response.skill_executions[0]
        nested = outer.children[0]
        assert_archived_skill(outer, "outer-exit")
        assert_archived_skill(nested, "nested-exit")
        nested_positions = history_positions(nested.history)
        assert (
            "assistant:nested-sub-tool#first-work,nested-sub-tool#second-work"
            in nested_positions
        )
        assert nested_positions.index("tool:nested-sub-tool#second-work") < next(
            index
            for index, value in enumerate(nested_positions)
            if value.startswith("user:producer-")
        )
        assert not work_release.is_set()
        assert "Wrong message sequence" not in caplog.text
    finally:
        sending_finished.set()
        work_release.set()
        ready.abort()
        for thread in producers:
            thread.join(3)
            assert not thread.is_alive()


def test_steering_on_automatic_output_delivery_preserves_nested_tool_pairs(
    make_agent, gates, launch_run, monkeypatch, caplog
):
    provider = SequenceProvider(
        [
            tool_reply(call("mock-skill", call_id="outer")),
            tool_reply(call("nested-skill", call_id="nested")),
            tool_reply(
                call("nested-sub-tool", {"execution_mode": "background"}, "background")
            ),
            tool_reply(work_call("shared-tool", "work")),
            tool_reply(exit_call("nested-exit")),
            tool_reply(exit_call("outer-exit")),
            AssistantMessage("before pending steering"),
            AssistantMessage("done"),
        ]
    )
    work_started = Event()
    work_release = gates()
    launch_run.release_on_cleanup(work_release)
    accepted = []
    delivery_events = []
    agent = None

    def background_work():
        work_started.set()
        assert work_release.wait(5)
        return "background result"

    def observe(event):
        if isinstance(event, DeepAgentExecutionEvent):
            phase = "start" if event.is_tool_running else "end"
            if event.tool_name == "read_tool_output":
                delivery_events.append(event)
            for index in range(2):
                text = f"{len(accepted)}:{event.tool_name}:{phase}/{index}"
                agent.steer(text)
                accepted.append(text)

    original_call = provider.call

    def complete_background_work(request):
        assert_complete_tool_batches(request.get_messages())
        assert set(steering_messages(request.get_messages())) <= set(accepted)
        if len(provider.requests) == 3:
            assert work_started.wait(3)
            work_release.set()
            assert (
                agent.root_node.control_tool_execution("background", wait_seconds=3)[
                    "status"
                ]
                == "completed"
            )
        return original_call(request)

    monkeypatch.setattr(provider, "call", complete_background_work)
    agent = make_agent(
        provider,
        Tool(background_work, name="nested-sub-tool"),
        Tool(inline_work, name="shared-tool", background_enabled=False),
        on_stream=as_agent_handler(observe),
    )
    response = agent.run("start")
    assert len(delivery_events) == 2
    assert len({event.tool_call_id for event in delivery_events}) == 1
    assert_final_history(provider, response, accepted)
    nested = response.skill_executions[0].children[0]
    assert_archived_skill(response.skill_executions[0], "outer-exit")
    assert_archived_skill(nested, "nested-exit")
    positions = history_positions(nested.history)
    delivery_id = delivery_events[0].tool_call_id
    call_position = positions.index(f"assistant:read_tool_output#{delivery_id}")
    result_position = positions.index(f"tool:read_tool_output#{delivery_id}")
    assert result_position == call_position + 1
    delivery_steering = [text for text in accepted if ":read_tool_output:" in text]
    assert all(
        positions.index(f"user:{text}") > result_position for text in delivery_steering
    )
    assert "Wrong message sequence" not in caplog.text


def test_steering_with_invalid_controls_and_empty_nested_response_preserves_history(
    make_agent, monkeypatch, caplog
):
    provider = SequenceProvider(
        [
            tool_reply(call("mock-skill", call_id="outer")),
            tool_reply(call("nested-skill", call_id="nested")),
            tool_reply(
                call(
                    "control_tool_execution",
                    {"tool_call_id": "missing-job"},
                    "bad-control",
                )
            ),
            AssistantMessage(""),
            tool_reply(
                exit_call("nested-exit"),
                call("read_tool_output", {"output_id": "missing-output"}, "bad-read"),
            ),
            tool_reply(exit_call("outer-exit")),
            AssistantMessage("before pending steering"),
            AssistantMessage("recovered"),
        ]
    )
    accepted = []
    agent = None

    def observe(event):
        if isinstance(event, DeepAgentExecutionEvent):
            phase = "start" if event.is_tool_running else "end"
            text = f"correction-{len(accepted)}:{event.tool_call_id}:{phase}"
            agent.steer(text)
            accepted.append(text)

    original_call = provider.call

    def check_request(request):
        assert_complete_tool_batches(request.get_messages())
        assert set(steering_messages(request.get_messages())) <= set(accepted)
        return original_call(request)

    monkeypatch.setattr(provider, "call", check_request)
    agent = make_agent(provider, on_stream=as_agent_handler(observe))
    response = agent.run("start")
    assert len(provider.requests) == 8
    assert any(
        isinstance(message, ErrorUserMessage)
        for message in provider.requests[4].get_messages()
    )
    final_messages = assert_final_history(provider, response, accepted)
    assert not any(isinstance(message, ErrorUserMessage) for message in final_messages)
    outer = response.skill_executions[0]
    nested = outer.children[0]
    assert_archived_skill(outer, "outer-exit")
    assert_archived_skill(nested, "nested-exit")
    positions = history_positions(nested.history)
    assert positions.index("tool:control_tool_execution#bad-control") < positions.index(
        "assistant:"
    )
    assert positions.index("tool:read_tool_output#bad-read") > positions.index(
        "assistant:exit_agent#nested-exit,read_tool_output#bad-read"
    )
    assert "Wrong message sequence" not in caplog.text
