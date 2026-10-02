from xml.etree import ElementTree as ET

import pytest
from PIL import Image

from qs.core.cache.node_cache import NodeCache
from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.tool import PreformattedToolResult, Tool
from qs.core.models.tool_message import ToolMessage
from qs.core.models.user_message import ImagesUserMessage, UserMessage
from tests.core.agents.node.steering_support import call, tool_reply
from tests.core.agents.node.test_node import SequenceProvider
from tests.core.agents.node.test_steering_history import assert_complete_tool_batches
from tests.event_handler import as_agent_handler


@pytest.fixture
def stored_records(monkeypatch):
    records = []
    original_store = NodeCache.store

    def store(cache, record):
        records.append(record)
        return original_store(cache, record)

    monkeypatch.setattr(NodeCache, "store", store)
    return records


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_skill_steering_is_recoverable_after_compaction(
    make_agent, monkeypatch, stored_records, nested, streaming
):
    records = stored_records
    responses = [tool_reply(call("mock-skill", call_id="open"))]
    if nested:
        responses.append(tool_reply(call("nested-skill", call_id="nested")))
    work_name = "nested-sub-tool" if nested else "mock-sub-tool"
    responses.extend(
        [
            tool_reply(call(work_name, {"text": "work"})),
            tool_reply(call("exit_agent", {"text": "done", "items": []}, "exit")),
        ]
    )
    if nested:
        responses.append(
            tool_reply(call("exit_agent", {"text": "done", "items": []}, "outer-exit"))
        )
    responses.append(AssistantMessage("done"))
    provider = SequenceProvider(responses)

    def observe(event):
        if getattr(event, "tool_name", None) == work_name and event.is_tool_running:
            agent.steer("Remember this correction")
            agent.steer("And this second correction")

    original_call = provider.call

    def steer_during_inference(request):
        response = original_call(request)
        if not streaming and any(
            item.get_name() == work_name for item in response.get_tool_calls() or ()
        ):
            agent.steer("Remember this correction")
            agent.steer("And this second correction")
        return response

    monkeypatch.setattr(provider, "call", steer_during_inference)
    agent = make_agent(
        provider, on_stream=as_agent_handler(observe) if streaming else None
    )
    response = agent.run("start")
    saved = [record for record in records if record.source == "UserMessage"]
    assert [record.result for record in saved] == [
        "Remember this correction",
        "And this second correction",
    ]
    assert len({record.id for record in saved}) == 2
    expected_path = "/mock-skill/nested-skill" if nested else "/mock-skill"
    assert all(record.skill_path == expected_path for record in saved)
    assert [m["content"] for m in response.history if m["role"] == "user"] == ["start"]
    for record in saved:
        read = agent.root_node.read_tool_output(record.id)
        assert record.result in read.content
        assert f"<id>{record.id}</id>" in read.content
        assert "<type>UserMessage</type>" in read.content
        assert record.id in str(response.history)
    for request in provider.requests:
        assert_complete_tool_batches(request.get_messages())


def test_root_steering_is_not_cached(make_agent, monkeypatch, stored_records):
    records = stored_records
    provider = SequenceProvider([AssistantMessage("first"), AssistantMessage("done")])
    original_call = provider.call

    def steer_once(request):
        response = original_call(request)
        if len(provider.requests) == 1:
            agent.steer("root correction")
        return response

    monkeypatch.setattr(provider, "call", steer_once)
    agent = make_agent(provider)
    assert agent.run("start").get_message() == "done"
    assert records == []


def test_reading_saved_steering_restores_images_in_model_request(
    make_agent, stored_records
):
    records = stored_records
    responses = [
        tool_reply(call("mock-skill", call_id="open")),
        tool_reply(call("mock-sub-tool", {"text": "work"})),
        tool_reply(call("exit_agent", {"text": "done", "items": []}, "exit")),
        AssistantMessage("first"),
    ]
    provider = SequenceProvider(responses)
    image = Image.new("RGB", (3, 2), "red")

    def observe(event):
        if (
            getattr(event, "tool_name", None) == "mock-sub-tool"
            and event.is_tool_running
        ):
            agent.steer(["Inspect this", ImagesUserMessage([image])])
            image.putpixel((0, 0), (0, 0, 0))

    agent = make_agent(provider, on_stream=as_agent_handler(observe))
    agent.run("start")
    record = next(record for record in records if record.source == "UserMessage")
    responses.extend(
        [
            tool_reply(
                call("read_tool_output", {"output_id": record.id}, "read"),
                call("mock-tool", {"text": "after"}, "sibling"),
            ),
            AssistantMessage("done"),
        ]
    )
    agent.run("Retrieve the correction")
    messages = provider.requests[-1].get_messages()
    read = next(
        m
        for m in messages
        if isinstance(m, ToolMessage) and m.get_tool_call_id() == "read"
    )
    assert "Inspect this" in read.get_message()
    attachments = read.get_attachments()
    assert attachments is not None
    pixels = [
        part
        for part in attachments.iter_content_parts()
        if isinstance(part, Image.Image)
    ]
    assert len(pixels) == 1
    assert pixels[0].getpixel((0, 0)) == (255, 0, 0)
    assert_complete_tool_batches(messages)


@pytest.mark.parametrize("queue_during_exit", [False, True])
def test_cache_policy_uses_depth_when_steering_is_delivered(
    make_agent, monkeypatch, stored_records, queue_during_exit
):
    responses = [
        tool_reply(call("mock-skill", call_id="open")),
        tool_reply(call("exit_agent", {"text": "done", "items": []}, "exit")),
        AssistantMessage("first"),
    ]
    if queue_during_exit:
        responses.append(AssistantMessage("done"))
    provider = SequenceProvider(responses)
    original_call = provider.call

    def queue_at_boundary(request):
        response = original_call(request)
        if len(provider.requests) == (2 if queue_during_exit else 1):
            agent.steer("boundary correction")
        return response

    monkeypatch.setattr(provider, "call", queue_at_boundary)
    agent = make_agent(provider)
    agent.run("start")
    saved = [record for record in stored_records if record.source == "UserMessage"]
    assert len(saved) == (0 if queue_during_exit else 1)
    if saved:
        assert saved[0].skill_path == "/mock-skill"


@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_mock_workflow_discovers_and_reads_steering_through_history(
    make_agent, monkeypatch, with_images, nested
):
    responses = [tool_reply(call("mock-skill", call_id="open"))]
    if nested:
        responses.append(tool_reply(call("nested-skill", call_id="nested")))
    work_name = "nested-sub-tool" if nested else "mock-sub-tool"
    responses.extend(
        [
            tool_reply(call(work_name, {"text": "MOCK_OUTPUT_TOKEN"}, "work")),
            tool_reply(call("get_execution_history", call_id="history-inside")),
            tool_reply(call("exit_agent", {"text": "done", "items": []}, "exit")),
        ]
    )
    if nested:
        responses.append(
            tool_reply(call("exit_agent", {"text": "done", "items": []}, "outer-exit"))
        )
    responses.extend(
        [
            tool_reply(call("get_execution_history", call_id="history-root")),
            AssistantMessage("read the IDs discovered in history"),
            AssistantMessage("done"),
        ]
    )
    provider = SequenceProvider(responses)
    original_call = provider.call
    history_ids = []
    expected_pixels = Image.new("RGB", (3, 2), "red")
    image = expected_pixels.copy()
    steering = (
        ["STEERING_CORRECTION_TOKEN", ImagesUserMessage([image])]
        if with_images
        else "STEERING_CORRECTION_TOKEN"
    )
    executions = []

    def mock_work(text: str) -> str:
        executions.append(text)
        return text

    def steer_during_work(event):
        if getattr(event, "tool_name", None) == work_name and event.is_tool_running:
            assert event.depth == (2 if nested else 1)
            agent.steer(steering)
            image.putpixel((0, 0), (0, 0, 0))

    def discover_output_ids(request):
        response = original_call(request)
        latest = request.get_messages()[-1]
        if not isinstance(latest, ToolMessage) or latest.get_tool_call_id() not in (
            "history-inside",
            "history-root",
        ):
            return response
        history = ET.fromstring(latest.get_message())
        steering_entries = [
            entry
            for entry in history.findall("call")
            if entry.findtext("name") == "steering_message"
        ]
        assert len(steering_entries) == 1
        steering_id = steering_entries[0].findtext("saved-output-id")
        work_entry = next(
            entry for entry in history.findall("call") if entry.findtext("id") == "work"
        )
        work_id = work_entry.findtext("saved-output-id")
        assert steering_id and work_id and steering_id != work_id
        history_ids.append((steering_id, work_id))
        if latest.get_tool_call_id() == "history-inside":
            assert request.get_tool_choice() == "required"
            return response
        assert history_ids == [history_ids[0], history_ids[0]]
        assert not any(
            "STEERING_CORRECTION_TOKEN" in str(message.get_message())
            for message in request.get_messages()
            if isinstance(message, UserMessage)
        )
        return ProviderResponse(
            tool_reply(
                call("read_tool_output", {"output_id": steering_id}, "read-steering"),
                call("read_tool_output", {"output_id": work_id}, "read-mock"),
            )
        )

    monkeypatch.setattr(provider, "call", discover_output_ids)
    agent = make_agent(
        provider,
        Tool(mock_work, name=work_name, background_enabled=False),
        include={"get_execution_history"},
        on_stream=as_agent_handler(steer_during_work),
    )
    assert agent.run("start").get_message() == "done"
    assert executions == ["MOCK_OUTPUT_TOKEN"]
    assert len(history_ids) == 2
    messages = provider.requests[-1].get_messages()
    reads = {
        message.get_tool_call_id(): message
        for message in messages
        if isinstance(message, ToolMessage)
    }
    steering_result = reads["read-steering"]
    work_result = reads["read-mock"]
    assert "STEERING_CORRECTION_TOKEN" in steering_result.get_message()
    assert "<type>UserMessage</type>" in steering_result.get_message()
    assert f"<id>{history_ids[0][0]}</id>" in steering_result.get_message()
    assert "MOCK_OUTPUT_TOKEN" in work_result.get_message()
    assert "<type>Tool</type>" in work_result.get_message()
    assert f"<id>{history_ids[0][1]}</id>" in work_result.get_message()
    replayed_images = [
        part
        for message in messages
        if isinstance(message, UserMessage)
        for part in message.iter_content_parts()
        if isinstance(part, Image.Image)
    ]
    if with_images:
        assert steering_result.get_attachments() is not None
        assert len(replayed_images) == 1
        assert replayed_images[0].tobytes() == expected_pixels.tobytes()
        assert (
            messages.index(steering_result)
            < messages.index(work_result)
            < len(messages) - 1
        )
    else:
        assert steering_result.get_attachments() is None
        assert replayed_images == []
    for request in provider.requests:
        assert_complete_tool_batches(request.get_messages())


@pytest.mark.parametrize("mode", ["direct", "wait", "background"])
@pytest.mark.parametrize("cache_result", [False, True])
def test_mock_tool_attachments_reach_provider_in_every_execution_mode(
    make_agent, monkeypatch, gates, stored_records, mode, cache_result
):
    release = gates()
    companion = UserMessage(
        ["tool context", ImagesUserMessage([Image.new("RGB", (2, 2), "blue")])]
    )
    executions = []

    def mock_work() -> PreformattedToolResult:
        executions.append("work")
        if mode == "background":
            assert release.wait(3)
        return PreformattedToolResult("completed output", attachments=companion)

    arguments = {} if mode == "direct" else {"execution_mode": mode}
    provider = SequenceProvider(
        [
            tool_reply(call("mock-tool", arguments, "work")),
            AssistantMessage("waiting" if mode == "background" else "done"),
            *([AssistantMessage("done")] if mode == "background" else []),
        ]
    )
    original_call = provider.call

    def complete_work_before_delivery(request):
        response = original_call(request)
        if mode == "background" and len(provider.requests) == 2:
            assert (
                agent.root_node.control_tool_execution("work", wait_seconds=0.01)[
                    "status"
                ]
                == "running"
            )
            release.set()
            assert (
                agent.root_node.control_tool_execution("work", wait_seconds=1)["status"]
                == "completed"
            )
        return response

    monkeypatch.setattr(provider, "call", complete_work_before_delivery)
    agent = make_agent(
        provider,
        Tool(
            mock_work,
            name="mock-tool",
            background_enabled=mode != "direct",
            cache_result=cache_result,
        ),
    )
    assert agent.run("start").get_message() == "done"
    assert executions == ["work"]
    messages = provider.requests[-1].get_messages()
    images = [
        part
        for message in messages
        if isinstance(message, UserMessage)
        for part in message.iter_content_parts()
        if isinstance(part, Image.Image)
    ]
    assert len(images) == 1
    assert images[0].getpixel((0, 0)) == (0, 0, 255)
    saved = [record for record in stored_records if record.name == "mock-tool"]
    assert bool(saved) == cache_result
    if saved:
        assert saved[-1].user_message is not None
    if mode == "background":
        automatic_read = next(
            message
            for message in messages
            if isinstance(message, ToolMessage)
            and message.get_name() == "read_tool_output"
        )
        assert "completed output" in automatic_read.get_message()
        assert automatic_read.get_attachments() is not None
    for request in provider.requests:
        assert_complete_tool_batches(request.get_messages())
