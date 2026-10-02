import pytest
from PIL import Image

from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    ImagesUserMessage,
    TextsUserMessage,
    UserMessage,
)
from tests.core.agents.node.steering_support import call, tool_reply, user_contents
from tests.core.agents.node.test_node import SequenceProvider
from tests.event_handler import as_agent_handler


@pytest.mark.parametrize("persistent", [True, False])
def test_run_preserves_mixed_content_in_history(make_agent, persistent):
    provider = SequenceProvider([AssistantMessage("first"), AssistantMessage("second")])
    agent = make_agent(provider, persistent=persistent)
    first = ["First image", ImagesUserMessage([Image.new("RGB", (2, 3), "red")])]
    second = ["Second image", ImagesUserMessage([Image.new("RGB", (3, 2), "blue")])]

    agent.run(first)
    response = agent.run(second)

    assert user_contents(provider.requests[0]) == [first]
    assert user_contents(provider.requests[1]) == (
        [first, second] if persistent else [second]
    )
    assert response.history[-2] == UserMessage(second).message_dump()


def test_steered_mixed_content_is_copied_and_archived_after_nested_compaction(
    make_agent,
):
    provider = SequenceProvider(
        [
            tool_reply(call("mock-skill", call_id="outer")),
            tool_reply(call("nested-skill", call_id="inner")),
            tool_reply(
                call("exit_agent", {"text": "inner done", "items": []}, "exit-inner")
            ),
            tool_reply(
                call("exit_agent", {"text": "outer done", "items": []}, "exit-outer")
            ),
            AssistantMessage("done"),
        ]
    )
    image = Image.new("RGB", (2, 3), "red")
    pixels = image.tobytes()
    document = DocumentUserMessage(b"file nonce", "notes", "txt", "text/plain")
    content = [
        "Inspect this image.",
        ImagesUserMessage([image]),
        TextsUserMessage(["Additional guidance"]),
        FilesUserMessage([document]),
    ]

    def observe(event):
        if (
            getattr(event, "tool_name", None) == "nested-skill"
            and event.is_tool_running
        ):
            agent.steer(content)
            content[0] = "Changed after queuing"
            content[2].texts.clear()
            document.file = b"changed"
            image.paste("blue", (0, 0, 2, 3))
            content[1].images.clear()
            content.clear()
            image.close()

    agent = make_agent(provider, on_stream=as_agent_handler(observe))
    response = agent.run("start")

    assert response.get_message() == "done"
    delivered = [
        message
        for message in user_contents(provider.requests[2])
        if isinstance(message, list)
    ]
    assert len(delivered) == 1
    assert delivered[0][0] == "Inspect this image."
    assert len(delivered[0][1].images) == 1
    assert delivered[0][1].images[0].tobytes() == pixels
    assert delivered[0][2].texts == ["Additional guidance"]
    assert delivered[0][3].documents[0].file == b"file nonce"
    assert not any(
        isinstance(message, list)
        for request in provider.requests[3:]
        for message in user_contents(request)
    )
    archived = [
        message.get_message()
        for message in response.skill_executions[0].children[0].history
        if isinstance(message, UserMessage) and isinstance(message.get_message(), list)
    ]
    assert len(archived) == 1
    assert archived[0][0] == "Inspect this image."
    assert archived[0][1].images[0].tobytes() == pixels
    assert archived[0][2].texts == ["Additional guidance"]
    assert archived[0][3].documents[0].file == b"file nonce"
    assert UserMessage(archived[0]).message_dump() not in response.history
