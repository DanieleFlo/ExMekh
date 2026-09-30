from typing import cast

import pytest

from qs.core.models.model_usage import ModelUsage
from qs.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from qs.core.models.user_message import TextsUserMessage, UserMessage


@pytest.mark.parametrize(
    "event",
    [
        ProviderStreamEvent(ProviderStreamEventType.START),
        ProviderStreamEvent(
            ProviderStreamEventType.REASONING,
            content="thinking",
        ),
        ProviderStreamEvent(
            ProviderStreamEventType.RESPONSE,
            content="answer",
        ),
        ProviderStreamEvent(
            ProviderStreamEventType.TOTALS,
            usage=ModelUsage(2, 3, 5),
        ),
        ProviderStreamEvent(ProviderStreamEventType.END),
        ProviderStreamEvent(
            ProviderStreamEventType.STEERING_ADDED, message=UserMessage("correction")
        ),
    ],
)
def test_stream_event_accepts_provider_neutral_event_shapes(event) -> None:
    assert isinstance(event, ProviderStreamEvent)


@pytest.mark.parametrize(
    "arguments",
    [
        {"type": "start"},
        {"type": ProviderStreamEventType.RESPONSE},
        {"type": ProviderStreamEventType.START, "content": "invalid"},
        {"type": ProviderStreamEventType.TOTALS},
        {
            "type": ProviderStreamEventType.END,
            "usage": ModelUsage(2, 3, 5),
        },
    ],
)
def test_stream_event_rejects_inconsistent_event_shapes(arguments) -> None:
    with pytest.raises((TypeError, ValueError)):
        ProviderStreamEvent(**arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "type": ProviderStreamEventType.RESPONSE,
            "content": cast(str, 1),
        },
        {
            "type": ProviderStreamEventType.TOTALS,
            "usage": cast(ModelUsage, object()),
        },
    ],
)
def test_stream_event_rejects_invalid_payload_types(arguments) -> None:
    with pytest.raises(TypeError):
        ProviderStreamEvent(**arguments)


def test_steering_event_copies_the_full_user_message() -> None:
    content = TextsUserMessage(["first", "second"])
    message = UserMessage(content, name="user")
    event = ProviderStreamEvent(ProviderStreamEventType.STEERING_ADDED, message=message)
    content.texts.append("later mutation")
    assert event.message is not message
    assert event.message is not None
    assert event.message.get_message() == TextsUserMessage(["first", "second"])
    assert event.message.get_name() == "user"


@pytest.mark.parametrize(
    "arguments",
    [
        {"type": ProviderStreamEventType.STEERING_ADDED},
        {"type": ProviderStreamEventType.STEERING_ADDED, "message": "correction"},
        {"type": ProviderStreamEventType.START, "message": UserMessage("correction")},
        {
            "type": ProviderStreamEventType.STEERING_ADDED,
            "message": UserMessage("x"),
            "content": "x",
        },
        {
            "type": ProviderStreamEventType.STEERING_ADDED,
            "message": UserMessage("x"),
            "usage": ModelUsage(2, 3, 5),
        },
    ],
)
def test_stream_event_rejects_invalid_steering_payloads(arguments) -> None:
    with pytest.raises((TypeError, ValueError)):
        ProviderStreamEvent(**arguments)
