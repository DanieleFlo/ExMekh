from typing import cast

import pytest

from qs.core.models.model_usage import ModelUsage
from qs.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)


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
