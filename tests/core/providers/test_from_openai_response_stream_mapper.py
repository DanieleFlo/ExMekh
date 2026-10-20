import pytest
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)

from exmekh_lib.core.models.provider_stream_event import ProviderStreamEventType
from exmekh_lib.core.providers.from_openai_response_stream_mapper import (
    FromOpenAIResponseStreamMapper,
)


def test_map_event_maps_text_and_reasoning_deltas() -> None:
    text_event = ResponseTextDeltaEvent.model_construct(delta="answer")
    reasoning_event = ResponseReasoningTextDeltaEvent.model_construct(delta="think")

    text = FromOpenAIResponseStreamMapper.map_event(text_event)
    reasoning = FromOpenAIResponseStreamMapper.map_event(reasoning_event)

    assert text[0].type is ProviderStreamEventType.RESPONSE
    assert text[0].content == "answer"
    assert reasoning[0].type is ProviderStreamEventType.REASONING
    assert reasoning[0].content == "think"


def test_map_event_maps_reasoning_summary_delta() -> None:
    event = ResponseReasoningSummaryTextDeltaEvent.model_construct(delta="summary")

    mapped = FromOpenAIResponseStreamMapper.map_event(event)

    assert mapped[0].type is ProviderStreamEventType.REASONING
    assert mapped[0].content == "summary"


def test_map_event_maps_completed_usage() -> None:
    response = Response.model_construct(
        usage=ResponseUsage.model_construct(
            input_tokens=2,
            output_tokens=3,
            total_tokens=5,
        )
    )
    event = ResponseCompletedEvent.model_construct(response=response)

    mapped = FromOpenAIResponseStreamMapper.map_event(event)

    assert mapped[0].type is ProviderStreamEventType.TOTALS
    assert mapped[0].usage.total_tokens == 5


def test_map_event_ignores_unrelated_events() -> None:
    assert FromOpenAIResponseStreamMapper.map_event(object()) == []


@pytest.mark.parametrize("event_type", [ResponseFailedEvent, ResponseIncompleteEvent])
def test_terminal_failure_keeps_usage_visible(event_type):
    response = Response.model_construct(
        usage=ResponseUsage.model_construct(
            input_tokens=2, output_tokens=3, total_tokens=5
        )
    )
    mapped = FromOpenAIResponseStreamMapper.map_event(
        event_type.model_construct(response=response)
    )
    assert mapped[0].type is ProviderStreamEventType.TOTALS
    assert mapped[0].usage.total_tokens == 5
