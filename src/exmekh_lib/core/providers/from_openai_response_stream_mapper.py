from __future__ import annotations

from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)

from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.providers.from_openai_response_mapper import (
    FromOpenAIResponseMapper,
)


class FromOpenAIResponseStreamMapper:
    @staticmethod
    def map_event(event: ResponseStreamEvent) -> list[ProviderStreamEvent]:
        if isinstance(event, ResponseTextDeltaEvent):
            return [
                ProviderStreamEvent(
                    ProviderStreamEventType.RESPONSE,
                    content=event.delta,
                )
            ]

        if isinstance(
            event,
            (ResponseReasoningTextDeltaEvent, ResponseReasoningSummaryTextDeltaEvent),
        ):
            return [
                ProviderStreamEvent(
                    ProviderStreamEventType.REASONING,
                    content=event.delta,
                )
            ]

        if isinstance(
            event,
            (ResponseCompletedEvent, ResponseFailedEvent, ResponseIncompleteEvent),
        ):
            return [
                ProviderStreamEvent(
                    ProviderStreamEventType.TOTALS,
                    usage=FromOpenAIResponseMapper.extract_usage(event.response),
                )
            ]

        return []
