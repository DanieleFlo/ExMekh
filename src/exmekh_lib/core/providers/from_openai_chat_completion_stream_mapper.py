from __future__ import annotations

from typing import ClassVar

from openai.types.chat import ChatCompletionChunk

from exmekh_lib.core.models.model_usage import ModelUsage
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)


class FromOpenAIChatCompletionStreamMapper:
    _REASONING_FIELDS: ClassVar[tuple[str, ...]] = (
        "reasoning",
        "reasoning_content",
        "thinking",
    )

    @classmethod
    def map_chunk(cls, chunk: ChatCompletionChunk) -> list[ProviderStreamEvent]:
        if not isinstance(chunk, ChatCompletionChunk):
            raise TypeError("Expected a ChatCompletionChunk instance")

        events: list[ProviderStreamEvent] = []
        for choice in chunk.choices:
            reasoning = cls._extract_reasoning(choice.delta)
            if reasoning:
                events.append(
                    ProviderStreamEvent(
                        ProviderStreamEventType.REASONING,
                        content=reasoning,
                    )
                )

            content = choice.delta.content
            if content:
                events.append(
                    ProviderStreamEvent(
                        ProviderStreamEventType.RESPONSE,
                        content=content,
                    )
                )

        if chunk.usage is not None:
            events.append(
                ProviderStreamEvent(
                    ProviderStreamEventType.TOTALS,
                    usage=ModelUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    ),
                )
            )

        return events

    @classmethod
    def _extract_reasoning(cls, delta: object) -> str | None:
        for field in cls._REASONING_FIELDS:
            value = getattr(delta, field, None)
            if isinstance(value, str) and value:
                return value
        return None
