from typing import cast

import pytest
from openai.types.chat import ChatCompletionChunk

from exmekh_lib.core.models.model_usage import ModelUsage
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.providers.from_openai_chat_completion_stream_mapper import (
    FromOpenAIChatCompletionStreamMapper,
)


def make_chunk(choices: list[dict], usage: dict | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chunk-1",
            "created": 1,
            "model": "model",
            "object": "chat.completion.chunk",
            "choices": choices,
            "usage": usage,
        }
    )


def test_map_chunk_maps_reasoning_response_and_totals_in_order() -> None:
    chunk = make_chunk(
        [
            {
                "index": 0,
                "delta": {
                    "reasoning_content": "thinking",
                    "content": "answer",
                },
                "finish_reason": "stop",
            }
        ],
        {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        },
    )

    result = FromOpenAIChatCompletionStreamMapper.map_chunk(chunk)

    assert result == [
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
    ]


def test_map_chunk_ignores_provider_metadata_without_user_data() -> None:
    chunk = make_chunk(
        [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    )

    assert FromOpenAIChatCompletionStreamMapper.map_chunk(chunk) == []


def test_map_chunk_maps_usage_only_final_chunk() -> None:
    chunk = make_chunk(
        [],
        {
            "prompt_tokens": 7,
            "completion_tokens": 11,
            "total_tokens": 18,
        },
    )

    assert FromOpenAIChatCompletionStreamMapper.map_chunk(chunk) == [
        ProviderStreamEvent(
            ProviderStreamEventType.TOTALS,
            usage=ModelUsage(7, 11, 18),
        )
    ]


def test_map_chunk_rejects_non_chunk() -> None:
    with pytest.raises(TypeError, match="Expected a ChatCompletionChunk"):
        FromOpenAIChatCompletionStreamMapper.map_chunk(
            cast(ChatCompletionChunk, object())
        )
