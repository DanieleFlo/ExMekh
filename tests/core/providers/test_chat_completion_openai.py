from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel, ValidationError

from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.model_thinking import ModelThinking
from qs.core.models.model_usage import ModelUsage
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from qs.core.models.reasoning_effort import ReasoningEffort, ReasoningEffortLMX
from qs.core.models.structured_assistant_message import StructuredAssistantMessage
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.core.models.system_message import SystemMessage
from qs.core.providers.chat_completion_openai import ChatCompletionOpenAI
from qs.core.providers.from_openai_chat_completion_mapper import (
    FromOpenAIChatCompletionMapper,
)
from qs.core.providers.openai_chat_completion_response_format_mapper import (
    OpenAIChatCompletionResponseFormatMapper,
)
from qs.core.providers.to_openai_chat_completion_request_mapper import (
    ToOpenAIChatCompletionRequestMapper,
)


class StructuredAnswer(BaseModel):
    value: int


def make_stream_chunk(
    choices: list[dict],
    usage: dict | None = None,
) -> ChatCompletionChunk:
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


def make_stream(content: str) -> list[ChatCompletionChunk]:
    return [
        make_stream_chunk(
            [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": "thinking",
                    },
                    "finish_reason": None,
                }
            ]
        ),
        make_stream_chunk(
            [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": "stop",
                }
            ]
        ),
        make_stream_chunk(
            [],
            {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
        ),
    ]


def test_init_preserves_client_and_model(openai_client) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")

    assert completion._client is openai_client
    assert completion._model == "model"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("client", object(), "Client must be an instance of OpenAI"),
        ("model", object(), "Model must be a string"),
        ("reasoning_effort", object(), "Reasoning effort must be"),
    ],
)
def test_init_rejects_invalid_inputs(
    openai_client, field: str, value: object, message: str
) -> None:
    arguments = {
        "client": openai_client,
        "model": "model",
        "reasoning_effort": None,
    }
    arguments[field] = value

    with pytest.raises(TypeError, match=message):
        ChatCompletionOpenAI(**arguments)


def test_call_executes_and_parses_provider_response(openai_client) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")
    request = MagicMock(spec=ProviderRequest)
    provider_response = object()
    parsed_response = ProviderResponse(AssistantMessage("answer"))

    with (
        patch.object(
            completion, "_execute_request", return_value=provider_response
        ) as execute,
        patch.object(
            completion, "_parse_response", return_value=parsed_response
        ) as parse,
    ):
        result = completion.call(request)

    assert result is parsed_response
    execute.assert_called_once_with(request)
    parse.assert_called_once_with(provider_response, request)


def test_call_streams_mapped_events_and_returns_accumulated_response(
    openai_client,
) -> None:
    events = []
    completion = ChatCompletionOpenAI(openai_client, "model")
    request = ProviderRequest(
        [SystemMessage("rules")],
        stream=True,
        on_event=events.append,
    )
    openai_client.chat.completions.create.return_value = make_stream("answer")

    result = completion.call(request)

    assert result.get_message() == "answer"
    assert result.thinking == ModelThinking("thinking")
    assert result.usage == ModelUsage(2, 3, 5)
    assert events == [
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
    ]


def test_consume_stream_closes_empty_stream_and_omits_end_event() -> None:
    events = []
    request = ProviderRequest(
        [SystemMessage("rules")], stream=True, on_event=events.append
    )
    stream = MagicMock()
    stream.__iter__.return_value = iter([])

    with pytest.raises(RuntimeError, match="No chunks returned"):
        ChatCompletionOpenAI._consume_stream(stream, request)

    stream.close.assert_called_once_with()
    assert events == [ProviderStreamEvent(ProviderStreamEventType.START)]


def test_execute_request_sends_expected_provider_arguments(openai_client) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")
    request = MagicMock(spec=ProviderRequest)
    payload = {
        "model": "model",
        "messages": [{"role": "user", "content": "question"}],
        "stream": False,
    }
    provider_response = object()
    openai_client.chat.completions.create.return_value = provider_response

    with patch.object(
        ToOpenAIChatCompletionRequestMapper,
        "map_request",
        return_value=payload,
    ) as mapper:
        result = completion._execute_request(request)

    assert result is provider_response
    mapper.assert_called_once_with(
        request,
        model="model",
        reasoning_effort=None,
    )
    openai_client.chat.completions.create.assert_called_once_with(**payload)


@pytest.mark.parametrize(
    "effort",
    [ReasoningEffortLMX.LOW, ReasoningEffortLMX.XHIGH, "provider-specific-effort"],
)
def test_execute_request_forwards_explicit_reasoning_effort(
    openai_client, effort: ReasoningEffort
) -> None:
    completion = ChatCompletionOpenAI(
        openai_client, "non-default-model", reasoning_effort=effort
    )
    request = MagicMock(spec=ProviderRequest)
    payload = {
        "model": "non-default-model",
        "messages": [],
        "stream": False,
        "reasoning_effort": effort if isinstance(effort, str) else effort.value,
    }

    with patch.object(
        ToOpenAIChatCompletionRequestMapper,
        "map_request",
        return_value=payload,
    ) as mapper:
        completion._execute_request(request)

    mapper.assert_called_once_with(
        request,
        model="non-default-model",
        reasoning_effort=effort,
    )
    openai_client.chat.completions.create.assert_called_once_with(**payload)


def test_parse_response_builds_response_with_content_usage_and_thinking(
    openai_client,
) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")
    provider_message = SimpleNamespace(content="answer")
    choices = [SimpleNamespace(message=provider_message)]
    response = cast(ChatCompletion, SimpleNamespace(choices=choices))
    request = ProviderRequest([SystemMessage("rules")])
    assistant_message = AssistantMessage("answer")
    usage = ModelUsage(10, 5, 15)
    thinking = ModelThinking("reasoning")

    with (
        patch.object(
            FromOpenAIChatCompletionMapper,
            "map_assistant_message",
            return_value=assistant_message,
        ) as map_assistant_message,
        patch.object(
            FromOpenAIChatCompletionMapper,
            "extract_usage",
            return_value=usage,
        ) as extract_usage,
        patch.object(
            FromOpenAIChatCompletionMapper,
            "extract_thinking",
            return_value=thinking,
        ) as extract_thinking,
    ):
        result = completion._parse_response(response, request)

    assert result.get_message() == assistant_message.get_message()
    assert result.get_name() == assistant_message.get_name()
    assert result.get_tool_calls() is assistant_message.get_tool_calls()
    assert result.usage is usage
    assert result.thinking is thinking
    map_assistant_message.assert_called_once_with(provider_message, tools=None)
    extract_usage.assert_called_once_with(response)
    extract_thinking.assert_called_once_with(response)


def test_parse_response_rejects_empty_choices(openai_client) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")
    request = ProviderRequest([SystemMessage("rules")])

    with pytest.raises(RuntimeError, match="No choices returned from the provider"):
        completion._parse_response(
            cast(ChatCompletion, SimpleNamespace(choices=[])),
            request,
        )


def test_call_structured_executes_and_parses_provider_response(openai_client) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")
    request = MagicMock(spec=StructuredProviderRequest)
    provider_response = object()
    parsed_response = StructuredProviderResponse(
        StructuredAssistantMessage(StructuredAnswer(value=42)),
    )

    with (
        patch.object(
            completion,
            "_execute_structured_request",
            return_value=provider_response,
        ) as execute,
        patch.object(
            completion,
            "_parse_structured_response",
            return_value=parsed_response,
        ) as parse,
    ):
        result = completion.call_structured(request)

    assert result is parsed_response
    execute.assert_called_once_with(request)
    parse.assert_called_once_with(provider_response, request)


def test_call_structured_streams_and_parses_accumulated_json(openai_client) -> None:
    events = []
    completion = ChatCompletionOpenAI(openai_client, "model")
    request = StructuredProviderRequest(
        [SystemMessage("rules")],
        StructuredAnswer,
        stream=True,
        on_event=events.append,
    )
    openai_client.chat.completions.create.return_value = make_stream('{"value":42}')

    result = completion.call_structured(request)

    assert result.get_message() == StructuredAnswer(value=42)
    assert [event.type for event in events] == [
        ProviderStreamEventType.START,
        ProviderStreamEventType.REASONING,
        ProviderStreamEventType.RESPONSE,
        ProviderStreamEventType.TOTALS,
        ProviderStreamEventType.END,
    ]


def test_execute_structured_request_sends_strict_response_format(
    openai_client,
) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")
    request = StructuredProviderRequest([SystemMessage("rules")], StructuredAnswer)
    payload = {
        "model": "model",
        "messages": [{"role": "user", "content": "question"}],
        "stream": False,
    }
    provider_response = object()
    response_format = {"type": "json_schema", "strict": True}
    openai_client.chat.completions.create.return_value = provider_response

    with (
        patch.object(
            ToOpenAIChatCompletionRequestMapper,
            "map_request",
            return_value=payload,
        ),
        patch.object(
            OpenAIChatCompletionResponseFormatMapper,
            "map",
            return_value=response_format,
        ) as map_response_format,
    ):
        result = completion._execute_structured_request(request)

    assert result is provider_response
    map_response_format.assert_called_once_with(StructuredAnswer, strict=True)
    openai_client.chat.completions.create.assert_called_once_with(
        model="model",
        messages=[{"role": "user", "content": "question"}],
        stream=False,
        response_format=response_format,
    )


def test_execute_structured_request_sends_non_strict_json_schema(
    openai_client,
) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")
    request = StructuredProviderRequest(
        [SystemMessage("rules")],
        StructuredAnswer,
        strict=False,
    )
    payload = {
        "model": "model",
        "messages": [{"role": "user", "content": "question"}],
        "stream": False,
    }
    provider_response = object()
    openai_client.chat.completions.create.return_value = provider_response

    with patch.object(
        ToOpenAIChatCompletionRequestMapper,
        "map_request",
        return_value=payload,
    ):
        result = completion._execute_structured_request(request)

    assert result is provider_response
    openai_client.chat.completions.create.assert_called_once_with(
        model="model",
        messages=[{"role": "user", "content": "question"}],
        stream=False,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "StructuredAnswer",
                "schema": StructuredAnswer.model_json_schema(),
                "strict": False,
            },
        },
    )


@pytest.mark.parametrize("strict", [True, False])
def test_parse_structured_response_returns_pydantic_content(
    openai_client,
    strict: bool,
) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")
    request = StructuredProviderRequest(
        [SystemMessage("rules")],
        StructuredAnswer,
        strict=strict,
    )
    provider_message = SimpleNamespace(content='{"value":42}', tool_calls=None)
    response = cast(
        ChatCompletion,
        SimpleNamespace(
            choices=[SimpleNamespace(message=provider_message)], usage=None
        ),
    )

    result = completion._parse_structured_response(
        response,
        request,
    )

    content = result.get_message()
    assert isinstance(content, StructuredAnswer)
    assert content.value == 42


def test_parse_structured_response_non_strict_mode_allows_coercion(
    openai_client,
) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")
    request = StructuredProviderRequest(
        [SystemMessage("rules")],
        StructuredAnswer,
        strict=False,
    )
    provider_message = SimpleNamespace(content='{"value":"42"}', tool_calls=None)
    response = cast(
        ChatCompletion,
        SimpleNamespace(
            choices=[SimpleNamespace(message=provider_message)], usage=None
        ),
    )

    result = completion._parse_structured_response(
        response,
        request,
    )

    assert result.get_message() == StructuredAnswer(value=42)


def test_parse_structured_response_rejects_empty_choices(openai_client) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")
    request = StructuredProviderRequest([SystemMessage("rules")], StructuredAnswer)
    response = cast(ChatCompletion, SimpleNamespace(choices=[]))

    with pytest.raises(RuntimeError, match="No choices returned from the provider"):
        completion._parse_structured_response(response, request)


@pytest.mark.parametrize(
    ("content", "error_type", "location"),
    [
        ('{"value":"42"}', "int_type", ("value",)),
        ('{"unexpected":42}', "missing", ("value",)),
        ("not-json", "json_invalid", ()),
    ],
)
def test_parse_structured_response_propagates_detailed_validation_error(
    openai_client,
    content: str,
    error_type: str,
    location: tuple,
) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")
    request = StructuredProviderRequest([SystemMessage("rules")], StructuredAnswer)
    provider_message = SimpleNamespace(content=content, tool_calls=None)
    response = cast(
        ChatCompletion,
        SimpleNamespace(
            choices=[SimpleNamespace(message=provider_message)],
            usage=None,
        ),
    )

    with pytest.raises(ValidationError) as error:
        completion._parse_structured_response(
            response,
            request,
        )

    assert error.value.errors()[0]["type"] == error_type
    assert error.value.errors()[0]["loc"] == location
