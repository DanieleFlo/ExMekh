from unittest.mock import MagicMock, patch

import pytest
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)
from pydantic import BaseModel

from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.provider_stream_event import ProviderStreamEventType
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.providers.response_openai import ResponseOpenAI
from exmekh_lib.core.providers.to_openai_response_request_mapper import (
    ToOpenAIResponseRequestMapper,
)
from exmekh_lib.interfaces.provider import ProviderResponseIncompleteError
from tests.event_handler import as_provider_handler


def make_response(content: str = "answer") -> Response:
    return Response.model_construct(
        output=[
            ResponseOutputMessage(
                id="message_1",
                content=[
                    ResponseOutputText(
                        annotations=[],
                        text=content,
                        type="output_text",
                        logprobs=[],
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        usage=None,
        error=None,
        status="completed",
    )


class StructuredAnswer(BaseModel):
    value: int


class DefaultAnswer(BaseModel):
    value: int = 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("client", object(), "Client must be an instance of OpenAI"),
        ("model", object(), "Model must be a string"),
        ("reasoning_effort", object(), "Reasoning effort must be"),
    ],
)
def test_init_rejects_invalid_inputs(
    openai_client,
    field: str,
    value: object,
    message: str,
) -> None:
    arguments = {
        "client": openai_client,
        "model": "model",
        "reasoning_effort": None,
    }
    arguments[field] = value

    with pytest.raises(TypeError, match=message):
        ResponseOpenAI(**arguments)


def test_call_executes_and_parses_provider_response(openai_client) -> None:
    completion = ResponseOpenAI(openai_client, "model")
    request = ProviderRequest([AssistantMessage("question")])
    response = make_response()
    openai_client.responses.create.return_value = response

    result = completion.call(request)

    assert isinstance(result, ProviderResponse)
    assert result.get_message() == "answer"


def test_execute_request_sends_expected_provider_arguments(openai_client) -> None:
    completion = ResponseOpenAI(openai_client, "model")
    request = ProviderRequest([AssistantMessage("question")])
    response = make_response()
    payload = {"model": "model", "input": [], "stream": False}
    openai_client.responses.create.return_value = response

    with patch.object(
        ToOpenAIResponseRequestMapper,
        "map_request",
        return_value=payload,
    ) as mapper:
        result = completion._execute_request(request)

    assert result is response
    mapper.assert_called_once_with(request, model="model", reasoning_effort=None)
    openai_client.responses.create.assert_called_once_with(**payload)


def test_execute_request_consumes_stream_and_emits_events(openai_client) -> None:
    completion = ResponseOpenAI(openai_client, "model")
    events = []
    request = ProviderRequest(
        [AssistantMessage("question")],
        stream=True,
        on_event=as_provider_handler(events.append),
    )
    response = make_response()
    stream = MagicMock()
    stream.__iter__.return_value = iter(
        [
            ResponseTextDeltaEvent.model_construct(delta="answer"),
            ResponseCompletedEvent.model_construct(response=response),
        ]
    )
    openai_client.responses.create.return_value = stream

    result = completion.call(request)

    assert result.get_message() == "answer"
    assert [event.type for event in events] == [
        ProviderStreamEventType.START,
        ProviderStreamEventType.RESPONSE,
        ProviderStreamEventType.TOTALS,
        ProviderStreamEventType.END,
    ]
    stream.close.assert_called_once_with()


def test_call_structured_executes_and_parses_response(openai_client) -> None:
    completion = ResponseOpenAI(openai_client, "model")
    request = StructuredProviderRequest(
        [AssistantMessage("question")],
        StructuredAnswer,
    )
    openai_client.responses.create.return_value = make_response('{"value":42}')

    result = completion.call_structured(request)

    assert result.get_message() == StructuredAnswer(value=42)
    payload = openai_client.responses.create.call_args.kwargs
    assert payload["text"]["format"]["name"] == "StructuredAnswer"


def test_consume_stream_rejects_missing_terminal_response() -> None:
    request = ProviderRequest([AssistantMessage("question")])

    with pytest.raises(RuntimeError, match="No completed response"):
        ResponseOpenAI._consume_stream([], request)


@pytest.mark.parametrize(
    ("event_type", "response", "message"),
    [
        (
            ResponseFailedEvent,
            Response.model_construct(
                error=MagicMock(message="failed"),
                status="failed",
            ),
            "Provider response failed: failed",
        ),
        (
            ResponseIncompleteEvent,
            Response.model_construct(error=None, status="incomplete"),
            "did not complete: incomplete",
        ),
    ],
)
def test_consume_stream_rejects_unsuccessful_terminal_response(
    event_type,
    response: Response,
    message: str,
) -> None:
    request = ProviderRequest([AssistantMessage("question")])
    event = event_type.model_construct(response=response)

    with pytest.raises(RuntimeError, match=message):
        ResponseOpenAI._consume_stream([event], request)


def test_incomplete_response_exposes_a_typed_error():
    response = Response.model_construct(error=None, status="incomplete")
    event = ResponseIncompleteEvent.model_construct(response=response)
    with pytest.raises(ProviderResponseIncompleteError):
        ResponseOpenAI._consume_stream(
            [event], ProviderRequest([AssistantMessage("question")])
        )


def test_parse_structured_response_uses_empty_object_for_missing_text(
    openai_client,
) -> None:
    completion = ResponseOpenAI(openai_client, "model")
    request = StructuredProviderRequest(
        [AssistantMessage("question")],
        DefaultAnswer,
    )
    response = Response.model_construct(
        output=[],
        usage=None,
        error=None,
        status="completed",
    )

    result = completion._parse_structured_response(response, request)

    assert result.get_message() == DefaultAnswer()
