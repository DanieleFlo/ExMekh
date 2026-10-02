from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.structured_assistant_message import (
    StructuredAssistantMessage,
)
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    TextsUserMessage,
    UserMessage,
)
from exmekh_lib.interfaces.provider import Provider, ProviderRequestRejectedError
from tests.event_handler import as_provider_handler


class StructuredAnswer(BaseModel):
    value: int


@pytest.mark.parametrize("provider_fails", [False, True])
def test_complete_tool_transition_retains_the_completed_batch(provider_fails):
    provider = MagicMock(spec=Provider)
    provider.call.return_value = ProviderResponse(AssistantMessage("superseded"))
    client = Client(provider, SystemMessage("rules"))
    client.add_user_message(UserMessage("start"))
    client.add_tool_calls([ToolCall("skill", FunctionCall("work", {}))])
    client.add_tool_response(ToolMessage("done", "skill", "work"))
    before = client.dump_history()
    extra = {"agentic": {"operation": "return", "skill_call_id": "skill"}}

    if provider_fails:
        provider.call.side_effect = RuntimeError("provider unavailable")
        with pytest.raises(RuntimeError, match="provider unavailable"):
            client.complete_tool_transition(extra=extra)
    else:
        client.complete_tool_transition(extra=extra)

    assert client.dump_history() == before
    request = provider.call.call_args.args[0]
    assert request.get_extra() == extra
    assert request.get_tool_choice() == "none"
    assert request.get_streaming() is False


@pytest.mark.parametrize("structured", [True, False])
def test_call_validates_provider_request_before_changing_history(structured):
    provider = MagicMock(spec=Provider)
    client = Client(provider, SystemMessage("rules"))
    client.tools = object()
    before = client.get_history()
    request = ClientRequest(UserMessage("input"))

    with pytest.raises(TypeError, match="Tools must be a Tools instance"):
        if structured:
            client.call_structured(request, StructuredAnswer)
        else:
            client.call(request)

    assert client.get_history() == before
    provider.call.assert_not_called()
    provider.call_structured.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"extra": []},
        {"tool_choice": 42},
        {"streaming": 1},
        {"on_event": lambda event: None},
    ],
)
def test_user_continuation_validates_options_before_changing_history(options):
    provider = MagicMock(spec=Provider)
    client = Client(provider, SystemMessage("rules"))
    before = client.get_history()

    with pytest.raises(TypeError):
        client.continue_with_user_messages(
            [UserMessage("first"), UserMessage("second")], **options
        )

    assert client.get_history() == before
    provider.call.assert_not_called()


def test_add_user_message_rejects_other_roles_without_changing_history() -> None:
    provider = MagicMock(spec=Provider)
    system = SystemMessage("rules")
    client = Client(provider, system)

    with pytest.raises(TypeError, match="Expected a user message"):
        client.add_user_message(AssistantMessage("unexpected answer"))

    assert client.get_history() == [system]
    provider.call.assert_not_called()


def test_continue_with_user_messages_preserves_order_and_request_options() -> None:
    provider = MagicMock(spec=Provider)
    response = ProviderResponse(AssistantMessage("updated"))
    provider.call.return_value = response
    client = Client(provider, SystemMessage("rules"))
    client.add_user_message(UserMessage("initial"))
    client.add_tool_calls([ToolCall("origin", FunctionCall("work", {}))])
    client.add_tool_response(ToolMessage("running", "origin", "work"))
    handler = as_provider_handler(lambda event: None)
    messages = [UserMessage("first correction"), UserMessage("second correction")]
    result = client.continue_with_user_messages(
        messages,
        extra={"agentic": "scope"},
        tool_choice="required",
        streaming=True,
        on_event=handler,
    )
    request = provider.call.call_args.args[0]
    assert result is response
    assert request.get_messages()[-2:] == messages
    assert request.get_extra() == {"agentic": "scope"}
    assert request.get_tool_choice() == "required"
    assert request.get_streaming()
    assert request.on_event is handler
    assert client.get_history()[-1] is response


@pytest.mark.parametrize(
    "messages", [[], [AssistantMessage("invalid")], [UserMessage("valid"), None]]
)
def test_continue_with_user_messages_validates_before_mutating_history(
    messages,
) -> None:
    provider = MagicMock(spec=Provider)
    system = SystemMessage("rules")
    client = Client(provider, system)
    with pytest.raises(ValueError, match="non-empty sequence"):
        client.continue_with_user_messages(messages)
    assert client.get_history() == [system]
    provider.call.assert_not_called()


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("provider_fails", [True, False])
def test_steering_events_use_on_event_after_insertion_before_inference(
    streaming, provider_fails
):
    provider = MagicMock(spec=Provider)
    client = Client(provider, SystemMessage("rules"))
    events = []
    messages = [UserMessage("first"), UserMessage("second")]

    def observe(event):
        if event.type is ProviderStreamEventType.STEERING_ADDED:
            assert client.get_history()[-2:] == messages
            provider.call.assert_not_called()
        events.append(event)

    handler = as_provider_handler(observe)

    def respond(request):
        assert request.on_event is handler
        assert len(events) == 2
        assert [message.get_message() for message in request.get_messages()[-2:]] == [
            "first",
            "second",
        ]
        if provider_fails:
            raise OSError("provider failed")
        if streaming:
            request.execute_on_stream_event(
                ProviderStreamEvent(ProviderStreamEventType.START)
            )
        return ProviderResponse(AssistantMessage("done"))

    provider.call.side_effect = respond
    if provider_fails:
        with pytest.raises(OSError, match="provider failed"):
            client.continue_with_user_messages(
                messages, streaming=streaming, on_event=handler
            )
    else:
        client.continue_with_user_messages(
            messages, streaming=streaming, on_event=handler
        )
    assert [event.type for event in events[:2]] == [
        ProviderStreamEventType.STEERING_ADDED
    ] * 2
    assert len(events) == (3 if streaming and not provider_fails else 2)


@pytest.mark.parametrize("steering", [[], [UserMessage("correction")]])
def test_initial_turn_message_does_not_emit_a_steering_event(steering):
    provider = MagicMock(spec=Provider)
    provider.call.return_value = ProviderResponse(AssistantMessage("done"))
    client = Client(provider, SystemMessage("rules"))
    events = []
    initial = UserMessage("initial")
    client.continue_with_user_messages(
        steering, initial_message=initial, on_event=as_provider_handler(events.append)
    )
    assert [event.message.get_message() for event in events] == [
        message.get_message() for message in steering
    ]
    assert provider.call.call_args.args[0].get_messages()[1:] == [initial, *steering]


def test_user_continuation_rejects_invalid_initial_message_before_insertion():
    provider = MagicMock(spec=Provider)
    client = Client(provider, SystemMessage("rules"))
    with pytest.raises(TypeError, match="Initial message"):
        client.continue_with_user_messages(
            [], initial_message=AssistantMessage("invalid")
        )
    assert len(client.get_history()) == 1
    provider.call.assert_not_called()


def test_steering_notifications_survive_handler_mutation_of_the_source_batch():
    provider = MagicMock(spec=Provider)
    provider.call.return_value = ProviderResponse(AssistantMessage("done"))
    client = Client(provider, SystemMessage("rules"))
    second = TextsUserMessage(["second"])
    messages = [UserMessage("first"), UserMessage(second)]
    events = []

    def observe(event):
        events.append(event)
        if len(events) == 1:
            messages.clear()
            second.texts.append("handler mutation")

    client.continue_with_user_messages(messages, on_event=as_provider_handler(observe))

    assert [event.message.get_message() for event in events] == [
        "first",
        TextsUserMessage(["second"]),
    ]


def test_init_accepts_provider_returned_by_mapper() -> None:
    provider = MagicMock(spec=Provider)
    system_message = SystemMessage("rules")

    client = Client(provider=provider, system_message=system_message)

    assert client.get_history() == [system_message]


def test_init_rejects_non_provider_returned_by_mapper() -> None:
    with pytest.raises(TypeError, match="Expected a Provider instance"):
        Client(
            provider=cast(Provider, object()),
            system_message=SystemMessage("rules"),
        )


def test_init_rejects_non_system_message() -> None:
    provider = MagicMock(spec=Provider)

    with pytest.raises(TypeError, match="Expected a SystemMessage"):
        Client(provider=provider, system_message=cast(SystemMessage, object()))


def test_init_rejects_invalid_tools() -> None:
    provider = MagicMock(spec=Provider)

    with pytest.raises(TypeError, match="Tools must be a Tools instance"):
        Client(
            provider=provider,
            system_message=SystemMessage("rules"),
            tools=cast(Tools, object()),
        )


def test_create_history_checkpoint_is_independent() -> None:
    provider = MagicMock(spec=Provider)
    system_message = SystemMessage("rules")
    client = Client(provider=provider, system_message=system_message)

    checkpoint = client.create_history_checkpoint()
    checkpoint.add(UserMessage("checkpoint-only"))

    assert client.get_history() == [system_message]
    assert checkpoint.get_history()[0] is not system_message


@pytest.mark.parametrize(
    "malicious_input",
    [
        None,
        0,
        object(),
        "not-a-message",
        [],
        [SystemMessage("custom-rules"), object()],
    ],
)
def test_call_rejects_non_request_without_contacting_provider(
    malicious_input,
) -> None:
    provider = MagicMock(spec=Provider)
    client = Client(provider=provider, system_message=SystemMessage("rules"))

    with pytest.raises(TypeError, match="Expected a ClientRequest"):
        client.call(malicious_input)

    provider.call.assert_not_called()


def test_call_sends_history_and_records_response() -> None:
    provider = MagicMock(spec=Provider)
    response = ProviderResponse(AssistantMessage("non-default-answer"))
    provider_history = []

    def call_provider(request):
        provider_history.extend(request.get_messages())
        return response

    provider.call.side_effect = call_provider
    system_message = SystemMessage("strict custom rules")
    user_message = UserMessage("unusual request")

    client = Client(provider=provider, system_message=system_message)

    result = client.call(ClientRequest(user_message))

    assert result is response
    assert client.get_history() == [system_message, user_message, response]
    assert client.dump_history() == [
        {"role": "system", "content": "strict custom rules"},
        {"role": "user", "content": "unusual request"},
        {"role": "assistant", "content": "non-default-answer", "tool_calls": "null"},
    ]
    provider.call.assert_called_once()
    assert provider_history == [system_message, user_message]


def test_call_appends_subsequent_turn_to_history() -> None:
    provider = MagicMock(spec=Provider)
    system_message = SystemMessage("rules")
    first_user_message = UserMessage("first question")
    first_response = AssistantMessage("first answer")
    second_user_message = UserMessage("second question")
    second_response = AssistantMessage("second answer")
    first_provider_response = ProviderResponse(first_response)
    second_provider_response = ProviderResponse(second_response)
    provider.call.side_effect = [first_provider_response, second_provider_response]

    client = Client(provider=provider, system_message=system_message)

    client.call(ClientRequest(first_user_message))
    result = client.call(ClientRequest(second_user_message))

    assert result is second_provider_response
    assert client.get_history() == [
        system_message,
        first_user_message,
        first_provider_response,
        second_user_message,
        second_provider_response,
    ]
    second_request = provider.call.call_args_list[1].args[0]
    assert second_request.get_messages() == [
        system_message,
        first_user_message,
        first_provider_response,
        second_user_message,
    ]


def test_call_forwards_tools_and_tool_choice() -> None:
    def search(query: str) -> str:
        return query

    provider = MagicMock(spec=Provider)
    provider.call.return_value = ProviderResponse(AssistantMessage("answer"))
    tools = Tools([Tool(search)])
    client = Client(
        provider=provider, system_message=SystemMessage("rules"), tools=tools
    )

    request = ClientRequest(
        UserMessage("search docs"),
        extra={"temperature": 0.2},
        tool_choice="auto",
    )
    client.call(request)

    provider_request = provider.call.call_args.args[0]
    provider.call.assert_called_once_with(provider_request)
    assert provider_request.get_tools() is tools
    assert provider_request.get_tool_choice() == "auto"
    assert provider_request.get_extra() == {"temperature": 0.2}


@pytest.mark.parametrize("structured", [False, True])
def test_call_forwards_streaming_options(structured: bool) -> None:
    provider = MagicMock(spec=Provider)
    provider.call.return_value = ProviderResponse(AssistantMessage("answer"))
    provider.call_structured.return_value = StructuredProviderResponse(
        StructuredAssistantMessage(StructuredAnswer(value=42))
    )
    client = Client(provider=provider, system_message=SystemMessage("rules"))

    def callback(event) -> None:
        del event

    handler = as_provider_handler(callback)
    request = ClientRequest(
        UserMessage("question"),
        streaming=True,
        on_event=handler,
    )

    if structured:
        client.call_structured(request, StructuredAnswer)
        provider_request = provider.call_structured.call_args.args[0]
    else:
        client.call(request)
        provider_request = provider.call.call_args.args[0]

    assert provider_request.get_streaming() is True
    assert provider_request.on_event is handler


def test_call_provider_failure_keeps_unanswered_request_in_history() -> None:
    provider = MagicMock(spec=Provider)
    system_message = SystemMessage("stable-rules")
    user_message = UserMessage("failing-request")

    def fail_after_materializing_request(request):
        request.get_messages()
        raise RuntimeError("simulated provider failure")

    provider.call.side_effect = fail_after_materializing_request

    client = Client(provider=provider, system_message=system_message)

    with pytest.raises(RuntimeError, match="simulated provider failure"):
        client.call(ClientRequest(user_message))

    assert client.get_history() == [system_message, user_message]


def test_call_invalid_provider_response_keeps_request_in_history() -> None:
    provider = MagicMock(spec=Provider)
    malformed_response = object()
    provider.call.return_value = malformed_response
    system_message = SystemMessage("stable-rules")
    user_message = UserMessage("valid-request")

    client = Client(provider=provider, system_message=system_message)

    with pytest.raises(TypeError, match="list of Message instances"):
        client.call(ClientRequest(user_message))

    assert client.get_history() == [system_message, user_message]


def test_continue_after_tools_sends_existing_history_without_new_message() -> None:
    provider = MagicMock(spec=Provider)
    system_message = SystemMessage("rules")
    user_message = UserMessage("look up the weather")
    tool_request = AssistantMessage(
        "", tool_calls=[ToolCall("call-123", FunctionCall("weather", {}))]
    )
    tool_response = ToolMessage('{"temperature":28}', "call-123", "weather")
    final_response = AssistantMessage("It is 28 degrees")
    tool_provider_response = ProviderResponse(tool_request)
    final_provider_response = ProviderResponse(final_response)
    provider.call.side_effect = [tool_provider_response, final_provider_response]
    client = Client(provider=provider, system_message=system_message)

    client.call(ClientRequest(user_message))
    client.add_tool_response(tool_response)
    result = client.continue_after_tools(
        extra={"temperature": 0.2},
        tool_choice="auto",
    )

    provider_request = provider.call.call_args_list[1].args[0]
    assert type(provider_request) is ProviderRequest
    assert provider_request.get_messages() == [
        system_message,
        user_message,
        tool_provider_response,
        tool_response,
    ]
    assert provider_request.get_extra() == {"temperature": 0.2}
    assert provider_request.get_tool_choice() == "auto"
    assert result.get_message() == final_response.get_message()
    assert client.get_history() == [
        system_message,
        user_message,
        tool_provider_response,
        tool_response,
        final_provider_response,
    ]


def test_continue_after_tools_requires_tool_response() -> None:
    provider = MagicMock(spec=Provider)
    client = Client(provider=provider, system_message=SystemMessage("rules"))

    with pytest.raises(TypeError, match="last message must be a tool response"):
        client.continue_after_tools()

    provider.call.assert_not_called()


def test_add_tool_response_rejects_non_tool_message_without_changing_history() -> None:
    provider = MagicMock(spec=Provider)
    system_message = SystemMessage("rules")
    client = Client(provider=provider, system_message=system_message)

    with pytest.raises(TypeError, match="Expected a ToolMessage"):
        client.add_tool_response(cast(ToolMessage, object()))

    assert client.get_history() == [system_message]


def test_continue_after_tools_structured_builds_structured_request() -> None:
    provider = MagicMock(spec=Provider)
    system_message = SystemMessage("rules")
    user_message = UserMessage("look up a value")
    tool_request = AssistantMessage(
        "", tool_calls=[ToolCall("call-123", FunctionCall("lookup", {}))]
    )
    tool_response = ToolMessage('{"value":42}', "call-123", "lookup")
    structured_message = StructuredAssistantMessage(StructuredAnswer(value=42))
    provider.call.return_value = ProviderResponse(tool_request)
    provider.call_structured.return_value = StructuredProviderResponse(
        structured_message
    )
    client = Client(provider=provider, system_message=system_message)

    client.call(ClientRequest(user_message))
    client.add_tool_response(tool_response)
    result = client.continue_after_tools_structured(
        StructuredAnswer,
        strict=False,
        extra={"temperature": 0.2},
        tool_choice="auto",
    )

    provider_request = provider.call_structured.call_args.args[0]
    assert isinstance(provider_request, StructuredProviderRequest)
    assert provider_request.get_messages() == [
        system_message,
        user_message,
        provider.call.return_value,
        tool_response,
    ]
    assert provider_request.get_response_model() is StructuredAnswer
    assert provider_request.get_strict() is False
    assert provider_request.get_extra() == {"temperature": 0.2}
    assert provider_request.get_tool_choice() == "auto"
    assert result.get_message() == structured_message.get_message()
    assert client.get_history()[-1] is result


def test_continue_after_tools_structured_requires_tool_response() -> None:
    provider = MagicMock(spec=Provider)
    client = Client(provider=provider, system_message=SystemMessage("rules"))

    with pytest.raises(TypeError, match="last message must be a tool response"):
        client.continue_after_tools_structured(StructuredAnswer)

    provider.call_structured.assert_not_called()


def test_call_validation_does_not_evaluate_hostile_repr() -> None:
    class HostileRepresentation:
        def __repr__(self) -> str:
            raise RuntimeError("repr must not execute")

    provider = MagicMock(spec=Provider)
    client = Client(provider=provider, system_message=SystemMessage("rules"))

    with pytest.raises(TypeError, match="Expected a ClientRequest"):
        client.call(cast(ClientRequest, HostileRepresentation()))

    provider.call.assert_not_called()


def test_call_structured_records_typed_response_and_uses_strict_default() -> None:
    provider = MagicMock(spec=Provider)
    content = StructuredAnswer(value=42)
    response = StructuredProviderResponse(
        StructuredAssistantMessage(content),
    )
    provider.call_structured.return_value = response
    client = Client(provider=provider, system_message=SystemMessage("rules"))
    request = ClientRequest(UserMessage("return an object"))

    result = client.call_structured(request, StructuredAnswer)

    provider_request = provider.call_structured.call_args.args[0]
    provider.call_structured.assert_called_once_with(provider_request)
    assert isinstance(provider_request, StructuredProviderRequest)
    assert provider_request.get_response_model() is StructuredAnswer
    assert provider_request.get_strict() is True
    assert result is response
    assert result.get_message() == content
    assert client.dump_history()[-1]["content"] == '{"value":42}'


def test_call_structured_rejects_non_request_without_contacting_provider() -> None:
    provider = MagicMock(spec=Provider)
    system_message = SystemMessage("rules")
    client = Client(provider=provider, system_message=system_message)

    with pytest.raises(TypeError, match="Expected a ClientRequest"):
        client.call_structured(cast(ClientRequest, object()), StructuredAnswer)

    provider.call_structured.assert_not_called()
    assert client.get_history() == [system_message]


def test_call_structured_forwards_non_strict_mode() -> None:
    provider = MagicMock(spec=Provider)
    provider.call_structured.return_value = StructuredProviderResponse(
        StructuredAssistantMessage(StructuredAnswer(value=42)),
    )
    client = Client(provider=provider, system_message=SystemMessage("rules"))

    client.call_structured(
        ClientRequest(UserMessage("return an object")),
        StructuredAnswer,
        strict=False,
    )

    provider_request = provider.call_structured.call_args.args[0]
    assert provider_request.get_strict() is False


@pytest.mark.parametrize("response_model", [object, StructuredAnswer(value=1)])
def test_call_structured_rejects_non_pydantic_model_type(response_model) -> None:
    provider = MagicMock(spec=Provider)
    client = Client(provider=provider, system_message=SystemMessage("rules"))

    with pytest.raises(TypeError, match="subclass of BaseModel"):
        client.call_structured(
            ClientRequest(UserMessage("return an object")),
            response_model,
        )
    provider.call_structured.assert_not_called()
    assert client.dump_history() == [{"role": "system", "content": "rules"}]


def test_call_structured_propagates_validation_error_with_details() -> None:
    provider = MagicMock(spec=Provider)
    try:
        StructuredAnswer.model_validate_json('{"value":"invalid"}', strict=True)
    except ValidationError as validation_error:
        provider.call_structured.side_effect = validation_error
    client = Client(provider=provider, system_message=SystemMessage("rules"))

    with pytest.raises(ValidationError) as error:
        client.call_structured(
            ClientRequest(UserMessage("return an object")),
            StructuredAnswer,
        )

    assert error.value.errors()[0]["loc"] == ("value",)
    assert error.value.errors()[0]["type"] == "int_type"


def test_client_delegates_tokenizer_and_id_generation_to_provider() -> None:
    provider = MagicMock(spec=Provider)
    mock_tokenizer = MagicMock()
    mock_session = MagicMock()
    provider.get_tokenizer.return_value = mock_tokenizer
    provider.generate_id.return_value = "generated-id"
    provider.get_session.return_value = mock_session
    client = Client(provider=provider, system_message=SystemMessage("rules"))

    assert client.get_tokenizer() is mock_tokenizer
    provider.get_tokenizer.assert_called_once_with()

    assert client.generate_id() == "generated-id"
    provider.generate_id.assert_called_once_with()

    assert client.get_session() is mock_session
    provider.get_session.assert_called_once_with()


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_rejected_attachment_restores_history_and_allows_next_turn(
    structured, streaming
):
    provider = MagicMock(spec=Provider)
    accepted = ProviderResponse(AssistantMessage("Remembered"))
    provider.call.return_value = accepted
    client = Client(provider, SystemMessage("Keep previous context"))
    client.call(ClientRequest(UserMessage("Remember ORBIT-731")))
    before = client.get_history()
    attachment = UserMessage(
        [
            "Read this",
            FilesUserMessage([DocumentUserMessage(b"binary", "report", "pdf")]),
        ]
    )
    error = ProviderRequestRejectedError("Only text and images are supported")

    def reject(request):
        assert request.get_messages() == [*before, attachment]
        assert client.get_history() == [*before, attachment]
        raise error

    if structured:
        provider.call_structured.side_effect = reject
    else:
        provider.call.side_effect = reject
    with pytest.raises(
        ProviderRequestRejectedError, match="Only text and images"
    ) as raised:
        request = ClientRequest(attachment, streaming=streaming)
        if structured:
            client.call_structured(request, StructuredAnswer)
        else:
            client.call(request)
    assert raised.value is error
    assert client.get_history() == before

    provider.call.side_effect = None
    following = UserMessage("What did I ask you to remember?")
    client.call(ClientRequest(following))
    assert provider.call.call_args.args[0].get_messages() == [*before, following]
    assert client.get_history() == [*before, following, accepted]


@pytest.mark.parametrize("with_initial", [False, True])
def test_rejected_user_continuation_restores_the_whole_added_batch(with_initial):
    provider = MagicMock(spec=Provider)
    client = Client(provider, SystemMessage("rules"))
    before = client.get_history()
    initial = UserMessage("initial") if with_initial else None
    messages = [
        UserMessage("steering"),
        UserMessage(FilesUserMessage([DocumentUserMessage(b"data", "report", "pdf")])),
    ]
    provider.call.side_effect = ProviderRequestRejectedError("Unsupported attachment")
    with pytest.raises(ProviderRequestRejectedError, match="Unsupported attachment"):
        client.continue_with_user_messages(messages, initial_message=initial)
    assert client.get_history() == before
    provider.call.side_effect = None
    provider.call.return_value = ProviderResponse(AssistantMessage("OK"))
    client.continue_with_user_messages([UserMessage("valid text")])
    assert len(client.get_history()) == 3


@pytest.mark.parametrize("structured", [False, True])
def test_rejected_tool_continuation_preserves_existing_tool_batch(structured):
    provider = MagicMock(spec=Provider)
    client = Client(provider, SystemMessage("rules"))
    client.add_user_message(UserMessage("start"))
    client.add_tool_calls([ToolCall("call-1", FunctionCall("read", {}))])
    client.add_tool_response(ToolMessage("result", "call-1", "read"))
    before = client.get_history()
    provider.call.side_effect = ProviderRequestRejectedError("Rejected request")
    provider.call_structured.side_effect = ProviderRequestRejectedError(
        "Rejected request"
    )
    with pytest.raises(ProviderRequestRejectedError):
        if structured:
            client.continue_after_tools_structured(StructuredAnswer)
        else:
            client.continue_after_tools()
    assert client.get_history() == before
