from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.client import Client
from qs.core.models.client_request import ClientRequest
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.structured_assistant_message import StructuredAssistantMessage
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool import Tool
from qs.core.models.tool_message import ToolMessage
from qs.core.models.tools import Tools
from qs.core.models.user_message import UserMessage
from qs.interfaces.provider import Provider


class StructuredAnswer(BaseModel):
    value: int


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
    callback = lambda event: None
    request = ClientRequest(
        UserMessage("question"),
        streaming=True,
        on_event=callback,
    )

    if structured:
        client.call_structured(request, StructuredAnswer)
        provider_request = provider.call_structured.call_args.args[0]
    else:
        client.call(request)
        provider_request = provider.call.call_args.args[0]

    assert provider_request.get_streaming() is True
    assert provider_request.on_event is callback


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
    tool_request = AssistantMessage("")
    tool_response = ToolMessage('{"temperature":28}', "call-123")
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
    tool_request = AssistantMessage("")
    tool_response = ToolMessage('{"value":42}', "call-123")
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
