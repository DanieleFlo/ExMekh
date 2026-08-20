from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall,
)
from openai.types.chat.chat_completion_message_custom_tool_call import (
    Custom as OpenAICustom,
)

from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.function_message import FunctionMessage
from qs.core.models.history import History
from qs.core.models.model_thinking import ModelThinking
from qs.core.models.model_usage import ModelUsage
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool_message import ToolMessage
from qs.core.models.user_message import UserMessage
from qs.core.providers.chat_completion_openai import ChatCompletionOpenAI
from qs.core.providers.from_openai_chat_completion_mapper import (
    FromOpenAIChatCompletionMapper,
)
from qs.core.providers.to_openai_chat_completion_mapper import (
    ToOpenAIChatCompletionMapper,
)
from qs.interfaces.message import Message


def test_init_preserves_client_model_and_number_of_choices(openai_client) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model", number_of_choices=3)

    assert completion._client is openai_client
    assert completion._model == "model"
    assert completion._number_of_choices == 3


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
    parse.assert_called_once_with(provider_response)


def test_execute_request_sends_expected_provider_arguments(openai_client) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model", number_of_choices=2)
    request = MagicMock(spec=ProviderRequest)
    mapped_messages = [{"role": "user", "content": "question"}]
    provider_response = object()
    openai_client.chat.completions.create.return_value = provider_response

    with patch.object(
        completion,
        "_map_request_to_provider_format",
        return_value=mapped_messages,
    ) as map_request:
        result = completion._execute_request(request)

    assert result is provider_response
    map_request.assert_called_once_with(request)
    openai_client.chat.completions.create.assert_called_once_with(
        model="model",
        messages=mapped_messages,
        n=2,
        stream=False,
    )


def test_parse_response_builds_response_with_content_usage_and_thinking(
    openai_client,
) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")
    choices = [SimpleNamespace(message=SimpleNamespace(content="answer"))]
    response = cast(ChatCompletion, SimpleNamespace(choices=choices))
    assistant_message = AssistantMessage("answer")
    usage = ModelUsage(10, 5, 15)
    thinking = ModelThinking("reasoning")

    with (
        patch.object(
            completion, "_extract_content", return_value=assistant_message
        ) as extract,
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
        result = completion._parse_response(response)

    assert result.response is assistant_message
    assert result.usage is usage
    assert result.thinking is thinking
    extract.assert_called_once_with(choices)
    extract_usage.assert_called_once_with(response)
    extract_thinking.assert_called_once_with(response)


def test_parse_response_rejects_empty_choices(openai_client) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")

    with pytest.raises(RuntimeError, match="No choices returned from the provider"):
        completion._parse_response(cast(ChatCompletion, SimpleNamespace(choices=[])))


def test_map_request_to_provider_format_maps_history_in_order(openai_client) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")
    system_message = SystemMessage("rules")
    user_message = UserMessage("question")
    request = MagicMock(spec=ProviderRequest)
    request.get_request.return_value = History([system_message, user_message])
    mapped_messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
    ]

    with patch.object(
        completion,
        "_map_message_to_provider_format",
        side_effect=mapped_messages,
    ) as map_message:
        result = completion._map_request_to_provider_format(request)

    assert result == mapped_messages
    assert [call.args[0] for call in map_message.call_args_list] == [
        system_message,
        user_message,
    ]


@pytest.mark.parametrize(
    ("message", "mapper_name"),
    [
        (SystemMessage("rules"), "map_system_message"),
        (UserMessage("question"), "map_user_message"),
        (AssistantMessage("answer"), "map_assistant_message"),
        (ToolMessage("result", "call_123"), "map_tool_message"),
        (FunctionMessage("result", "calculate"), "map_function_message"),
    ],
)
def test_map_message_to_provider_format_dispatches_supported_message(
    openai_client, message, mapper_name
) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")
    mapped = {"mapped": True}

    with patch.object(
        ToOpenAIChatCompletionMapper,
        mapper_name,
        return_value=mapped,
    ) as mapper:
        result = completion._map_message_to_provider_format(message)

    assert result is mapped
    mapper.assert_called_once_with(message)


def test_map_message_to_provider_format_rejects_unknown_message(
    openai_client,
) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")

    with pytest.raises(TypeError, match="Unsupported message type"):
        completion._map_message_to_provider_format(cast(Message, object()))


def test_extract_content_returns_first_choice_as_assistant_message(
    openai_client,
) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")
    choices = [
        Choice(
            finish_reason="stop",
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(content="first", role="assistant"),
        ),
        Choice(
            finish_reason="stop",
            index=1,
            logprobs=None,
            message=ChatCompletionMessage(content="second", role="assistant"),
        ),
    ]

    result = completion._extract_content(choices)

    assert isinstance(result, AssistantMessage)
    assert result.get_message() == "first"
    assert result.get_tool_calls() is None


def test_content_from_maps_custom_tool_call() -> None:
    choice = Choice(
        finish_reason="tool_calls",
        index=0,
        logprobs=None,
        message=ChatCompletionMessage(
            content=None,
            role="assistant",
            tool_calls=[
                ChatCompletionMessageCustomToolCall(
                    id="call_123",
                    custom=OpenAICustom(
                        name="execute_python",
                        input="print(2 + 2)",
                    ),
                    type="custom",
                )
            ],
        ),
    )

    result = ChatCompletionOpenAI._content_from(choice)

    assert isinstance(result, AssistantMessage)
    assert result.get_message() is None
    assert result.get_tool_calls() == [
        {
            "id": "call_123",
            "custom": {
                "name": "execute_python",
                "input": "print(2 + 2)",
            },
            "type": "custom",
        }
    ]


def test_content_from_rejects_missing_message() -> None:
    with pytest.raises(RuntimeError, match="Choice message is None"):
        ChatCompletionOpenAI._content_from(cast(Choice, SimpleNamespace(message=None)))
