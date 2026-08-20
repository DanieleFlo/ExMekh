from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat.chat_completion import ChatCompletion

from qs.config.local_provider import ReasoningEffort
from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.function_message import FunctionMessage
from qs.core.models.history import MessageHistory
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


def test_init_preserves_client_and_model(openai_client) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")

    assert completion._client is openai_client
    assert completion._model == "model"


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
    completion = ChatCompletionOpenAI(openai_client, "model")
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
        stream=False,
    )


@pytest.mark.parametrize("effort", [ReasoningEffort.LOW, ReasoningEffort.XHIGH])
def test_execute_request_forwards_explicit_reasoning_effort(
    openai_client, effort: ReasoningEffort
) -> None:
    completion = ChatCompletionOpenAI(
        openai_client, "non-default-model", reasoning_effort=effort
    )
    request = MagicMock(spec=ProviderRequest)
    mapped_messages = [{"role": "user", "content": "different question"}]

    with patch.object(
        completion, "_map_request_to_provider_format", return_value=mapped_messages
    ):
        completion._execute_request(request)

    openai_client.chat.completions.create.assert_called_once_with(
        model="non-default-model",
        messages=mapped_messages,
        stream=False,
        reasoning_effort=effort.value,
    )


def test_parse_response_builds_response_with_content_usage_and_thinking(
    openai_client,
) -> None:
    completion = ChatCompletionOpenAI(openai_client, "model")
    provider_message = SimpleNamespace(content="answer")
    choices = [SimpleNamespace(message=provider_message)]
    response = cast(ChatCompletion, SimpleNamespace(choices=choices))
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
        result = completion._parse_response(response)

    assert result.response is assistant_message
    assert result.usage is usage
    assert result.thinking is thinking
    map_assistant_message.assert_called_once_with(provider_message)
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
    request.get_request.return_value = MessageHistory([system_message, user_message])
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
