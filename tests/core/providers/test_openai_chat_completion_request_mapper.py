from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.function_message import FunctionMessage
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.reasoning_effort import (
    ReasoningEffort,
    ReasoningEffortLMH,
    ReasoningEffortLMHXM,
    ReasoningEffortLMX,
)
from qs.core.models.structured_assistant_message import StructuredAssistantMessage
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool import Tool
from qs.core.models.tool_message import ToolMessage
from qs.core.models.tools import Tools
from qs.core.models.user_message import UserMessage
from qs.core.providers.openai_chat_completion_tool_mapper import (
    OpenAIChatCompletionToolMapper,
)
from qs.core.providers.to_openai_chat_completion_mapper import (
    ToOpenAIChatCompletionMapper,
)
from qs.core.providers.to_openai_chat_completion_request_mapper import (
    ToOpenAIChatCompletionRequestMapper,
)
from qs.interfaces.message import Message


class StructuredAnswer(BaseModel):
    value: int


def test_map_request_builds_minimal_payload() -> None:
    request = ProviderRequest([SystemMessage("rules"), UserMessage("question")])

    payload = ToOpenAIChatCompletionRequestMapper.map_request(
        request,
        model="model",
    )

    assert payload == {
        "model": "model",
        "messages": [
            {"role": "system", "content": "rules", "name": ""},
            {"role": "user", "content": "question", "name": ""},
        ],
        "stream": False,
    }


def test_map_request_includes_usage_for_streaming() -> None:
    request = ProviderRequest([UserMessage("question")], stream=True)

    payload = ToOpenAIChatCompletionRequestMapper.map_request(
        request,
        model="model",
    )

    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_map_request_maps_all_provider_options() -> None:
    def search(query: str) -> str:
        """Search documents."""
        return query

    tools = Tools([Tool(search)])
    request = ProviderRequest(
        [SystemMessage("rules")],
        extra={"temperature": 0.2},
        tool_choice="auto",
        tools=tools,
    )
    mapped_tools = [{"type": "function", "function": {"name": "search"}}]

    with patch.object(
        OpenAIChatCompletionToolMapper,
        "map_tools",
        return_value=mapped_tools,
    ) as map_tools:
        payload = ToOpenAIChatCompletionRequestMapper.map_request(
            request,
            model="reasoning-model",
            reasoning_effort=ReasoningEffortLMH.HIGH,
        )

    assert payload == {
        "model": "reasoning-model",
        "messages": [{"role": "system", "content": "rules", "name": ""}],
        "stream": False,
        "reasoning_effort": "high",
        "extra_body": {"temperature": 0.2},
        "tool_choice": "auto",
        "tools": mapped_tools,
    }
    map_tools.assert_called_once_with(tools)


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        (ReasoningEffortLMX.XHIGH, "xhigh"),
        (ReasoningEffortLMH.HIGH, "high"),
        (ReasoningEffortLMHXM.Max, "max"),
        ("provider-specific-effort", "provider-specific-effort"),
    ],
)
def test_map_request_maps_supported_reasoning_efforts(
    effort: ReasoningEffort,
    expected: str,
) -> None:
    request = ProviderRequest([UserMessage("question")])

    payload = ToOpenAIChatCompletionRequestMapper.map_request(
        request,
        model="reasoning-model",
        reasoning_effort=effort,
    )

    assert payload["reasoning_effort"] == expected


@pytest.mark.parametrize(
    ("message", "mapper_name"),
    [
        (SystemMessage("rules"), "map_system_message"),
        (UserMessage("question"), "map_user_message"),
        (AssistantMessage("answer"), "map_assistant_message"),
        (
            StructuredAssistantMessage(StructuredAnswer(value=42)),
            "map_structured_assistant_message",
        ),
        (ToolMessage("result", "call_123"), "map_tool_message"),
        (FunctionMessage("result", "calculate"), "map_function_message"),
    ],
)
def test_map_message_dispatches_supported_message(message, mapper_name) -> None:
    mapped = {"mapped": True}

    with patch.object(
        ToOpenAIChatCompletionMapper,
        mapper_name,
        return_value=mapped,
    ) as mapper:
        result = ToOpenAIChatCompletionRequestMapper._map_message(message)

    assert result is mapped
    mapper.assert_called_once_with(message)


def test_map_request_rejects_invalid_request() -> None:
    with pytest.raises(TypeError, match="Expected a ProviderRequest"):
        ToOpenAIChatCompletionRequestMapper.map_request(object(), model="model")


def test_map_message_rejects_unknown_message() -> None:
    with pytest.raises(TypeError, match="Unsupported message type"):
        ToOpenAIChatCompletionRequestMapper._map_message(MagicMock(spec=Message))


def test_map_request_omits_empty_tool_registry() -> None:
    request = ProviderRequest([SystemMessage("rules")], tools=Tools([]))

    payload = ToOpenAIChatCompletionRequestMapper.map_request(request, model="model")

    assert "tools" not in payload


def test_get_reasoning_effort_value_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError, match="string or an instance of ReasoningEffort"):
        ToOpenAIChatCompletionRequestMapper._get_reasoning_effort_value(
            cast(ReasoningEffort, object())
        )
