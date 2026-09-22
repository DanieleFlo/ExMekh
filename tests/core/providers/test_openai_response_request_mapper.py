from pathlib import Path
from typing import Annotated, cast

import pytest
from openai.types.responses import ResponseFunctionToolCall

from qs.core.models.assistant_message import AssistantMessage, FunctionCall, ToolCall
from qs.core.models.function_message import FunctionMessage
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.reasoning_effort import ReasoningEffort, ReasoningEffortLMH
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool import Tool
from qs.core.models.tool_message import ToolMessage
from qs.core.models.tools import Tools
from qs.core.models.user_message import UserMessage
from qs.core.providers.openai_response_tool_mapper import OpenAIResponseToolMapper
from qs.core.providers.to_openai_response_request_mapper import (
    ToOpenAIResponseRequestMapper,
)


def search(query: Annotated[str, "Text to search"]) -> str:
    """Search documents."""
    return query


def test_map_request_preserves_agentic_options_and_flattens_tools() -> None:
    request = ProviderRequest(
        [SystemMessage("system"), UserMessage("question")],
        extra={"agentic": {"allowed_tools": ["search"]}},
        tool_choice="required",
        tools=Tools([Tool(search)]),
        stream=True,
    )

    payload = ToOpenAIResponseRequestMapper.map_request(
        request,
        model="local-model",
        reasoning_effort=ReasoningEffortLMH.HIGH,
    )

    assert payload == {
        "model": "local-model",
        "input": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
        "stream": True,
        "reasoning": {"effort": "high", "summary": "auto"},
        "extra_body": {"agentic": {"allowed_tools": ["search"]}},
        "tool_choice": "required",
        "tools": [
            {
                "type": "function",
                "name": "search",
                "description": "Search documents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Text to search",
                        }
                    },
                    "additionalProperties": False,
                    "required": ["query"],
                },
                "strict": True,
            }
        ],
    }


def test_map_request_maps_tool_call_history_and_output() -> None:
    tool_call = ToolCall(
        id="call_123",
        function=FunctionCall(name="search", arguments={"query": "docs"}),
    )
    request = ProviderRequest(
        [
            AssistantMessage(None, tool_calls=[tool_call]),
            ToolMessage("result", "call_123"),
        ]
    )

    payload = ToOpenAIResponseRequestMapper.map_request(request, model="model")

    assert payload["input"] == [
        {
            "type": "function_call",
            "call_id": "call_123",
            "name": "search",
            "arguments": '{"query":"docs"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "result",
        },
    ]


def test_map_request_rejects_legacy_function_messages() -> None:
    request = ProviderRequest([FunctionMessage("result", "search")])

    with pytest.raises(TypeError, match="legacy function messages"):
        ToOpenAIResponseRequestMapper.map_request(request, model="model")


def test_map_request_rejects_invalid_request() -> None:
    with pytest.raises(TypeError, match="ProviderRequest"):
        ToOpenAIResponseRequestMapper.map_request(object(), model="model")


def test_map_request_maps_bound_tool_schema_without_work_root(tmp_path: Path) -> None:
    from functools import partial

    def run(work_root: Path, command: str) -> str:
        return str(work_root / command)

    request = ProviderRequest(
        [UserMessage("run")],
        tools=Tools([Tool(partial(run, tmp_path), name="run")]),
    )

    payload = ToOpenAIResponseRequestMapper.map_request(request, model="model")

    assert payload["tools"][0]["parameters"]["properties"] == {
        "command": {"type": "string"}
    }


def test_map_request_omits_empty_tool_registry() -> None:
    request = ProviderRequest([UserMessage("question")], tools=Tools([]))

    payload = ToOpenAIResponseRequestMapper.map_request(request, model="model")

    assert "tools" not in payload


def test_get_reasoning_effort_value_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError, match="string or an instance of ReasoningEffort"):
        ToOpenAIResponseRequestMapper._get_reasoning_effort_value(
            cast(ReasoningEffort, object())
        )


def test_map_request_accepts_provider_specific_reasoning_effort() -> None:
    request = ProviderRequest([UserMessage("question")])

    payload = ToOpenAIResponseRequestMapper.map_request(
        request,
        model="model",
        reasoning_effort="provider-specific",
    )

    assert payload["reasoning"] == {"effort": "provider-specific", "summary": "auto"}


def test_map_tools_omits_empty_description() -> None:
    def count(value: int = 1) -> int:
        return value

    mapped = OpenAIResponseToolMapper.map_tools(Tools([Tool(count)]))

    assert "description" not in mapped[0]
    assert mapped[0]["strict"] is True
    assert mapped[0]["parameters"]["properties"]["value"] == {"type": "integer"}
    assert mapped[0]["parameters"]["required"] == ["value"]


def test_resolve_tool_call_rejects_invalid_response_type() -> None:
    with pytest.raises(TypeError, match="Response function tool call"):
        OpenAIResponseToolMapper.resolve_tool_call(
            cast(ResponseFunctionToolCall, object()),
            Tools([]),
        )
