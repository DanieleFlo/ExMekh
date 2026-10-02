from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageCustomToolCall,
)
from openai.types.chat.chat_completion_message_custom_tool_call import Custom
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    Function as OpenAIFunction,
)
from PIL import Image
from pydantic import BaseModel

from qs.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from qs.core.models.function_message import FunctionMessage
from qs.core.models.structured_assistant_message import StructuredAssistantMessage
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool import Tool
from qs.core.models.tool_message import ToolMessage
from qs.core.models.tools import Tools
from qs.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    ImagesUserMessage,
    TextsUserMessage,
    UserMessage,
)
from qs.core.providers.from_openai_chat_completion_mapper import (
    FromOpenAIChatCompletionMapper,
)
from qs.core.providers.to_openai_chat_completion_mapper import (
    ToOpenAIChatCompletionMapper,
)


class StructuredAnswer(BaseModel):
    value: int


def search(query: str) -> None:
    pass


def test_extract_usage_returns_empty_usage_when_provider_omits_it() -> None:
    usage = FromOpenAIChatCompletionMapper.extract_usage(SimpleNamespace(usage=None))

    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.total_tokens is None


def test_extract_usage_maps_provider_token_counts() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
    )

    usage = FromOpenAIChatCompletionMapper.extract_usage(response)

    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        10,
        5,
        15,
    )


def test_extract_thinking_returns_first_available_thinking() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="answer")),
            SimpleNamespace(message=SimpleNamespace(reasoning_content="reasoning")),
            SimpleNamespace(message=SimpleNamespace(thinking="ignored")),
        ]
    )

    thinking = FromOpenAIChatCompletionMapper.extract_thinking(response)

    assert thinking is not None
    assert thinking.thoughts == "reasoning"


def test_extract_thinking_returns_none_when_no_choice_contains_it() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )

    assert FromOpenAIChatCompletionMapper.extract_thinking(response) is None


def test_map_provider_message_returns_assistant_message() -> None:
    message = ChatCompletionMessage(content="answer", role="assistant")

    result = FromOpenAIChatCompletionMapper.map_assistant_message(message)

    assert isinstance(result, AssistantMessage)
    assert result.get_message() == "answer"
    assert result.get_tool_calls() is None


@pytest.mark.parametrize("role", ["system", "developer"])
def test_map_system_message_maps_supported_roles(role: str) -> None:
    message = SystemMessage("instructions", name="policy")

    with patch.object(message, "get_role_name", return_value=role):
        mapped = ToOpenAIChatCompletionMapper.map_system_message(message)

    assert mapped == {
        "content": "instructions",
        "role": role,
        "name": "policy",
    }


def test_map_system_message_rejects_unsupported_role() -> None:
    message = SystemMessage("instructions")

    with (
        patch.object(message, "get_role_name", return_value="assistant"),
        pytest.raises(
            ValueError, match="Expected role 'system' or 'developer', got 'assistant'"
        ),
    ):
        ToOpenAIChatCompletionMapper.map_system_message(message)


def test_map_user_message_maps_content_name_and_role() -> None:
    message = UserMessage("question", name="customer")

    mapped = ToOpenAIChatCompletionMapper.map_user_message(message)

    assert mapped == {
        "content": "question",
        "role": "user",
        "name": "customer",
    }


def test_map_user_message_rejects_unsupported_role() -> None:
    message = UserMessage("question")

    with (
        patch.object(message, "get_role_name", return_value="assistant"),
        pytest.raises(ValueError, match="Expected role 'user', got 'assistant'"),
    ):
        ToOpenAIChatCompletionMapper.map_user_message(message)


def test_map_assistant_message_maps_content_name_and_role() -> None:
    message = AssistantMessage("answer", name="agent")

    assert ToOpenAIChatCompletionMapper.map_assistant_message(message) == {
        "content": "answer",
        "role": "assistant",
        "name": "agent",
    }


def test_map_assistant_message_maps_tool_calls() -> None:
    message = AssistantMessage(
        "",
        tool_calls=[ToolCall("call-123", FunctionCall("search", {"query": "docs"}))],
    )

    assert ToOpenAIChatCompletionMapper.map_assistant_message(message) == {
        "content": "",
        "role": "assistant",
        "name": "",
        "tool_calls": [
            {
                "id": "call-123",
                "function": {
                    "name": "search",
                    "arguments": '{"query":"docs"}',
                },
                "type": "function",
            }
        ],
    }


def test_map_structured_assistant_message_serializes_pydantic_content() -> None:
    message = StructuredAssistantMessage(StructuredAnswer(value=42), name="agent")

    assert ToOpenAIChatCompletionMapper.map_structured_assistant_message(message) == {
        "content": '{"value":42}',
        "role": "assistant",
        "name": "agent",
    }


def test_map_structured_assistant_message_rejects_unsupported_role() -> None:
    message = StructuredAssistantMessage(StructuredAnswer(value=42))

    with (
        patch.object(message, "get_role_name", return_value="user"),
        pytest.raises(ValueError, match="Expected role 'assistant', got 'user'"),
    ):
        ToOpenAIChatCompletionMapper.map_structured_assistant_message(message)


def test_map_structured_assistant_message_rejects_non_model_content() -> None:
    message = StructuredAssistantMessage(StructuredAnswer(value=42))

    with (
        patch.object(message, "get_message", return_value=object()),
        pytest.raises(TypeError, match="Pydantic BaseModel"),
    ):
        ToOpenAIChatCompletionMapper.map_structured_assistant_message(message)


def test_map_structured_assistant_message_maps_tool_calls() -> None:
    message = StructuredAssistantMessage(
        StructuredAnswer(value=42),
        tool_calls=[
            ToolCall(
                id="call_123",
                function=FunctionCall(name="search", arguments={}),
            )
        ],
    )

    assert ToOpenAIChatCompletionMapper.map_structured_assistant_message(message) == {
        "content": '{"value":42}',
        "role": "assistant",
        "name": "",
        "tool_calls": [
            {
                "id": "call_123",
                "function": {"name": "search", "arguments": "{}"},
                "type": "function",
            }
        ],
    }


def test_map_tool_call_rejects_unsupported_tool_call() -> None:
    with pytest.raises(TypeError, match="Unsupported tool call type: object"):
        ToOpenAIChatCompletionMapper._map_tool_call(cast(ToolCall, object()))


def test_map_assistant_message_rejects_unsupported_role() -> None:
    message = AssistantMessage("answer")

    with (
        patch.object(message, "get_role_name", return_value="user"),
        pytest.raises(ValueError, match="Expected role 'assistant', got 'user'"),
    ):
        ToOpenAIChatCompletionMapper.map_assistant_message(message)


def test_map_tool_calls_returns_none_when_provider_omits_them() -> None:
    assert FromOpenAIChatCompletionMapper.map_tool_calls(None) is None


def test_map_tool_calls_rejects_unsupported_provider_type() -> None:
    with pytest.raises(
        TypeError,
        match="Unsupported OpenAI tool call type: SimpleNamespace",
    ):
        FromOpenAIChatCompletionMapper.map_tool_calls([SimpleNamespace()])


def test_map_tool_calls_resolves_registered_function_arguments() -> None:
    tool_call = ChatCompletionMessageFunctionToolCall(
        id="call-123",
        type="function",
        function=OpenAIFunction(name="search", arguments='{"query":"docs"}'),
    )

    mapped = FromOpenAIChatCompletionMapper.map_tool_calls(
        [tool_call],
        Tools([Tool(search)]),
    )

    assert mapped == [ToolCall("call-123", FunctionCall("search", {"query": "docs"}))]


def test_map_tool_calls_requires_tools_for_function_call() -> None:
    tool_call = ChatCompletionMessageFunctionToolCall(
        id="call-123",
        type="function",
        function=OpenAIFunction(name="search", arguments="{}"),
    )

    with pytest.raises(ValueError, match="Tools are required"):
        FromOpenAIChatCompletionMapper.map_tool_calls([tool_call])


def test_map_tool_calls_rejects_deprecated_custom_call() -> None:
    tool_call = ChatCompletionMessageCustomToolCall(
        id="call-123",
        type="custom",
        custom=Custom(name="script", input="print('done')"),
    )

    with pytest.raises(DeprecationWarning, match="Custom tool calls are deprecated"):
        FromOpenAIChatCompletionMapper.map_tool_calls([tool_call])


def test_map_tool_message_maps_content_name_and_role() -> None:
    message = ToolMessage("result", "call_123", name="search")

    assert ToOpenAIChatCompletionMapper.map_tool_message(message) == {
        "content": "result",
        "role": "tool",
        "tool_call_id": "call_123",
    }


def test_map_tool_message_rejects_unsupported_role() -> None:
    message = ToolMessage("result", "call_123")

    with (
        patch.object(message, "get_role_name", return_value="assistant"),
        pytest.raises(ValueError, match="Expected role 'tool', got 'assistant'"),
    ):
        ToOpenAIChatCompletionMapper.map_tool_message(message)


def test_map_function_message_maps_content_name_and_role() -> None:
    message = FunctionMessage("result", "calculate")

    assert ToOpenAIChatCompletionMapper.map_function_message(message) == {
        "content": "result",
        "role": "function",
        "name": "calculate",
    }


def test_map_function_message_rejects_unsupported_role() -> None:
    message = FunctionMessage("result", "calculate")

    with (
        patch.object(message, "get_role_name", return_value="assistant"),
        pytest.raises(ValueError, match="Expected role 'function', got 'assistant'"),
    ):
        ToOpenAIChatCompletionMapper.map_function_message(message)


def test_map_user_message_rejects_unsupported_content_parts() -> None:
    class InvalidMessage(UserMessage):
        def iter_content_parts(self):
            yield 42

    with pytest.raises(TypeError, match="Unsupported user content type: int"):
        ToOpenAIChatCompletionMapper.map_user_message(InvalidMessage([]))


def test_map_user_message_maps_each_text_in_order() -> None:
    message = UserMessage(TextsUserMessage(["first", "second"]))

    assert ToOpenAIChatCompletionMapper.map_user_message(message)["content"] == [
        {"text": "first", "type": "text"},
        {"text": "second", "type": "text"},
    ]


def test_map_user_message_maps_each_image_to_data_url() -> None:
    images = [Image.new("RGB", (1, 1), color) for color in ("red", "blue")]
    message = UserMessage(ImagesUserMessage(images))

    with patch(
        "qs.core.providers.to_openai_chat_completion_mapper.pil_image_to_data_url",
        side_effect=["data:image/png;base64,first", "data:image/png;base64,second"],
    ):
        result = ToOpenAIChatCompletionMapper.map_user_message(message)["content"]

    assert result == [
        {"image_url": {"url": "data:image/png;base64,first"}, "type": "image_url"},
        {"image_url": {"url": "data:image/png;base64,second"}, "type": "image_url"},
    ]


def test_map_user_message_maps_each_document_with_its_index() -> None:
    message = UserMessage(
        FilesUserMessage(
            [
                DocumentUserMessage(b"first", "report", "pdf"),
                DocumentUserMessage(b"name,value", "records", "csv", "text/csv"),
            ]
        )
    )

    assert ToOpenAIChatCompletionMapper.map_user_message(message)["content"] == [
        {
            "type": "file",
            "file": {
                "filename": "report_0.pdf",
                "file_data": "data:application/pdf;base64,Zmlyc3Q=",
            },
        },
        {
            "type": "file",
            "file": {
                "filename": "records_1.csv",
                "file_data": "data:text/csv;base64,bmFtZSx2YWx1ZQ==",
            },
        },
    ]


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        ({"reasoning": "first", "thinking": "second"}, "first"),
        ({"reasoning": "  ", "reasoning_content": "detail"}, "detail"),
        ({"thinking": "thought"}, "thought"),
        ({"reasoning": None, "thinking": ""}, None),
        ({"content": "answer"}, None),
    ],
)
def test_thinking_from_uses_supported_fields_in_precedence_order(
    attributes, expected
) -> None:
    message = SimpleNamespace(**attributes)

    assert FromOpenAIChatCompletionMapper._thinking_from(message) == expected


def test_assistant_mapping_uses_the_public_tool_call_interface():
    class PublicCall(ToolCall):
        def get_id(self):
            return "public-call"

        def get_name(self):
            return "public-tool"

        def get_arguments(self):
            return {"text": "public-argument"}

    message = AssistantMessage(
        None, tool_calls=[PublicCall("internal", FunctionCall("internal", {}))]
    )

    assert ToOpenAIChatCompletionMapper.map_assistant_message(message)[
        "tool_calls"
    ] == [
        {
            "id": "public-call",
            "type": "function",
            "function": {
                "name": "public-tool",
                "arguments": '{"text":"public-argument"}',
            },
        },
    ]
