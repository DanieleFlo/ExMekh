from types import SimpleNamespace
from unittest.mock import patch

import pytest
from openai.types.chat.chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall,
)
from openai.types.chat.chat_completion_message_custom_tool_call import (
    Custom as OpenAICustom,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    Function as OpenAIFunction,
)

from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.function_message import FunctionMessage
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool_message import ToolMessage
from qs.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    ImagesUserMessage,
    TextsUserMessage,
    UserMessage,
)
from qs.core.providers.openai_chat_completion_mapper import (
    OpenAIChatCompletionMapper,
)


def test_extract_usage_returns_empty_usage_when_provider_omits_it() -> None:
    usage = OpenAIChatCompletionMapper.extract_usage(SimpleNamespace(usage=None))

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

    usage = OpenAIChatCompletionMapper.extract_usage(response)

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

    thinking = OpenAIChatCompletionMapper.extract_thinking(response)

    assert thinking is not None
    assert thinking.thoughts == "reasoning"


def test_extract_thinking_returns_none_when_no_choice_contains_it() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )

    assert OpenAIChatCompletionMapper.extract_thinking(response) is None


@pytest.mark.parametrize("role", ["system", "developer"])
def test_map_system_message_maps_supported_roles(role: str) -> None:
    message = SystemMessage("instructions", role=role, name="policy")

    mapped = OpenAIChatCompletionMapper.map_system_message(message)

    assert mapped == {
        "content": "instructions",
        "role": role,
        "name": "policy",
    }


def test_map_system_message_rejects_unsupported_role() -> None:
    message = SystemMessage("instructions", role="assistant")

    with pytest.raises(
        ValueError, match="Expected role 'system' or 'developer', got 'assistant'"
    ):
        OpenAIChatCompletionMapper.map_system_message(message)


def test_map_user_message_maps_content_name_and_role() -> None:
    message = UserMessage("question", name="customer")

    mapped = OpenAIChatCompletionMapper.map_user_message(message)

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
        OpenAIChatCompletionMapper.map_user_message(message)


def test_map_assistant_message_maps_content_name_and_role() -> None:
    message = AssistantMessage("answer", name="agent")

    assert OpenAIChatCompletionMapper.map_assistant_message(message) == {
        "content": "answer",
        "role": "assistant",
        "name": "agent",
    }


def test_map_assistant_message_maps_tool_calls() -> None:
    message = AssistantMessage(
        None,
        tool_calls=[
            {
                "id": "call_123",
                "function": {"name": "search", "arguments": "{}"},
                "type": "function",
            },
            {
                "id": "call_456",
                "custom": {
                    "name": "execute_python",
                    "input": "print(2 + 2)",
                },
                "type": "custom",
            },
        ],
    )

    assert OpenAIChatCompletionMapper.map_assistant_message(message) == {
        "content": None,
        "role": "assistant",
        "name": "",
        "tool_calls": [
            {
                "id": "call_123",
                "function": {"name": "search", "arguments": "{}"},
                "type": "function",
            },
            {
                "id": "call_456",
                "custom": {
                    "name": "execute_python",
                    "input": "print(2 + 2)",
                },
                "type": "custom",
            },
        ],
    }


def test_map_assistant_message_rejects_unsupported_role() -> None:
    message = AssistantMessage("answer", role="user")

    with pytest.raises(ValueError, match="Expected role 'assistant', got 'user'"):
        OpenAIChatCompletionMapper.map_assistant_message(message)


def test_map_tool_calls_returns_none_when_provider_omits_them() -> None:
    assert OpenAIChatCompletionMapper.map_tool_calls(None) is None


def test_map_tool_calls_maps_function_and_custom_calls() -> None:
    tool_calls = [
        ChatCompletionMessageFunctionToolCall(
            id="call_123",
            function=OpenAIFunction(name="search", arguments='{"query":"docs"}'),
            type="function",
        ),
        ChatCompletionMessageCustomToolCall(
            id="call_456",
            custom=OpenAICustom(
                name="execute_python",
                input="print(2 + 2)",
            ),
            type="custom",
        ),
    ]

    assert OpenAIChatCompletionMapper.map_tool_calls(tool_calls) == [
        {
            "id": "call_123",
            "function": {
                "name": "search",
                "arguments": '{"query":"docs"}',
            },
            "type": "function",
        },
        {
            "id": "call_456",
            "custom": {
                "name": "execute_python",
                "input": "print(2 + 2)",
            },
            "type": "custom",
        },
    ]


def test_map_tool_calls_rejects_unsupported_provider_type() -> None:
    with pytest.raises(
        TypeError,
        match="Unsupported OpenAI tool call type: SimpleNamespace",
    ):
        OpenAIChatCompletionMapper.map_tool_calls([SimpleNamespace()])


def test_map_tool_message_maps_content_name_and_role() -> None:
    message = ToolMessage("result", "call_123", name="search")

    assert OpenAIChatCompletionMapper.map_tool_message(message) == {
        "content": "result",
        "role": "tool",
        "tool_call_id": "call_123",
    }


def test_map_tool_message_rejects_unsupported_role() -> None:
    message = ToolMessage("result", "call_123", role="assistant")

    with pytest.raises(ValueError, match="Expected role 'tool', got 'assistant'"):
        OpenAIChatCompletionMapper.map_tool_message(message)


def test_map_function_message_maps_content_name_and_role() -> None:
    message = FunctionMessage("result", "calculate")

    assert OpenAIChatCompletionMapper.map_function_message(message) == {
        "content": "result",
        "role": "function",
        "name": "calculate",
    }


def test_map_function_message_rejects_unsupported_role() -> None:
    message = FunctionMessage("result", "calculate", role="assistant")

    with pytest.raises(ValueError, match="Expected role 'function', got 'assistant'"):
        OpenAIChatCompletionMapper.map_function_message(message)


def test_map_user_content_returns_string_unchanged() -> None:
    assert OpenAIChatCompletionMapper._map_user_content("question") == "question"


@pytest.mark.parametrize(
    ("content", "method_name"),
    [
        (TextsUserMessage(), "_map_text_parts"),
        (ImagesUserMessage(), "_map_image_parts"),
        (FilesUserMessage(), "_map_file_parts"),
    ],
)
def test_map_user_content_dispatches_structured_content(content, method_name) -> None:
    expected = [{"mapped": True}]

    with patch.object(
        OpenAIChatCompletionMapper, method_name, return_value=expected
    ) as mapper:
        result = OpenAIChatCompletionMapper._map_user_content(content)

    assert result is expected
    mapper.assert_called_once_with(content)


def test_map_user_content_rejects_unsupported_content() -> None:
    with pytest.raises(TypeError, match="Unsupported user content type: int"):
        OpenAIChatCompletionMapper._map_user_content(42)


def test_map_text_parts_maps_each_text_in_order() -> None:
    content = TextsUserMessage()
    content.texts = ["first", "second"]

    assert OpenAIChatCompletionMapper._map_text_parts(content) == [
        {"text": "first", "type": "text"},
        {"text": "second", "type": "text"},
    ]


def test_map_image_parts_maps_each_image_to_data_url() -> None:
    first_image = object()
    second_image = object()
    content = ImagesUserMessage()
    content.images = [first_image, second_image]

    with patch(
        "qs.core.providers.openai_chat_completion_mapper.pil_image_to_data_url",
        side_effect=["data:image/png;base64,first", "data:image/png;base64,second"],
    ) as convert:
        result = OpenAIChatCompletionMapper._map_image_parts(content)

    assert result == [
        {
            "image_url": {"url": "data:image/png;base64,first"},
            "type": "image_url",
        },
        {
            "image_url": {"url": "data:image/png;base64,second"},
            "type": "image_url",
        },
    ]
    assert [call.args[0] for call in convert.call_args_list] == [
        first_image,
        second_image,
    ]


def test_map_file_parts_maps_each_document_with_its_index() -> None:
    first = DocumentUserMessage(b"first", "report", "pdf")
    second = DocumentUserMessage(b"second", "data", "csv")
    content = FilesUserMessage()
    content.documents = [first, second]
    mapped = [{"file": "first"}, {"file": "second"}]

    with patch.object(
        OpenAIChatCompletionMapper,
        "_map_file_document",
        side_effect=mapped,
    ) as map_document:
        result = OpenAIChatCompletionMapper._map_file_parts(content)

    assert result == mapped
    assert map_document.call_args_list[0].args == (first, 0)
    assert map_document.call_args_list[1].args == (second, 1)


def test_map_file_document_builds_indexed_data_url() -> None:
    document = DocumentUserMessage(
        b"name,value", "records", "csv", mime_type="text/csv"
    )

    with patch(
        "qs.core.providers.openai_chat_completion_mapper.convert_bytes_to_base64",
        return_value="bmFtZSx2YWx1ZQ==",
    ) as convert:
        mapped = OpenAIChatCompletionMapper._map_file_document(document, 3)

    assert mapped == {
        "type": "file",
        "file": {
            "filename": "records_3.csv",
            "file_data": "data:text/csv;base64,bmFtZSx2YWx1ZQ==",
        },
    }
    convert.assert_called_once_with(b"name,value")


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

    assert OpenAIChatCompletionMapper._thinking_from(message) == expected
