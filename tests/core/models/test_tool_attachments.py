import base64

import pytest
from PIL import Image

from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    ImagesUserMessage,
    UserMessage,
)
from exmekh_lib.core.providers.to_openai_chat_completion_request_mapper import (
    ToOpenAIChatCompletionRequestMapper,
)
from exmekh_lib.core.providers.to_openai_response_request_mapper import (
    ToOpenAIResponseRequestMapper,
)


@pytest.mark.parametrize(
    "mapper", [ToOpenAIChatCompletionRequestMapper, ToOpenAIResponseRequestMapper]
)
def test_provider_sends_restored_attachments_after_complete_tool_batch(mapper):
    image = Image.new("RGB", (2, 2), "red")
    document = DocumentUserMessage(b"notes", "notes", "txt", "text/plain")
    companion = UserMessage(
        [
            "Saved guidance",
            ImagesUserMessage([image]),
            "Supporting file",
            FilesUserMessage([document]),
        ]
    )
    request = ProviderRequest(
        [
            SystemMessage("system"),
            UserMessage("start"),
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "read", FunctionCall("read_tool_output", {"output_id": "saved"})
                    ),
                    ToolCall("sibling", FunctionCall("other", {})),
                ],
            ),
            ToolMessage("saved text", "read", attachments=companion),
            ToolMessage("other output", "sibling"),
        ]
    )
    messages = request.get_messages()
    assert [m.get_role_name() for m in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
    ]
    assert request.get_messages() == messages
    mapped = mapper.map_request(request, model="test")
    if mapper is ToOpenAIChatCompletionRequestMapper:
        items = mapped["messages"]
        assert [m["tool_call_id"] for m in items if m["role"] == "tool"] == [
            "read",
            "sibling",
        ]
        content = items[-1]["content"]
        assert [p["type"] for p in content] == ["text", "image_url", "text", "file"]
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert (
            base64.b64decode(content[3]["file"]["file_data"].split(",", 1)[1])
            == b"notes"
        )
    else:
        items = mapped["input"]
        assert [
            m["call_id"] for m in items if m.get("type") == "function_call_output"
        ] == ["read", "sibling"]
        content = items[-1]["content"]
        assert [p["type"] for p in content] == [
            "input_text",
            "input_image",
            "input_text",
            "input_file",
        ]
        assert content[1]["image_url"].startswith("data:image/png;base64,")
        assert base64.b64decode(content[3]["file_data"].split(",", 1)[1]) == b"notes"


@pytest.mark.parametrize(
    "factory",
    [
        lambda attachments: PreformattedToolResult("text", attachments=attachments),
        lambda attachments: ToolMessage("text", "read", attachments=attachments),
    ],
)
def test_tool_attachments_require_a_user_message(factory):
    with pytest.raises(TypeError, match="attachments must be a UserMessage"):
        factory("invalid")
