import base64
from io import BytesIO

import pytest
from openai.types.chat import ChatCompletion
from openai.types.responses import ResponseCompletedEvent, ResponseTextDeltaEvent
from PIL import Image
from pydantic import BaseModel

from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    ImagesUserMessage,
    TextsUserMessage,
    UserMessage,
)
from exmekh_lib.core.providers.chat_completion_openai import ChatCompletionOpenAI
from exmekh_lib.core.providers.response_openai import ResponseOpenAI
from exmekh_lib.core.providers.to_openai_chat_completion_mapper import (
    ToOpenAIChatCompletionMapper,
)
from exmekh_lib.core.providers.to_openai_response_mapper import ToOpenAIResponseMapper
from tests.core.providers.test_chat_completion_openai import make_stream
from tests.core.providers.test_response_openai import make_response


class ImageAnswer(BaseModel):
    value: int


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("content_kind", ["images_only", "interleaved"])
def test_provider_sends_one_ordered_mixed_message(
    openai_client, api, streaming, structured, content_kind
):
    images = [
        Image.new("RGB", (2, 3), "red"),
        Image.new("RGBA", (3, 2), (0, 0, 255, 128)),
    ]
    document = DocumentUserMessage(b"file nonce", "notes", "txt", "text/plain")
    if content_kind == "images_only":
        content = ImagesUserMessage(images)
        expected = images
    else:
        content = [
            "First image:",
            ImagesUserMessage([images[0]]),
            TextsUserMessage(["Between <&>", "Second line\nwith newline"]),
            FilesUserMessage([document]),
            "Second image:",
            ImagesUserMessage([images[1]]),
            "Compare in order.",
        ]
        expected = [
            "First image:",
            images[0],
            "Between <&>",
            "Second line\nwith newline",
            document,
            "Second image:",
            images[1],
            "Compare in order.",
        ]
    messages = [
        SystemMessage("Describe the supplied images."),
        UserMessage(content),
    ]
    reply = '{"value":42}'
    if api == "chat":
        completion = ChatCompletionOpenAI(openai_client, "vision-model")
        create = openai_client.chat.completions.create
        response = ChatCompletion.model_validate(
            {
                "id": "image-response",
                "created": 1,
                "model": "vision-model",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": reply},
                    }
                ],
            }
        )
        create.return_value = make_stream(reply) if streaming else response
        messages_key, text_type, image_type = "messages", "text", "image_url"
    else:
        completion = ResponseOpenAI(openai_client, "vision-model")
        create = openai_client.responses.create
        response = make_response(reply)
        create.return_value = (
            [
                ResponseTextDeltaEvent.model_construct(delta=reply),
                ResponseCompletedEvent.model_construct(response=response),
            ]
            if streaming
            else response
        )
        messages_key, text_type, image_type = "input", "input_text", "input_image"

    if structured:
        request = StructuredProviderRequest(messages, ImageAnswer, stream=streaming)
        result = completion.call_structured(request)
        assert result.get_message() == ImageAnswer(value=42)
    else:
        result = completion.call(ProviderRequest(messages, stream=streaming))
        assert result.get_message() == reply

    create.assert_called_once()
    payload = create.call_args.kwargs
    assert payload["stream"] is streaming
    parts = payload[messages_key][1]["content"]
    assert len(parts) == len(expected)
    for part, original in zip(parts, expected, strict=True):
        if isinstance(original, str):
            assert part == {"type": text_type, "text": original}
        elif isinstance(original, DocumentUserMessage):
            file = part["file"] if api == "chat" else part
            assert part["type"] == ("file" if api == "chat" else "input_file")
            assert file["filename"] == "notes_0.txt"
            assert file["file_data"] == "data:text/plain;base64,ZmlsZSBub25jZQ=="
        else:
            assert part["type"] == image_type
            url = part["image_url"]["url"] if api == "chat" else part["image_url"]
            prefix, encoded = url.split(",", maxsplit=1)
            assert prefix == "data:image/png;base64"
            with Image.open(BytesIO(base64.b64decode(encoded))) as decoded:
                assert decoded.size == original.size
                assert decoded.mode == original.mode
                assert decoded.tobytes() == original.tobytes()


@pytest.mark.parametrize(
    ("mapper", "text_type"),
    [
        (ToOpenAIChatCompletionMapper, "text"),
        (ToOpenAIResponseMapper, "input_text"),
    ],
)
def test_provider_maps_list_with_text_only(mapper, text_type):
    mapped = mapper.map_user_message(UserMessage(["Describe the next image."]))
    message = mapped[0] if isinstance(mapped, list) else mapped

    assert message["content"] == [
        {"type": text_type, "text": "Describe the next image."}
    ]


@pytest.mark.parametrize(
    "mapper", [ToOpenAIChatCompletionMapper, ToOpenAIResponseMapper]
)
@pytest.mark.parametrize(
    "content",
    [
        [],
        [TextsUserMessage(), ImagesUserMessage(), FilesUserMessage()],
    ],
)
def test_empty_mixed_content_maps_to_an_empty_parts_list(mapper, content):
    mapped = mapper.map_user_message(UserMessage(content))
    message = mapped[0] if isinstance(mapped, list) else mapped

    assert message["content"] == []


@pytest.mark.parametrize(
    "mapper", [ToOpenAIChatCompletionMapper, ToOpenAIResponseMapper]
)
def test_file_indices_are_unique_across_containers_in_one_message(mapper):
    documents = [
        DocumentUserMessage(value, "notes", "txt", "text/plain")
        for value in (b"first", b"second", b"third")
    ]
    content = [
        FilesUserMessage([documents[0]]),
        "between",
        FilesUserMessage(documents[1:]),
    ]

    mapped = mapper.map_user_message(UserMessage(content))
    message = mapped[0] if isinstance(mapped, list) else mapped
    files = [
        part.get("file", part)
        for part in message["content"]
        if part["type"] in ("file", "input_file")
    ]

    assert [file["filename"] for file in files] == [
        "notes_0.txt",
        "notes_1.txt",
        "notes_2.txt",
    ]
    assert [base64.b64decode(file["file_data"].split(",", 1)[1]) for file in files] == [
        document.file for document in documents
    ]


@pytest.mark.parametrize(
    ("mapper", "text_type"),
    [
        (ToOpenAIChatCompletionMapper, "text"),
        (ToOpenAIResponseMapper, "input_text"),
    ],
)
def test_provider_uses_the_message_content_iterator(mapper, text_type):
    class IterableMessage(UserMessage):
        def iter_content_parts(self):
            yield "provided through the public interface"

    mapped = mapper.map_user_message(IterableMessage([]))
    message = mapped[0] if isinstance(mapped, list) else mapped

    assert message["content"] == [
        {"type": text_type, "text": "provided through the public interface"},
    ]
