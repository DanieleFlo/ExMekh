"""Opt-in checks of mixed content against a real image-capable model endpoint."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from PIL import Image

from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    ImagesUserMessage,
    TextsUserMessage,
    UserMessage,
)
from tests.live.model_support import CacheTestTokenizer, LiveModelProvider

pytestmark = pytest.mark.live


@pytest.fixture(params=["chat_completions", "responses"])
def mixed_live_client(request, tmp_path):
    load_dotenv()
    if not os.getenv(DEFAULT_PROVIDER_SETTINGS.base_url_env_var):
        pytest.skip("Set BASE_URL to an image-capable model endpoint.")
    provider = LiveModelProvider(
        model=os.getenv("EXMEKH_LIB_LIVE_MODEL", DEFAULT_PROVIDER_SETTINGS.model),
        api=request.param,
        reasoning_effort=ReasoningEffortLMH.OFF,
        tokenizer=CacheTestTokenizer("mixed-content-live", tmp_path / "cache"),
    )
    client = Client(
        provider,
        SystemMessage(
            "Read supplied text and attachments. Follow the output format exactly."
        ),
    )
    limit_name = (
        "max_tokens" if request.param == "chat_completions" else "max_output_tokens"
    )
    try:
        yield client, {"temperature": 0, limit_name: 96}, provider
    finally:
        provider.client.close()


@pytest.mark.parametrize("layout", ["text_then_images", "interleaved", "text_groups"])
def test_live_text_and_images_in_one_ordered_message(mixed_live_client, layout):
    client, extra, _provider = mixed_live_client
    first = Image.new("RGB", (256, 256), "red")
    second = Image.new("RGB", (256, 256), "blue")
    reference = "EXMEKH-LIB-MIX-731"
    instructions = (
        "Reply with the dominant color of the FIRST image, the dominant color "
        "of the SECOND image, and the reference token from the user text, "
        "in that order separated by |. Use lowercase color names. No other text."
    )
    if layout == "text_then_images":
        content = [
            f"Reference: {reference}",
            ImagesUserMessage([first, second]),
            instructions,
        ]
    elif layout == "interleaved":
        content = [
            ImagesUserMessage([first]),
            f"Reference: {reference}",
            ImagesUserMessage([second]),
            instructions,
        ]
    else:
        content = [
            TextsUserMessage([instructions, f"Reference: {reference}"]),
            ImagesUserMessage([first]),
            "Second image:",
            ImagesUserMessage([second]),
        ]

    response = client.call(
        ClientRequest(UserMessage(content), extra=extra, tool_choice="none")
    )
    reply = response.get_message() or ""
    assert reply.strip().casefold() == f"red|blue|{reference}".casefold(), (
        f"MODEL: mixed image/text content or order was not understood: {reply!r}"
    )
    followup = client.call(
        ClientRequest(
            UserMessage(
                "Using the preceding message, repeat the two colors and reference token "
                "in the same order separated by |. No other text."
            ),
            extra=extra,
            tool_choice="none",
        )
    )
    assert (
        followup.get_message() or ""
    ).strip().casefold() == f"red|blue|{reference}".casefold(), (
        f"MODEL: mixed input was not retained across turns: {followup.get_message()!r}"
    )


def test_live_real_photo_and_text(mixed_live_client):
    image_path = os.getenv("EXMEKH_LIB_LIVE_IMAGE_PATH")
    if not image_path:
        pytest.skip("Set EXMEKH_LIB_LIVE_IMAGE_PATH to the Earth photo fixture.")
    with Image.open(Path(image_path)) as source:
        photo = source.copy()
    client, extra, _provider = mixed_live_client
    reference = "EXMEKH-LIB-PHOTO-594"
    response = client.call(
        ClientRequest(
            UserMessage(
                [
                    f"Reference: {reference}",
                    ImagesUserMessage([photo]),
                    TextsUserMessage(
                        [
                            "Name the main object in the photo and repeat the reference token.",
                            "Reply with object|token only.",
                        ]
                    ),
                ]
            ),
            extra=extra,
            tool_choice="none",
        )
    )
    reply = (response.get_message() or "").strip().casefold()
    assert ("earth" in reply or "planet" in reply) and reference.casefold() in reply, (
        f"MODEL: real photo or accompanying text was not understood: {reply!r}"
    )


@pytest.mark.parametrize(
    "layout", ["single_file", "file_and_text", "text_image_file_text"]
)
def test_live_native_file_content_is_read(mixed_live_client, layout):
    client, extra, provider = mixed_live_client
    reference = "EXMEKH-LIB-FILE-482"
    attachment = FilesUserMessage(
        [
            DocumentUserMessage(
                f"Reference: {reference}".encode(), "attachment", "txt", "text/plain"
            )
        ]
    )
    instructions = (
        "Read the attached file and reply with its reference token only. "
        "If the file is unavailable, reply UNREADABLE. Do not guess."
    )
    content = [attachment, instructions]
    if layout == "single_file":
        client = Client(
            provider,
            SystemMessage(instructions),
        )
        content = attachment
    if layout == "text_image_file_text":
        content = [
            "The attachment contains the token to return.",
            ImagesUserMessage([Image.new("RGB", (256, 256), "red")]),
            attachment,
            instructions,
        ]
    response = client.call(
        ClientRequest(UserMessage(content), extra=extra, tool_choice="none")
    )
    reply = (response.get_message() or "").strip()
    assert reply == reference, (
        f"MODEL: native file content was not read; the endpoint may not support file inputs: {reply!r}"
    )
