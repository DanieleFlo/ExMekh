"""Opt-in DS4 checks: unsupported attachment errors must leave history usable."""

import os
from uuid import uuid4

import pytest
from PIL import Image
from pydantic import BaseModel

from qs.core.models.client import Client
from qs.core.models.client_request import ClientRequest
from qs.core.models.reasoning_effort import ReasoningEffortLMH
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    ImagesUserMessage,
    UserMessage,
)
from qs.interfaces.provider import ProviderRequestRejectedError
from qs.providers.local import LocalProvider

pytestmark = pytest.mark.live


class Reference(BaseModel):
    reference: str


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_rejected_attachments_allow_same_client_to_continue(api):
    url = os.getenv("QS_LIVE_REJECTION_URL")
    if not url:
        pytest.skip("Set QS_LIVE_REJECTION_URL to the updated DS4 endpoint")
    provider = LocalProvider(
        model=os.getenv("QS_LIVE_MODEL", "Qwen3.8-27B-UD-Q4_K_S"),
        api=api,
        base_url=url,
        api_key="local-test",
        reasoning_effort=ReasoningEffortLMH.OFF,
    )
    provider.client = provider.client.with_options(timeout=300, max_retries=0)
    client = Client(
        provider, SystemMessage("Follow the requested output format exactly.")
    )
    limit = "max_tokens" if api == "chat_completions" else "max_output_tokens"
    extra = {limit: 32, "temperature": 0, "extra_body": {"thinking": False}}
    secret = uuid4().hex[:12]
    try:
        initial = client.call(
            ClientRequest(
                UserMessage(f"Remember reference {secret}. Reply READY only."),
                extra=extra,
                tool_choice="none",
            )
        )
        assert (initial.get_message() or "").strip() == "READY"
        before = client.get_history()
        for index, extension in enumerate(
            ["txt", "csv", "json", "xml", "pdf", "docx", "xlsx", "pptx", "zip", "wav"]
        ):
            message = UserMessage(
                [
                    "Read the attachment.",
                    FilesUserMessage(
                        [
                            DocumentUserMessage(
                                b"Reference: discarded",
                                "attachment",
                                extension,
                                "application/octet-stream",
                            )
                        ]
                    ),
                ]
            )
            request = ClientRequest(
                message, extra=extra, tool_choice="none", streaming=index % 2 == 0
            )
            with pytest.raises(
                ProviderRequestRejectedError, match="only text and images are supported"
            ):
                if index % 3 == 0:
                    client.call_structured(request, Reference)
                else:
                    client.call(request)
            assert client.get_history() == before
        if os.getenv("QS_LIVE_REJECTION_VISION") == "1":
            mixed = UserMessage(
                [
                    ImagesUserMessage([Image.new("RGB", (256, 256), "blue")]),
                    FilesUserMessage([DocumentUserMessage(b"data", "report", "pdf")]),
                    "Read everything.",
                ]
            )
            with pytest.raises(
                ProviderRequestRejectedError, match="only text and images are supported"
            ):
                client.call(ClientRequest(mixed, extra=extra, tool_choice="none"))
            assert client.get_history() == before
            image = client.call(
                ClientRequest(
                    UserMessage(
                        [
                            ImagesUserMessage([Image.new("RGB", (256, 256), "blue")]),
                            "Name the dominant color. One word.",
                        ]
                    ),
                    extra=extra,
                    tool_choice="none",
                )
            )
            assert (image.get_message() or "").strip().lower() == "blue"
            before = client.get_history()
        final = client.call(
            ClientRequest(
                UserMessage(
                    "Repeat the reference I asked you to remember before the rejected attachments. Only the reference."
                ),
                extra=extra,
                tool_choice="none",
            )
        )
        assert (final.get_message() or "").strip() == secret
        assert len(client.get_history()) == len(before) + 2
    finally:
        provider.client.close()
