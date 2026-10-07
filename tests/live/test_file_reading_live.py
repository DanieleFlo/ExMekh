"""Opt-in real model reads of text, an image, and selected PDF page images."""

import os

import pytest
from dotenv import load_dotenv

from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from tests.file_reading_support import (
    initialize_reading_node,
    make_read_fixture,
    verify_read_client_flow,
)
from tests.live.model_support import CacheTestTokenizer, LiveModelProvider

pytestmark = pytest.mark.live


class RecordingReadProvider(LiveModelProvider):
    def __init__(self, directory):
        self.requests = []
        super().__init__(
            model=os.getenv("EXMEKH_LIB_LIVE_MODEL", "Qwen3.8-27B-UD-Q4_K_S"),
            api=os.getenv("EXMEKH_LIB_API", "responses"),
            reasoning_effort=ReasoningEffortLMH.OFF,
            tokenizer=CacheTestTokenizer("live-file-reading", directory / "cache"),
        )

    def call(self, request):
        calls = [
            c.get_name()
            for m in request.get_messages()
            if isinstance(m, AssistantMessage)
            for c in m.get_tool_calls() or ()
        ]
        assert len(self.requests) < 12, (
            f"MODEL: exceeded file-reading turn budget; calls={calls}"
        )
        self.requests.append(request)
        return super().call(request)


@pytest.fixture
def read_provider(tmp_path):
    load_dotenv()
    if not os.getenv("BASE_URL"):
        pytest.skip("Set BASE_URL to a live image-capable model endpoint.")
    provider = RecordingReadProvider(tmp_path)
    try:
        yield provider
    finally:
        provider.get_session().close()
        provider.client.close()


@pytest.mark.parametrize("kind", ["text", "image", "pdf"])
def test_live_client_reads_files_and_identifies_their_contents(
    tmp_path, read_provider, kind
):
    fixture = make_read_fixture(tmp_path, kind)
    node = initialize_reading_node(read_provider, tmp_path)
    response = node.run(fixture.prompt)
    verify_read_client_flow(read_provider.requests, node, fixture)
    reply = (response.get_message() or "").strip().casefold()
    assert fixture.answer.casefold() in {line.strip() for line in reply.splitlines()}, (
        f"MODEL: expected {fixture.answer!r}, got {reply!r}"
    )
    assert node.depth == 0, "MODEL: file-processing skill was not closed"
