"""Opt-in real model reads of text, an image, and selected PDF page images."""

import json
import os

import pytest
from dotenv import load_dotenv

from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.core.agents.node import DefaultCapabilities, Node
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from tests.file_reading_support import (
    make_read_fixture,
    verify_read_client_flow,
)
from tests.live.model_support import (
    CacheTestTokenizer,
    LiveModelProvider,
    fresh_live_context_id,
)

pytestmark = pytest.mark.live


class RecordingReadProvider(LiveModelProvider):
    def __init__(self, directory):
        self.directory = directory
        self.requests = []
        super().__init__(
            model=os.getenv("EXMEKH_LIB_LIVE_MODEL", DEFAULT_PROVIDER_SETTINGS.model),
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
            f"COVERAGE: exceeded file-reading turn budget; calls={calls}"
        )
        self.requests.append(request)
        result = super().call(request)
        (self.directory / f"live-request-{len(self.requests):02d}.json").write_text(
            json.dumps(
                {
                    "input": [
                        message.message_dump() for message in request.get_messages()
                    ],
                    "output": result.message_dump(),
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        return result


@pytest.fixture
def read_provider(tmp_path):
    load_dotenv()
    if not os.getenv(DEFAULT_PROVIDER_SETTINGS.base_url_env_var):
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
    hds = tmp_path / "capabilities" / "hds"
    hds.mkdir(parents=True)
    (hds / "main.md").write_text(
        "---\ndescription: Read workspace files.\nexit-tool: exit_agent\nsub-tools:\n"
        "  - inspect-file\n  - read-file\n---\n"
        "Use the file tools directly. After the requested read, reply DONE and stop."
    )
    node = Node(
        [hds.parent],
        name_hds="main",
        agent_context_id=fresh_live_context_id("live-file-read"),
        default_capabilities=DefaultCapabilities(
            harness=False, include={"inspect-file", "read-file", "exit_agent"}
        ),
    )
    node.initialize(read_provider, work_root=tmp_path)
    prerequisite = (
        "Read the image directly, without requiring an inspection."
        if kind == "image"
        else "Inspect the file before reading."
    )
    response = node.run(
        prerequisite
        + " Call read-file with this argument object: "
        + json.dumps(fixture.arguments())
        + ". After its result, reply DONE and stop."
    )
    verify_read_client_flow(read_provider.requests, node, fixture)
    assert isinstance(response.get_message(), str), (
        "CLIENT: expected a final text response"
    )
