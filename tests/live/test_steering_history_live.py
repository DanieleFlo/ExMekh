"""Real client/server reuse when steering supersedes a completed inference."""

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.interactive import InteractiveDefinition
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.providers.interactive_openai import InteractiveResponseOpenAI
from exmekh_lib.providers.local import LocalProvider
from tests.live.model_support import CacheTestTokenizer

pytestmark = pytest.mark.live
BASE = os.getenv("DS4_STEERING_CACHE_BASE_URL", "")


class SteeringHandler:
    requested = False

    def begin(self):
        pass

    def prepare(self, event, definition):
        pytest.fail("Only ordinary calls are requested")

    def validate_completed(self, calls):
        assert not self.requested

    def cancellation_requested(self):
        return self.requested


@pytest.mark.skipif(
    not BASE, reason="set DS4_STEERING_CACHE_BASE_URL to one idle server"
)
@pytest.mark.parametrize("streaming", [False, True])
def test_two_superseded_completions_keep_client_server_history(
    tmp_path, monkeypatch, streaming
):
    wire = []
    completed_result = InteractiveResponseOpenAI._completed_result

    def record_completed(self, result, request, contract):
        wire.append(result)
        return completed_result(self, result, request, contract)

    monkeypatch.setattr(
        InteractiveResponseOpenAI, "_completed_result", record_completed
    )

    def warm():
        pytest.fail("Protocol test calls must never execute")

    def patch(edits: str):
        pytest.fail("Protocol test calls must never execute")

    provider = LocalProvider(
        os.getenv("DS4_STEERING_CACHE_MODEL", "Qwen3.8-27B-UD-Q4_K_S"),
        api="responses",
        base_url=BASE.rstrip("/") + "/v1",
        reasoning_effort=ReasoningEffortLMH.OFF,
        tokenizer=CacheTestTokenizer("steering-live", tmp_path / "cache"),
    )
    tools = Tools(
        [
            Tool(
                warm,
                name="warm",
                description="Return a synthetic checkpoint",
                background_enabled=False,
            ),
            Tool(
                patch,
                name="patch",
                background_enabled=False,
                interactive=InteractiveDefinition(
                    "edit_file.v3", {"document": "edits"}
                ),
            ),
        ]
    )
    client = Client(
        provider,
        SystemMessage("Call warm exactly once whenever asked. Never call patch."),
        tools,
    )
    handler = SteeringHandler()
    client.interactive_handler = handler
    extra = {
        "agentic": {
            "agent_context_id": "steering-live:" + uuid4().hex,
            "allowed_tools": ["warm", "patch"],
            "allowed_skills": [],
        },
        "max_output_tokens": 128,
        "temperature": 0,
    }
    measurements = []
    try:
        archive = "\n".join(
            f"Archive {index}: retain source contents KEEP." for index in range(160)
        )
        response = client.call(
            ClientRequest(
                UserMessage(archive + "\nCall warm exactly once now."),
                extra=extra,
                tool_choice="required",
                streaming=streaming,
            )
        )
        assert [call.get_name() for call in response.get_tool_calls() or ()] == ["warm"]
        client.add_tool_response(
            ToolMessage(
                "Warm complete. Call warm once again.",
                response.get_tool_calls()[0].get_id(),
                "warm",
            )
        )
        for index in range(2):
            handler.requested = True
            cancelled = client.continue_after_tools(
                extra=extra, tool_choice="required", streaming=streaming
            )
            assert not cancelled.get_tool_calls()
            assert cancelled.superseded_response is not None
            assert [
                call.get_name()
                for call in cancelled.superseded_response.get_tool_calls() or ()
            ] == ["warm"]
            for interruption in cancelled.interruptions:
                client.add_user_message(interruption)
            baseline = wire[-1]["usage"]["input_tokens"]
            handler.requested = False
            resumed = client.continue_with_user_messages(
                [UserMessage("Steering: call warm exactly once now.")],
                extra=extra,
                tool_choice="required",
                streaming=streaming,
            )
            assert [call.get_name() for call in resumed.get_tool_calls() or ()] == [
                "warm"
            ]
            usage = wire[-1]["usage"]
            observed = {
                "iteration": index + 1,
                "streaming": streaming,
                "minimum_cached_tokens": baseline,
                "cached_tokens": usage["input_tokens_details"]["cached_tokens"],
                "input_tokens": usage["input_tokens"],
            }
            measurements.append(observed)
            print(json.dumps(observed), flush=True)
            assert observed["cached_tokens"] >= baseline
            client.add_tool_response(
                ToolMessage(
                    "Warm complete. Call warm once again.",
                    resumed.get_tool_calls()[0].get_id(),
                    "warm",
                )
            )
    finally:
        report = os.getenv("DS4_COMPLETED_STEERING_REPORT")
        if report:
            Path(report + ("-sse.json" if streaming else "-json.json")).write_text(
                json.dumps({"measurements": measurements, "responses": wire}, indent=2)
                + "\n"
            )
        provider.get_session().close()
        provider.client.close()
