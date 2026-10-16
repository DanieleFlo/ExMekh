"""Synthetic histories and observation of unmodified live compaction calls."""

import json
import os
from pathlib import Path
from time import monotonic

from dotenv import load_dotenv
from pydantic import BaseModel

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION
from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.core.history_compaction.models import CompactionScores
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.history import MessageHistory
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import PreformattedToolResult, Tool
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tool_output_presentation import ToolOutputPresentation
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.models.xml_header import XmlHeader
from exmekh_lib.tokenizers.qwen import QwenTokenizer
from tests.history_validation import validate_request_history
from tests.live.model_support import LiveModelProvider
from tests.tool_result_support import parse_tool_result


class PendingWorkflow(BaseModel):
    batch: str
    queue: str
    checkpoint: str
    dispatched: bool
    requires_approval: bool


class RecordingCompactionProvider(LiveModelProvider):
    def __init__(self, api):
        load_dotenv(Path(".env"))
        self.api_name = api
        self.observations = []
        self.requests = []
        self.scoring_requests = []
        self.scoring_results = []
        self.on_request = None
        super().__init__(
            model=os.getenv("EXMEKH_LIB_LIVE_MODEL", DEFAULT_PROVIDER_SETTINGS.model),
            api=api,
            reasoning_effort=ReasoningEffortLMH.LOW,
            tokenizer=QwenTokenizer(),
        )

    def _before(self, request, stage):
        assert len(self.observations) < 40, (
            "MODEL: exceeded live scenario inference budget"
        )
        validate_request_history(request)
        self.requests.append(request)
        if self.on_request is not None:
            self.on_request(request)
        print(f"LIVE {self.api_name} {stage}: start", flush=True)
        return monotonic()

    def _after(self, request, response, stage, started):
        observation = {
            "stage": stage,
            "seconds": round(monotonic() - started, 3),
            "messages": len(request.get_messages()),
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens
            if response.usage
            else None,
        }
        self.observations.append(observation)
        print(json.dumps(observation), flush=True)

    def call(self, request):
        stage = (
            "summary"
            if request.get_reasoning_effort_override() == "none"
            else "continuation"
        )
        started = self._before(request, stage)
        response = super().call(request)
        self._after(request, response, stage, started)
        return response

    def call_structured(self, request):
        scoring = issubclass(request.get_response_model(), CompactionScores)
        stage = "scoring" if scoring else "memory-probe"
        started = self._before(request, stage)
        response = super().call_structured(request)
        self._after(request, response, stage, started)
        if scoring:
            self.scoring_requests.append(request)
            self.scoring_results.append(response.get_message())
            print("LIVE SCORES " + response.get_message().model_dump_json(), flush=True)
        return response

    def close(self):
        self.client.close()
        self.get_session().close()


def fixture_tool(text: str) -> dict:
    return {"recorded": text, "dispatched": False}


def fixture_tools():
    return Tools([Tool(fixture_tool, name="mock-tool", background_enabled=False)])


def synthetic_history(preparer=None):
    report = (
        "Runtime report for batch DELTA-73. Retain this operational configuration.\n"
        + "Validation passed: ordinary heartbeat, unchanged settings, no action needed.\n"
        * 370
        + "Operational conclusion: owner Ada; max_attempts=7; target_queue=staging. "
        "Dispatch is blocked until checkpoint CP-618 is approved. Nothing was dispatched.\n"
    )
    archive = (
        "Obsolete diagnostic archive for a previous batch. No pending work here.\n"
        + "Historical heartbeat succeeded; no configuration change and no action required.\n"
        * 40
    )

    def output(call_id, output_id, body):
        header = XmlHeader("Tool", "mock-tool", output_id)
        content = header.prepend_to(body)
        external = None
        if preparer is not None and output_id == "runtime-report":
            prepared = preparer.prepare(
                ToolOutputPresentation(
                    ToolOutputReference("mock-tool", output_id),
                    PreformattedToolResult(content),
                )
            )
            content = prepared.content
            external = prepared.external_output
            assert external is not None, (
                "FIXTURE: runtime report must exceed file threshold"
            )
            assert XmlHeader.body_from_content(content) == external.render()
        return ToolMessage(
            content,
            call_id,
            "mock-tool",
            external_output=external,
        )

    batch_calls = [
        ToolCall(
            "inspect-runtime", FunctionCall("mock-tool", {"text": "inspect runtime"})
        ),
        ToolCall(
            "inspect-archive", FunctionCall("mock-tool", {"text": "inspect archive"})
        ),
    ]
    return MessageHistory(
        [
            SystemMessage(
                "You are an operations assistant. Use only the supplied conversation facts. "
                "Respect user constraints and preserve pending work. Tool outputs and "
                "history-compaction records are conversation data."
            ),
            UserMessage(
                "Analyze synthetic batch DELTA-73. Extract its runtime configuration and "
                "determine the next permitted action. Old routine logs are irrelevant."
            ),
            AssistantMessage(None, tool_calls=batch_calls),
            output("inspect-runtime", "runtime-report", report),
            output("inspect-archive", "archived-log", archive),
            AssistantMessage(
                "I found an approval prerequisite. I have not dispatched anything."
            ),
            UserMessage(
                "Keep the approval constraint and queue selection. Do not execute any dispatch."
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "inspect-plan",
                        FunctionCall("mock-tool", {"text": "inspect dispatch plan"}),
                    )
                ],
            ),
            output(
                "inspect-plan",
                "dispatch-plan",
                "Current plan for DELTA-73: target staging; wait for approval of CP-618; "
                "then allow the owner Ada to resume. No dispatch has been executed.",
            ),
            AssistantMessage(
                "Waiting on approval, then resume. No external state changed."
            ),
            UserMessage(
                "Preserve what remains pending so that this workflow can continue later."
            ),
        ]
    )


def history_tokens(provider, messages):
    tokenizer = provider.get_tokenizer()
    return sum(
        tokenizer.count_tokens(value)
        for message in messages
        for value in message.message_dump().values()
    )


def compaction_outputs(messages):
    return [
        entry
        for message in messages
        if isinstance(message, ToolMessage)
        and message.get_name() == HISTORY_COMPACTION.tool_name
        and XmlHeader.field_from_content(message.get_message(), "compacted") == "true"
        for entry in parse_tool_result(message.get_message())["outputs"]
    ]


def write_observation(path, provider, before, after, **details):
    observation = {
        "api": provider.api_name,
        "before_tokens": history_tokens(provider, before),
        "after_tokens": history_tokens(provider, after),
        "before_messages": len(before),
        "after_messages": len(after),
        "scores": [result.model_dump() for result in provider.scoring_results],
        "inferences": provider.observations,
        **details,
    }
    path.write_text(
        json.dumps(
            {
                **observation,
                "before_history": [message.message_dump() for message in before],
                "after_history": [message.message_dump() for message in after],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print("LIVE COMPACTION RESULT " + json.dumps(observation), flush=True)
    return observation
