"""Full deep-thinking workflow with real tools and a controlled call sequence."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from qs.core.agents.capabilities.defaults import BUILTIN_CAPABILITIES_PATH
from qs.core.agents.deep_agent.deep_agent import DeepAgent
from qs.core.agents.node import Node
from qs.core.cache.session_cache import SessionCache
from qs.core.models.assistant_message import AssistantMessage, FunctionCall, ToolCall
from qs.core.models.model_thinking import ModelThinking
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.reasoning_effort import ReasoningEffortLMH
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.core.models.tool_message import ToolMessage
from qs.interfaces.provider import Provider
from qs.interfaces.tokenizer import Tokenizer
from qs.providers.local import LocalProvider

pytestmark = pytest.mark.live

STATISTICS_URL = "https://docs.python.org/3/library/statistics.html"
MATH_URL = "https://docs.python.org/3/library/math.html"


class _Tokenizer(Tokenizer):
    def __init__(self, session: SessionCache) -> None:
        self.session = session
        self.next_id = 0

    def count_tokens(self, text: str) -> int:
        return len(text)

    def truncate_text(self, text: str, start_tokens: int, end_tokens: int = 0) -> str:
        return text[:start_tokens] + (text[-end_tokens:] if end_tokens else "")

    def initialize(self) -> None:
        pass

    def generate_id(self) -> str:
        self.next_id += 1
        return f"live-output-{self.next_id}"

    def get_session(self) -> SessionCache:
        return self.session


class _ScriptedProvider(Provider):
    def __init__(
        self,
        session: SessionCache,
        responses: list[tuple[AssistantMessage, str | None]],
    ) -> None:
        super().__init__(_Tokenizer(session))
        self.responses = responses
        self.requests: list[ProviderRequest] = []
        self.search_retries = 0

    def call(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        latest_message = request.get_messages()[-1]
        if (
            isinstance(latest_message, ToolMessage)
            and latest_message.get_name() == "search-web"
        ):
            search_result = _json_body(latest_message.get_message())
            if (
                search_result.get("error_kind") == "search_failed"
                and self.search_retries < 2
            ):
                self.search_retries += 1
                time.sleep(2)
                retry = _call(
                    "search-web",
                    f"search-retry-{self.search_retries}",
                    query="Python statistics mean standard library documentation",
                    max_results=5,
                )
                return ProviderResponse(
                    AssistantMessage(None, tool_calls=[retry]),
                    ModelThinking("Retry transient source discovery."),
                )
        message, thinking = self.responses.pop(0)
        return ProviderResponse(
            message, ModelThinking(thinking) if thinking is not None else None
        )

    def call_structured[ResponseModel: BaseModel](
        self, request: StructuredProviderRequest[ResponseModel]
    ) -> StructuredProviderResponse[ResponseModel]:
        raise NotImplementedError


def _call(name: str, call_id: str, **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, function=FunctionCall(name, arguments))


def _scripted_responses() -> list[tuple[AssistantMessage, str | None]]:
    steps = [
        (_call("deep-thinking", "open"), None),
        (
            _call(
                "search-web",
                "search",
                query="Python statistics mean standard library documentation",
                max_results=5,
            ),
            "Find the official source.",
        ),
        (
            _call("read-web-page", "read", url=STATISTICS_URL),
            "Read the official reference.",
        ),
        (
            _call(
                "interact-web-page",
                "navigate",
                url=STATISTICS_URL,
                actions=[{"kind": "navigate", "url": MATH_URL}],
            ),
            "Check a related page on the same site.",
        ),
        (
            _call(
                "calculate", "calculate", operation="evaluate", expression="(3+5+7)/3"
            ),
            "Compute the sample mean.",
        ),
        (
            _call(
                "logic-consistency",
                "consistency",
                variables={"m": "Real"},
                premises=[{"op": "=", "left": {"var": "m"}, "right": {"const": 5}}],
            ),
            "Check the formal premise is consistent.",
        ),
        (
            _call(
                "logic-check",
                "check",
                variables={"m": "Real"},
                premises=[{"op": "=", "left": {"var": "m"}, "right": {"const": 5}}],
                conclusion={
                    "op": ">",
                    "left": {"var": "m"},
                    "right": {"const": 4},
                },
            ),
            "Check a consequence of the calculated mean.",
        ),
        (
            _call(
                "logic-counterexample",
                "counterexample",
                variables={"m": "Real"},
                premises=[{"op": ">", "left": {"var": "m"}, "right": {"const": 0}}],
                conclusion={
                    "op": ">",
                    "left": {"var": "m"},
                    "right": {"const": 10},
                },
            ),
            "Find a model refuting an overstrong claim.",
        ),
        (_call("exit_agent", "exit", text="The sample mean is 5.", items=[]), None),
    ]
    return [
        (AssistantMessage(None, tool_calls=[tool_call]), thinking)
        for tool_call, thinking in steps
    ] + [(AssistantMessage("The sample mean is 5."), None)]


def _tool_messages(requests: list[ProviderRequest]) -> dict[str, str]:
    return {
        message.get_tool_call_id(): message.get_message()
        for request in requests
        for message in request.get_messages()
        if isinstance(message, ToolMessage)
    }


def _json_body(content: str) -> dict[str, object]:
    return json.loads(content.split("</header>", 1)[1].strip())


def test_live_full_skill_uses_every_tool_in_one_task(tmp_path: Path) -> None:
    with SessionCache(tmp_path / "cache") as session:
        provider = _ScriptedProvider(session, _scripted_responses())
        node = Node([BUILTIN_CAPABILITIES_PATH], name_hds="main")
        node.initialize(provider, work_root=tmp_path)
        response = node.run(
            "Find the official statistics documentation, navigate to the related "
            "math page, calculate the mean of 3, 5, 7, and check formal claims about it."
        )

        assert response.get_message() == "The sample mean is 5."
        assert provider.responses == []
        outputs = _tool_messages(provider.requests)
        assert {
            "search",
            "read",
            "navigate",
            "calculate",
            "consistency",
            "check",
            "counterexample",
            "open",
        } <= outputs.keys()
        search_call_id = (
            f"search-retry-{provider.search_retries}"
            if provider.search_retries
            else "search"
        )
        assert _json_body(outputs[search_call_id])["result_count"] > 0
        assert "statistics" in outputs["read"].lower()
        assert MATH_URL in outputs["navigate"]
        assert _json_body(outputs["calculate"])["result"] == "5.00000000000000"
        assert _json_body(outputs["consistency"])["status"] == "consistent"
        assert _json_body(outputs["check"])["status"] == "valid"
        assert _json_body(outputs["counterexample"])["status"] == "found"
        for call_id in (
            "search",
            "read",
            "navigate",
            "calculate",
            "consistency",
            "check",
            "counterexample",
        ):
            assert "<thinking>" in outputs[call_id]


def test_live_model_can_choose_and_combine_every_skill_tool(tmp_path: Path) -> None:
    load_dotenv()
    model = os.getenv("QS_LIVE_DEEP_THINKING_MODEL")
    if not model:
        pytest.skip("set QS_LIVE_DEEP_THINKING_MODEL for the model-driven macro test")
    provider = LocalProvider(
        model=model,
        reasoning_effort=ReasoningEffortLMH.LOW,
        api="responses",
    )
    node = Node([BUILTIN_CAPABILITIES_PATH], name_hds="main")
    response = DeepAgent(provider=provider, node=node, work_root=tmp_path).run(
        "Use deep-thinking to complete this audit. Search for the official Python "
        "statistics.mean documentation, read that page, navigate on the same "
        "docs.python.org site to the math module page with interact-web-page, "
        "calculate the mean of 3, 5, 7 with calculate. Use logic-consistency "
        "to check that m = 5 is consistent, logic-check to prove m > 4 from "
        "m = 5, and logic-counterexample to refute m > 10 from m > 0. Use all "
        "seven tools and then report the mean and source URLs."
    )
    names = {
        call["function"]["name"]
        for message in response.history
        for call in json.loads(message.get("tool_calls", "null")) or []
    }
    assert {
        "deep-thinking",
        "search-web",
        "read-web-page",
        "interact-web-page",
        "calculate",
        "logic-check",
        "logic-consistency",
        "logic-counterexample",
    } <= names
