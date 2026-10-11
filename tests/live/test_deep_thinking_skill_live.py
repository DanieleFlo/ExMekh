"""Opt-in end-to-end deep-thinking check with the real model and public web."""

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
from dotenv import load_dotenv

from exmekh_lib.core.agents.capabilities.defaults import BUILTIN_CAPABILITIES_PATH
from exmekh_lib.core.agents.deep_agent.deep_agent import DeepAgent
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from tests.live.model_support import LiveModelProvider, fresh_live_context_id

pytestmark = pytest.mark.live


class DeepThinkingLiveProvider(LiveModelProvider):
    """Bound the macro workflow and preserve actual calls for diagnosis."""

    def __init__(self, model: str, directory: Path, *, max_turns: int = 24):
        self._trace_path = directory / "macro-model-trace.json"
        self._turns: list[list[dict[str, object]]] = []
        self._max_turns = max_turns
        super().__init__(
            model=model, reasoning_effort=ReasoningEffortLMH.LOW, api="responses"
        )

    def call(self, request: ProviderRequest) -> ProviderResponse:
        if len(self._turns) >= self._max_turns:
            raise AssertionError(
                f"MODEL: exceeded {self._max_turns} macro turns; last calls={self._turns[-5:]}"
            )
        result = super().call(request)
        self._turns.append(
            [
                {"name": call.get_name(), "arguments": deepcopy(call.get_arguments())}
                for call in result.get_tool_calls() or []
            ]
        )
        self._trace_path.write_text(
            json.dumps(self._turns, ensure_ascii=True, indent=2)
        )
        return result


def test_live_model_can_choose_and_combine_every_skill_tool(tmp_path: Path) -> None:
    load_dotenv()
    model = os.getenv("EXMEKH_LIB_LIVE_DEEP_THINKING_MODEL")
    if not model:
        pytest.skip(
            "set EXMEKH_LIB_LIVE_DEEP_THINKING_MODEL for the model-driven macro test"
        )
    provider = DeepThinkingLiveProvider(model, tmp_path)
    node = Node(
        [BUILTIN_CAPABILITIES_PATH],
        name_hds="main",
        agent_context_id=fresh_live_context_id("live-deep-thinking"),
    )
    try:
        response = DeepAgent(provider=provider, node=node, work_root=tmp_path).run(
            "Use deep-thinking to complete this audit. Search for the official Python "
            "statistics.mean documentation, read that page, navigate on the same "
            "docs.python.org site to the math module page with interact-web-page, "
            "calculate the mean of 3, 5, 7 with calculate. Use logic-consistency "
            "to check that m = 5 is consistent, logic-check to prove m > 4 from "
            "m = 5, and logic-counterexample to refute m > 10 from m > 0. Use all "
            "seven tools and then report only the mean and source URLs."
        )
    finally:
        provider.get_session().close()
        provider.client.close()
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
