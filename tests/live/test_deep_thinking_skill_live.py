"""Opt-in end-to-end deep-thinking check with the real model and public web."""

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from exmekh_lib.core.agents.capabilities.defaults import BUILTIN_CAPABILITIES_PATH
from exmekh_lib.core.agents.deep_agent.deep_agent import DeepAgent
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from tests.live.model_support import LiveModelProvider

pytestmark = pytest.mark.live


def test_live_model_can_choose_and_combine_every_skill_tool(tmp_path: Path) -> None:
    load_dotenv()
    model = os.getenv("EXMEKH_LIB_LIVE_DEEP_THINKING_MODEL")
    if not model:
        pytest.skip(
            "set EXMEKH_LIB_LIVE_DEEP_THINKING_MODEL for the model-driven macro test"
        )
    provider = LiveModelProvider(
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
        "seven tools and then report only the mean and source URLs."
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
