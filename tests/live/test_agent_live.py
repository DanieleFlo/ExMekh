from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

import qs
from qs.core.agents.deep_agent.deep_agent import DeepAgent
from qs.core.agents.node import Node
from qs.core.models.reasoning_effort import ReasoningEffortLMH
from qs.providers.local import LocalProvider

pytestmark = pytest.mark.live


def build_live_agent() -> DeepAgent:
    load_dotenv()
    model = os.getenv("QS_LIVE_MODEL", "qwen3.5-9b-mtp")
    provider = LocalProvider(
        model=model,
        reasoning_effort=ReasoningEffortLMH.LOW,
    )
    capabilities_path = Path(qs.__file__).parent / "capabilities"
    node = Node(
        capabilities_paths=[capabilities_path],
        name_hds="main",
        persistent_history=False,
    )
    return DeepAgent(provider=provider, node=node)


def test_live_agent_answers_a_greeting() -> None:
    response = build_live_agent().run("Ciao")

    message = response.get_message()
    assert isinstance(message, str) and message.strip(), (
        "MODEL: expected a non-empty greeting response"
    )


def test_live_agent_uses_an_available_tool() -> None:
    token = "LIVE-TOOL-OK"
    response = build_live_agent().run(
        f"Usa il tool mock-tool una sola volta con il testo esatto {token}. "
        f"Poi rispondi con una frase breve che contenga {token}."
    )

    tool_messages = [
        message for message in response.history if message.get("role") == "tool"
    ]
    assert tool_messages, "MODEL: expected mock-tool to be called"
    assert token in tool_messages[-1]["content"], (
        f"MODEL: mock-tool result does not contain {token!r}"
    )

    message = response.get_message()
    assert isinstance(message, str) and token in message, (
        f"MODEL: final response does not contain {token!r}"
    )
