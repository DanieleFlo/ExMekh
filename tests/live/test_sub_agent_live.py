from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

import qs
from qs.core.agents.deep_agent.agent_response import AgentResponse
from qs.core.agents.deep_agent.deep_agent import DeepAgent
from qs.core.agents.node import Node
from qs.core.models.reasoning_effort import ReasoningEffortLMH
from tests.live.model_support import LiveModelProvider

pytestmark = pytest.mark.live


def build_live_sub_agent_system(
    *,
    parent_persistent: bool = True,
    child_persistent: bool = True,
    parent_context_id: str = "main-subagent-live",
) -> DeepAgent:
    load_dotenv()
    model = os.getenv("QS_LIVE_MODEL", "Qwen3.8-27B-UD-Q4_K_S")
    provider = LiveModelProvider(
        model=model,
        reasoning_effort=ReasoningEffortLMH.LOW,
        api="responses",
    )
    capabilities_path = Path(qs.__file__).parent / "capabilities"
    child = Node(
        capabilities_paths=[capabilities_path],
        name_hds="mock-sub-agent",
        persistent_history=child_persistent,
        agent_context_id="mock-sub-agent",
    )
    parent = Node(
        capabilities_paths=[capabilities_path],
        name_hds="main",
        children=[child],
        persistent_history=parent_persistent,
        agent_context_id=parent_context_id,
    )
    return DeepAgent(provider=provider, node=parent, work_root=Path.cwd())


def get_sub_agent_tool_calls(
    response: AgentResponse,
    *,
    since: int = 0,
) -> list[tuple[str, dict[str, object]]]:
    calls: list[tuple[str, dict[str, object]]] = []
    for message in response.history[since:]:
        serialized_calls = message.get("tool_calls")
        if not serialized_calls or serialized_calls == "null":
            continue

        parsed_calls = json.loads(serialized_calls)
        assert isinstance(parsed_calls, list), (
            "CLIENT: assistant tool calls were not serialized as a list"
        )
        for call in parsed_calls:
            function = call.get("function")
            assert isinstance(function, dict), (
                "CLIENT: serialized tool call has no function object"
            )
            name = function.get("name")
            arguments = function.get("arguments")
            if name == "call_sub_agent" and isinstance(arguments, dict):
                calls.append((name, arguments))
    return calls


@pytest.mark.parametrize(
    "child_persistent", [True, False], ids=["persistent", "ephemeral"]
)
def test_live_sub_agent_memory_matches_history_policy(child_persistent: bool) -> None:
    code = "ALPHA-PERSIST-7712"
    agent = build_live_sub_agent_system(
        child_persistent=child_persistent,
        parent_context_id=f"parent-memory-{child_persistent}",
    )
    first = agent.run(
        "Call call_sub_agent with sub_agent='mock-sub-agent' "
        f"and text='Remember code {code}. Reply with only an acknowledgement.'. "
        "After delegation, reply only ACK."
    )
    assert get_sub_agent_tool_calls(first), (
        "MODEL: expected delegation to store the code"
    )
    second = agent.run(
        "Call call_sub_agent with sub_agent='mock-sub-agent' "
        "and text='What code did you save earlier? Reply with only the code or NO SAVED CODE.'. "
        "Then return only the sub-agent's answer."
    )
    assert get_sub_agent_tool_calls(second, since=len(first.history)), (
        "MODEL: expected delegation to recall the code"
    )
    results = [
        message["content"]
        for message in second.history[len(first.history) :]
        if message.get("role") == "tool"
    ]
    assert results, "CLIENT: missing sub-agent result"
    if child_persistent:
        assert any(code.casefold() in result.casefold() for result in results), (
            f"MODEL: stored code was not recalled: {results!r}"
        )
        assert code.casefold() in (second.get_message() or "").casefold()
    else:
        assert all(code.casefold() not in result.casefold() for result in results), (
            f"CLIENT: ephemeral child retained the stored code: {results!r}"
        )
        assert any("no saved code" in result.casefold() for result in results), (
            f"MODEL: expected NO SAVED CODE: {results!r}"
        )
