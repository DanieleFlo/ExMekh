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
from qs.providers.local import LocalProvider

pytestmark = pytest.mark.live


def build_live_sub_agent_system(
    *,
    parent_persistent: bool = True,
    child_persistent: bool = True,
    parent_context_id: str = "main-subagent-live",
) -> DeepAgent:
    load_dotenv()
    model = os.getenv("QS_LIVE_MODEL", "Qwen3.8-27B-UD-Q4_K_S")
    provider = LocalProvider(
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


def test_live_sub_agent_remembers_code_with_persistent_history() -> None:
    code = "ALPHA-PERSIST-7712"
    agent = build_live_sub_agent_system(
        parent_persistent=True,
        child_persistent=True,
        parent_context_id="parent-persist-subagent",
    )

    # Turn 1: Send secret code to sub-agent to memorize
    turn1_response = agent.run(
        "Call call_sub_agent exactly once with sub_agent='mock-sub-agent' "
        f"and text='Remember code {code}'. "
        "After the sub-agent responds, give a brief summary."
    )
    turn1_calls = get_sub_agent_tool_calls(turn1_response)
    assert turn1_calls, "MODEL: expected call_sub_agent in turn 1"
    turn1_tool_msgs = [
        msg for msg in turn1_response.history if msg.get("role") == "tool"
    ]
    assert turn1_tool_msgs, "MODEL: expected tool message in turn 1"
    assert any(
        "code recorded" in msg["content"].casefold() for msg in turn1_tool_msgs
    ), f"MODEL: expected registration confirmation in {turn1_tool_msgs!r}"

    # Turn 2: Ask sub-agent for the memorized code
    turn2_response = agent.run(
        "Call call_sub_agent exactly once with sub_agent='mock-sub-agent' "
        "and text='What code did you save earlier?'. "
        "Then return the sub-agent's response including the code."
    )
    turn2_calls = get_sub_agent_tool_calls(
        turn2_response, since=len(turn1_response.history)
    )
    assert turn2_calls, "MODEL: expected call_sub_agent in turn 2"
    turn2_tool_msgs = [
        msg
        for msg in turn2_response.history[len(turn1_response.history) :]
        if msg.get("role") == "tool"
    ]
    assert turn2_tool_msgs, "MODEL: expected tool message in turn 2"

    # With persistent_history=True, sub-agent remembers the code from turn 1
    assert any(
        code.casefold() in msg["content"].casefold() for msg in turn2_tool_msgs
    ), f"MODEL: sub-agent should remember {code!r}, got {turn2_tool_msgs!r}"
    assert code.casefold() in (turn2_response.get_message() or "").casefold(), (
        f"MODEL: parent should report {code!r}"
    )


def test_live_sub_agent_forgets_code_without_persistent_history() -> None:
    code = "BETA-FORGET-9934"
    agent = build_live_sub_agent_system(
        parent_persistent=True,
        child_persistent=False,
        parent_context_id="parent-ephemeral-subagent",
    )

    # Turn 1: Send secret code to sub-agent to memorize
    turn1_response = agent.run(
        "Call call_sub_agent exactly once with sub_agent='mock-sub-agent' "
        f"and text='Remember code {code}'. "
        "After the sub-agent responds, give a brief summary."
    )
    turn1_calls = get_sub_agent_tool_calls(turn1_response)
    assert turn1_calls, "MODEL: expected call_sub_agent in turn 1"

    # Turn 2: Ask sub-agent for the memorized code
    turn2_response = agent.run(
        "Call call_sub_agent exactly once with sub_agent='mock-sub-agent' "
        "and text='What code did you save earlier?'. "
        "Then return the sub-agent's response."
    )
    turn2_calls = get_sub_agent_tool_calls(
        turn2_response, since=len(turn1_response.history)
    )
    assert turn2_calls, "MODEL: expected call_sub_agent in turn 2"
    turn2_tool_msgs = [
        msg
        for msg in turn2_response.history[len(turn1_response.history) :]
        if msg.get("role") == "tool"
    ]
    assert turn2_tool_msgs, "MODEL: expected tool message in turn 2"

    # With persistent_history=False, the child session was reset and forgets the code
    sub_agent_results = [msg["content"] for msg in turn2_tool_msgs]
    assert all(
        code.casefold() not in result.casefold() for result in sub_agent_results
    ), f"MODEL: sub-agent should not remember {code!r}, got {sub_agent_results!r}"
    assert any("no saved code" in result.casefold() for result in sub_agent_results), (
        f"MODEL: expected NO SAVED CODE, got {sub_agent_results!r}"
    )


def test_live_direct_sub_agent_node_memory_persistence() -> None:
    load_dotenv()
    model = os.getenv("QS_LIVE_MODEL", "Qwen3.8-27B-UD-Q4_K_S")
    provider = LocalProvider(
        model=model,
        reasoning_effort=ReasoningEffortLMH.LOW,
        api="responses",
    )
    capabilities_path = Path(qs.__file__).parent / "capabilities"

    # 1. Test Node directly with persistent_history=True
    code_persist = "DIRECT-PERSIST-1010"
    node_persist = Node(
        capabilities_paths=[capabilities_path],
        name_hds="mock-sub-agent",
        persistent_history=True,
        agent_context_id="direct-persist-subagent",
    )
    node_persist.initialize(provider, work_root=Path.cwd())

    r1 = node_persist.run(f"Remember code {code_persist}")
    assert "code recorded" in (r1.get_message() or "").casefold()

    r2 = node_persist.run("What code did you save earlier?")
    assert code_persist.casefold() in (r2.get_message() or "").casefold()

    # 2. Test Node directly with persistent_history=False
    code_reset = "DIRECT-RESET-2020"
    node_reset = Node(
        capabilities_paths=[capabilities_path],
        name_hds="mock-sub-agent",
        persistent_history=False,
        agent_context_id="direct-reset-subagent",
    )
    node_reset.initialize(provider, work_root=Path.cwd())

    r3 = node_reset.run(f"Remember code {code_reset}")
    assert "code recorded" in (r3.get_message() or "").casefold()

    r4 = node_reset.run("What code did you save earlier?")
    assert code_reset.casefold() not in (r4.get_message() or "").casefold()
    assert "no saved code" in (r4.get_message() or "").casefold()
