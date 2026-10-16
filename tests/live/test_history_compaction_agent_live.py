"""Real agent continuation after node-selected and in-flight history compaction."""

from pathlib import Path

import pytest

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION
from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.agents.deep_agent.deep_agent import DeepAgent
from exmekh_lib.core.agents.deep_agent.stream_event import (
    DeepAgentExecutionEvent,
    DeepAgentExecutionEventType,
)
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.xml_header import XmlHeader
from tests.core.agents.node.test_node import _write_nested_capabilities
from tests.event_handler import as_agent_handler
from tests.history_validation import validate_request_history
from tests.live.history_compaction_support import (
    RecordingCompactionProvider,
    compaction_outputs,
    history_tokens,
    synthetic_history,
    write_observation,
)
from tests.live.model_support import fresh_live_context_id

pytestmark = pytest.mark.live


def node_messages(node):
    return node._runtime._client.get_history()


def make_node(capabilities, label, calls, *, children=()):
    def record(text: str) -> dict:
        calls.append(text)
        return {"recorded": text, "dispatched": False}

    return Node(
        [capabilities],
        name_hds="main",
        persistent_history=True,
        agent_context_id=fresh_live_context_id(label),
        children=children,
        default_capabilities=DefaultCapabilities(
            harness=False,
            overrides={
                "mock-tool": Tool(record, name="mock-tool", background_enabled=False)
            },
        ),
    )


def test_live_explicit_child_compaction_preserves_root_and_allows_real_tool_continuation(
    tmp_path: Path,
):
    provider = RecordingCompactionProvider("responses")
    root_calls, child_calls = [], []
    capabilities = _write_nested_capabilities(tmp_path / "capabilities-fixture")
    child = make_node(capabilities, "compact-child", child_calls)
    root = make_node(capabilities, "compact-root", root_calls, children=(child,))
    agent = DeepAgent(provider, root, tmp_path)
    seed = synthetic_history().get_history()[1:]
    root.restore_history(seed)
    child.restore_history(seed)
    root_before = [message.message_dump() for message in node_messages(root)]
    before = node_messages(child)
    try:
        agent.compact_history(node=child)
        after = node_messages(child)
        assert [
            message.message_dump() for message in node_messages(root)
        ] == root_before
        assert history_tokens(provider, after) < history_tokens(provider, before)
        assert compaction_outputs(after)
        validate_request_history(ProviderRequest(after))

        result = child.run(
            "Use mock-tool exactly once to record a follow-up containing the pending "
            "checkpoint identifier and target queue from the retained conversation. "
            "Do not dispatch anything or open a skill. After recording, briefly confirm."
        )
        assert isinstance(result.get_message(), str) and result.get_message().strip()
        assert root_calls == [], (
            "CLIENT: compaction or continuation executed in the root"
        )
        assert len(child_calls) == 1, "MODEL: follow-up was not recorded exactly once"
        assert "CP-618" in child_calls[0] and "staging" in child_calls[0], (
            "MODEL: retained workflow did not support the correct next action"
        )
        assert [
            message.message_dump() for message in node_messages(root)
        ] == root_before
        validate_request_history(ProviderRequest(node_messages(child)))
        write_observation(
            tmp_path / "live-child-compaction.json",
            provider,
            before,
            after,
            root_unchanged=True,
            followup_calls=len(child_calls),
            continuation_messages=len(result.history),
        )
    finally:
        provider.close()


def test_live_compaction_inside_nested_skills_keeps_scope_and_allows_both_exits(
    tmp_path,
):
    provider = RecordingCompactionProvider("responses")
    capabilities = _write_nested_capabilities(tmp_path / "capabilities-fixture")
    root = make_node(capabilities, "compact-active-skills", [])
    events = []
    captured = {}
    agent = None

    def observe(event):
        if not isinstance(event, DeepAgentExecutionEvent):
            return
        events.append(event)
        if (
            event.type is DeepAgentExecutionEventType.TOOL_END
            and event.tool_name == "nested-sub-tool"
            and not captured
        ):
            assert agent is not None
            captured["scope_ids"] = root._runtime._executor.skill_scope_ids
            captured["instructions"] = [
                message.get_message()
                for message in node_messages(root)
                if isinstance(message, ToolMessage)
                and message.get_name() in {"mock-skill", "nested-skill"}
            ]
            captured["before"] = node_messages(root)
            assert event.depth == 2
            agent.compact_history(node=root)

    def observe_request(request):
        if not captured or "after" in captured or request.get_extra() is None:
            return
        compacted = [
            message
            for message in request.get_messages()
            if isinstance(message, ToolMessage)
            and message.get_name() == HISTORY_COMPACTION.tool_name
            and XmlHeader.field_from_content(message.get_message(), "compacted")
            == "true"
        ]
        if not compacted:
            return
        assert root._runtime._executor.skill_scope_ids == captured["scope_ids"]
        instructions = [
            message.get_message()
            for message in request.get_messages()
            if isinstance(message, ToolMessage)
            and message.get_name() in {"mock-skill", "nested-skill"}
        ]
        assert instructions == captured["instructions"]
        captured["after"] = node_messages(root)
        assert history_tokens(provider, captured["after"]) < history_tokens(
            provider, captured["before"]
        )

    provider.on_request = observe_request
    agent = DeepAgent(provider, root, tmp_path, on_stream=as_agent_handler(observe))
    root.restore_history(synthetic_history().get_history()[1:])
    try:
        result = agent.run(
            "Perform this synthetic validation workflow in order: open mock-skill; "
            "inside it open nested-skill; inside the nested skill call nested-sub-tool "
            "once with text='checkpoint CP-618; target staging; still waiting for approval'. "
            "Then use exit_agent to return from the nested skill, and use exit_agent "
            "again to return from the outer skill, each with a concise status and "
            "items=[]. Finally give a brief status to the user. Do not dispatch, "
            "open any other skills or call any other work tools."
        )
        assert captured.get("after"), "MODEL: nested workflow did not reach compaction"
        assert len(captured["instructions"]) == 2
        assert root.depth == 0, "CLIENT: skill scope was left open"
        assert result.get_message() and result.get_message().strip()
        assert result.skill_executions[0].output is not None
        assert result.skill_executions[0].children[0].output is not None
        assert all(not event.failed for event in events), (
            "CLIENT: a tool or scope transition failed"
        )
        final_history = node_messages(root)
        validate_request_history(ProviderRequest(final_history))
        retained_outputs = compaction_outputs(captured["after"])
        assert all(
            entry in compaction_outputs(final_history) for entry in retained_outputs[:3]
        )
        assert history_tokens(provider, final_history) < history_tokens(
            provider, captured["before"]
        )
        scoring_index = next(
            index
            for index, item in enumerate(provider.observations)
            if item["stage"] == "scoring"
        )
        server_tokens_before = provider.observations[scoring_index - 1]["prompt_tokens"]
        resumed = [
            item["prompt_tokens"]
            for item in provider.observations[scoring_index + 1 :]
            if item["stage"] == "continuation"
        ]
        assert len(resumed) >= 3
        assert all(tokens < server_tokens_before for tokens in resumed), (
            "SERVER: a skill return restored history replaced by compaction",
            server_tokens_before,
            resumed,
        )
        catalog_request = provider.scoring_requests[0]
        catalog_text = catalog_request.get_messages()[-1].get_message()
        assert (
            "Mock skill body" not in catalog_text
            and "Nested skill body" not in catalog_text
        )
        write_observation(
            tmp_path / "live-nested-compaction.json",
            provider,
            captured["before"],
            captured["after"],
            scope_preserved=True,
            final_depth=root.depth,
            final_tokens=history_tokens(provider, final_history),
            server_tokens_before=server_tokens_before,
            resumed_server_prompt_tokens=resumed,
            final_history=[message.message_dump() for message in final_history],
            completed_tools=[
                event.tool_name
                for event in events
                if event.type is DeepAgentExecutionEventType.TOOL_END
            ],
        )
    finally:
        provider.close()
