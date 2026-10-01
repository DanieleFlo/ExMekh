from copy import deepcopy
from typing import cast

import pytest

from qs.core.agents.deep_agent.stream_event import (
    DeepAgentExecutionEvent,
    DeepAgentExecutionEventType,
    DeepAgentStreamEvent,
)
from qs.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)


def _execution_event(**overrides: object) -> DeepAgentExecutionEvent:
    arguments = {
        "type": DeepAgentExecutionEventType.TOOL_START,
        "depth": 0,
        "skill_path": "/",
        "history": [{"role": "user", "content": "hello"}],
        "history_tree": [],
        "tool_name": "mock-tool",
        "tool_call_id": "call-1",
    }
    arguments.update(overrides)
    return DeepAgentExecutionEvent(**arguments)


def test_stream_event_wraps_only_the_provider_event() -> None:
    provider_event = ProviderStreamEvent(
        ProviderStreamEventType.RESPONSE,
        content="token",
    )

    event = DeepAgentStreamEvent(provider_event, depth=2, skill_path="/parent/child")

    assert event.provider_event is provider_event
    assert event.depth == 2
    assert event.skill_path == "/parent/child"
    assert not hasattr(event, "history")


def test_stream_event_rejects_invalid_provider_event() -> None:
    with pytest.raises(TypeError, match="must be a ProviderStreamEvent"):
        DeepAgentStreamEvent(
            cast(ProviderStreamEvent, object()),
            depth=0,
            skill_path="/",
        )


def test_execution_event_copies_context_snapshots() -> None:
    history = [{"role": "user", "content": "hello"}]
    history_tree = [{"skill": "mock-skill", "children": []}]
    expected_history = deepcopy(history)
    expected_tree = deepcopy(history_tree)

    event = _execution_event(history=history, history_tree=history_tree)
    history[0]["content"] = "changed"
    history_tree[0]["skill"] = "changed"

    assert event.history == expected_history
    assert event.history_tree == expected_tree
    assert event.is_tool_running is True


def test_execution_events_expose_the_running_state() -> None:
    start = _execution_event(type=DeepAgentExecutionEventType.TOOL_START)
    end = _execution_event(type=DeepAgentExecutionEventType.TOOL_END)

    assert start.is_tool_running is True
    assert end.is_tool_running is False


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"type": "tool_start"}, TypeError, "Event type"),
        ({"failed": "true"}, TypeError, "failure must be a boolean"),
        ({"failed": True}, ValueError, "running tool cannot have a terminal failure"),
        ({"depth": "0"}, TypeError, "Depth must be an integer"),
        ({"depth": True}, TypeError, "Depth must be an integer"),
        ({"depth": -1}, ValueError, "Depth cannot be negative"),
        ({"skill_path": 1}, ValueError, "Skill path must start"),
        ({"skill_path": "mock-skill"}, ValueError, "Skill path must start"),
        ({"history": ()}, TypeError, "History must be a list"),
        ({"history_tree": ()}, TypeError, "History tree must be a list"),
        ({"tool_name": None}, ValueError, "require a tool name"),
        ({"tool_name": ""}, ValueError, "require a tool name"),
        ({"tool_call_id": None}, ValueError, "require a tool call id"),
        ({"tool_call_id": ""}, ValueError, "require a tool call id"),
    ],
)
def test_rejects_invalid_event_shapes(
    overrides: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _execution_event(**overrides)


def test_type_checker_escape_does_not_bypass_history_validation() -> None:
    with pytest.raises(TypeError, match="History must be a list"):
        _execution_event(history=cast(list[dict[str, object]], object()))


def test_terminal_execution_event_exposes_a_failure() -> None:
    event = _execution_event(type=DeepAgentExecutionEventType.TOOL_END, failed=True)
    assert event.failed
    assert not event.is_tool_running
