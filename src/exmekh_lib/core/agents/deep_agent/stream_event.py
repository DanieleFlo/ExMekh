from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.core.models.provider_stream_event import ProviderStreamEvent


class DeepAgentExecutionEventType(StrEnum):
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"


@dataclass(frozen=True, slots=True)
class DeepAgentStreamEvent:
    provider_event: ProviderStreamEvent
    depth: int
    skill_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider_event, ProviderStreamEvent):
            raise TypeError(AGENT_TEXT.provider_event_must_be_a_providerstreamevent)
        _validate_position(self.depth, self.skill_path)


@dataclass(frozen=True, slots=True)
class DeepAgentExecutionEvent:
    type: DeepAgentExecutionEventType
    depth: int
    skill_path: str
    history: list[dict[str, Any]]
    history_tree: list[dict[str, Any]]
    tool_name: str
    tool_call_id: str
    failed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.type, DeepAgentExecutionEventType):
            raise TypeError(AGENT_TEXT.event_type_must_be_a_deepagentexecutioneventtype)
        _validate_position(self.depth, self.skill_path)
        if not isinstance(self.failed, bool):
            raise TypeError(AGENT_TEXT.execution_failure_must_be_a_boolean)
        if self.failed and self.is_tool_running:
            raise ValueError(AGENT_TEXT.a_running_tool_cannot_have_a_terminal_failure)
        if not isinstance(self.history, list):
            raise TypeError(AGENT_TEXT.history_must_be_a_list)
        if not isinstance(self.history_tree, list):
            raise TypeError(AGENT_TEXT.history_tree_must_be_a_list)
        if not isinstance(self.tool_name, str) or not self.tool_name:
            raise ValueError(AGENT_TEXT.execution_events_require_a_tool_name)
        if not isinstance(self.tool_call_id, str) or not self.tool_call_id:
            raise ValueError(AGENT_TEXT.execution_events_require_a_tool_call_id)

        object.__setattr__(self, "history", deepcopy(self.history))
        object.__setattr__(self, "history_tree", deepcopy(self.history_tree))

    @property
    def is_tool_running(self) -> bool:
        return self.type is DeepAgentExecutionEventType.TOOL_START


@dataclass(frozen=True, slots=True)
class DeepAgentHistoryCompactionEvent:
    depth: int
    skill_path: str
    history: list[dict[str, Any]]
    history_tree: list[dict[str, Any]]
    replaced_history: dict[str, list[dict[str, Any]]]
    agent_context_id: str = ""

    def __post_init__(self) -> None:
        _validate_position(self.depth, self.skill_path)
        for name in ("history", "history_tree", "replaced_history"):
            object.__setattr__(self, name, deepcopy(getattr(self, name)))


type DeepAgentEvent = (
    DeepAgentStreamEvent | DeepAgentExecutionEvent | DeepAgentHistoryCompactionEvent
)


def _validate_position(depth: int, skill_path: str) -> None:
    if not isinstance(depth, int) or isinstance(depth, bool):
        raise TypeError(AGENT_TEXT.depth_must_be_an_integer)
    if depth < 0:
        raise ValueError(AGENT_TEXT.depth_cannot_be_negative)
    if not isinstance(skill_path, str) or not skill_path.startswith("/"):
        raise ValueError(AGENT_TEXT.skill_path_must_start_with)
