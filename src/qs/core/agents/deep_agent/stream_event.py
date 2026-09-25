from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from qs.core.models.provider_stream_event import ProviderStreamEvent


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
            raise TypeError("provider_event must be a ProviderStreamEvent")
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

    def __post_init__(self) -> None:
        if not isinstance(self.type, DeepAgentExecutionEventType):
            raise TypeError("Event type must be a DeepAgentExecutionEventType")
        _validate_position(self.depth, self.skill_path)
        if not isinstance(self.history, list):
            raise TypeError("History must be a list")
        if not isinstance(self.history_tree, list):
            raise TypeError("History tree must be a list")
        if not isinstance(self.tool_name, str) or not self.tool_name:
            raise ValueError("Execution events require a tool name")
        if not isinstance(self.tool_call_id, str) or not self.tool_call_id:
            raise ValueError("Execution events require a tool call id")

        object.__setattr__(self, "history", deepcopy(self.history))
        object.__setattr__(self, "history_tree", deepcopy(self.history_tree))

    @property
    def is_tool_running(self) -> bool:
        return self.type is DeepAgentExecutionEventType.TOOL_START


type DeepAgentEvent = DeepAgentStreamEvent | DeepAgentExecutionEvent


def _validate_position(depth: int, skill_path: str) -> None:
    if not isinstance(depth, int) or isinstance(depth, bool):
        raise TypeError("Depth must be an integer")
    if depth < 0:
        raise ValueError("Depth cannot be negative")
    if not isinstance(skill_path, str) or not skill_path.startswith("/"):
        raise ValueError("Skill path must start with '/'")
