from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from qs.core.agents.deep_skill.deep_skill import DeepSkill
from qs.core.models.tool_message import ToolMessage
from qs.interfaces.message import Message

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DeepSkillExecution:
    skill: DeepSkill
    opening_tool_call_id: str
    start_index: int
    opening_arguments: dict[str, Any] = field(default_factory=dict)
    opening_thinking: str | None = None
    opening_history_position: int = 0
    end_index: int | None = None
    output: str | None = None
    history: list[Message[object]] = field(default_factory=list)
    children: list[DeepSkillExecution] = field(default_factory=list)
    preceding_tool_responses: tuple[ToolMessage, ...] = ()

    @classmethod
    def open(
        cls,
        skill: DeepSkill,
        opening_tool_call_id: str,
        start_index: int,
        opening_arguments: dict[str, Any] | None = None,
        opening_thinking: str | None = None,
        preceding_tool_responses: tuple[ToolMessage, ...] = (),
        opening_history_position: int = 0,
    ) -> DeepSkillExecution:
        return cls(
            skill=deepcopy(skill),
            opening_tool_call_id=opening_tool_call_id,
            start_index=start_index,
            opening_arguments=deepcopy(opening_arguments or {}),
            opening_thinking=opening_thinking,
            opening_history_position=opening_history_position,
            preceding_tool_responses=deepcopy(preceding_tool_responses),
        )

    def close(
        self,
        *,
        end_index: int,
        output: str,
        history: list[Message[object]],
    ) -> None:
        if self.output is not None:
            raise RuntimeError("Deep skill execution is already closed")

        self.end_index = end_index
        self.output = output
        self.history = history

    def history_tree(
        self,
        current_history: list[Message[object]] | None = None,
    ) -> dict[str, Any]:
        history = self.history
        if self.output is None and current_history is not None:
            history = current_history[self.start_index :]
        return {
            "skill": self.skill.name,
            "output": self.output,
            "history": [message.message_dump() for message in history],
            "children": [
                child.history_tree(current_history) for child in self.children
            ],
        }


class DeepSkillExecutionTracker:
    def __init__(self) -> None:
        self._active: list[DeepSkillExecution] = []
        self._roots: list[DeepSkillExecution] = []

    @property
    def depth(self) -> int:
        return len(self._active)

    @property
    def roots(self) -> tuple[DeepSkillExecution, ...]:
        return tuple(self._roots)

    @property
    def active_skills(self) -> tuple[DeepSkill, ...]:
        return tuple(execution.skill for execution in self._active)

    def is_current_skill(self, name: str) -> bool:
        """Report the innermost active skill without exposing the stack."""
        return bool(self._active and self._active[-1].skill.name == name)

    @property
    def active_path(self) -> str:
        return "/" + "/".join(skill.name for skill in self.active_skills)

    @property
    def active_scope_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{execution.opening_tool_call_id}:{execution.opening_history_position}"
            for execution in self._active
        )

    def history_tree(
        self,
        current_history: list[Message[object]] | None = None,
    ) -> list[dict[str, str]]:
        return [execution.history_tree(current_history) for execution in self._roots]

    def get_current_start_index(self) -> int:
        return self.current.start_index

    @property
    def current(self) -> DeepSkillExecution:
        if not self._active:
            raise RuntimeError("No deep skill execution is active")
        return self._active[-1]

    def open(
        self,
        skill: DeepSkill,
        opening_tool_call_id: str,
        start_index: int,
        opening_arguments: dict[str, Any] | None = None,
        opening_thinking: str | None = None,
        preceding_tool_responses: tuple[ToolMessage, ...] = (),
        opening_history_position: int = 0,
    ) -> DeepSkillExecution:
        execution = DeepSkillExecution.open(
            skill,
            opening_tool_call_id,
            start_index,
            opening_arguments,
            opening_thinking,
            preceding_tool_responses,
            opening_history_position,
        )
        if self._active:
            self._active[-1].children.append(execution)
            parent_name = self._active[-1].skill.name
        else:
            self._roots.append(execution)
            parent_name = None
        self._active.append(execution)
        logger.debug(
            "Tracked deep skill opening: name=%s parent=%s active=%d roots=%d",
            skill.name,
            parent_name,
            len(self._active),
            len(self._roots),
        )
        return execution

    def close(
        self,
        *,
        end_index: int,
        output: str,
        history: list[Message[object]],
    ) -> DeepSkillExecution:
        if not self._active:
            raise RuntimeError("Cannot close a deep skill when none is active")

        execution = self._active.pop()
        execution.close(end_index=end_index, output=output, history=history)
        logger.debug(
            "Tracked deep skill closure: name=%s remaining_active=%d messages=%d",
            execution.skill.name,
            len(self._active),
            len(history),
        )
        return execution
