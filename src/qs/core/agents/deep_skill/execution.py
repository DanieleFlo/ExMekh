from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from qs.core.agents.deep_skill.deep_skill import DeepSkill
from qs.interfaces.message import Message

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DeepSkillExecution:
    skill: DeepSkill
    opening_tool_call_id: str
    start_index: int
    end_index: int | None = None
    output: str | None = None
    history: list[Message] = field(default_factory=list)
    children: list[DeepSkillExecution] = field(default_factory=list)

    @classmethod
    def open(
        cls,
        skill: DeepSkill,
        opening_tool_call_id: str,
        start_index: int,
    ) -> DeepSkillExecution:
        return cls(
            skill=deepcopy(skill),
            opening_tool_call_id=opening_tool_call_id,
            start_index=start_index,
        )

    def close(
        self,
        *,
        end_index: int,
        output: str,
        history: list[Message],
    ) -> None:
        if self.output is not None:
            raise RuntimeError("Deep skill execution is already closed")

        self.end_index = end_index
        self.output = output
        self.history = history

    def history_tree(self) -> dict[str, Any]:
        return {
            "skill": self.skill.name,
            "output": self.output,
            "history": [message.message_dump() for message in self.history],
            "children": [child.history_tree() for child in self.children],
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
    def current(self) -> DeepSkillExecution:
        if not self._active:
            raise RuntimeError("No deep skill execution is active")
        return self._active[-1]

    def open(
        self,
        skill: DeepSkill,
        opening_tool_call_id: str,
        start_index: int,
    ) -> DeepSkillExecution:
        execution = DeepSkillExecution.open(
            skill,
            opening_tool_call_id,
            start_index,
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
        history: list[Message],
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
