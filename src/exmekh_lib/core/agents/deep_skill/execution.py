from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.interfaces.message import Message

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
    _used_tools: list[ToolOutputReference] = field(
        default_factory=list, init=False, repr=False
    )

    _started_clock: float = field(
        default_factory=lambda: monotonic(), init=False, repr=False
    )
    _finished_clock: float | None = field(default=None, init=False, repr=False)

    @property
    def elapsed_seconds(self) -> float:
        """Time since opening, fixed after closure, including nested work."""
        end = self._finished_clock if self._finished_clock is not None else monotonic()
        return end - self._started_clock

    @property
    def name(self) -> str:
        return self.skill.get_name()

    @property
    def scope_id(self) -> str:
        return f"{self.opening_tool_call_id}:{self.opening_history_position}"

    @property
    def used_tools(self) -> tuple[ToolOutputReference, ...]:
        return tuple(self._used_tools)

    def record_tool_output(self, reference: ToolOutputReference) -> None:
        if self.output is not None:
            raise RuntimeError(AGENT_TEXT.cannot_record_a_tool_output_in_a_closed_deep)
        self._used_tools.append(reference)

    def add_child(self, execution: DeepSkillExecution) -> None:
        if self.output is not None:
            raise RuntimeError(AGENT_TEXT.cannot_add_a_child_to_a_closed_deep_skill)
        self.children.append(execution)

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
            raise RuntimeError(AGENT_TEXT.deep_skill_execution_is_already_closed)

        self._finished_clock = monotonic()
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
            "skill": self.name,
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
        return bool(self._active and self._active[-1].name == name)

    @property
    def active_path(self) -> str:
        return "/" + "/".join(skill.get_name() for skill in self.active_skills)

    @property
    def active_scope_ids(self) -> tuple[str, ...]:
        return tuple(execution.scope_id for execution in self._active)

    def history_tree(
        self,
        current_history: list[Message[object]] | None = None,
    ) -> list[dict[str, Any]]:
        return [execution.history_tree(current_history) for execution in self._roots]

    @property
    def protected_history_indices(self) -> tuple[int, ...]:
        """Protect opening batches, including every active skill's instructions."""
        return tuple(execution.start_index for execution in self._active)

    def rebase_active_history(self, index_map: tuple[int, ...]) -> None:
        """Keep live opening anchors aligned without changing stable scope IDs."""
        positions = tuple(
            index_map[execution.start_index] for execution in self._active
        )
        for execution, position in zip(self._active, positions, strict=True):
            execution.start_index = position

    def get_current_start_index(self) -> int:
        return self.current.start_index

    def record_tool_output(self, name: str, output_id: str) -> None:
        """Keep output references after descendant histories are compacted."""
        reference = ToolOutputReference(name, output_id)
        for execution in self._active:
            execution.record_tool_output(reference)

    @property
    def current(self) -> DeepSkillExecution:
        if not self._active:
            raise RuntimeError(AGENT_TEXT.no_deep_skill_execution_is_active)
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
            self._active[-1].add_child(execution)
            parent_name = self._active[-1].name
        else:
            self._roots.append(execution)
            parent_name = None
        self._active.append(execution)
        logger.debug(
            AGENT_TEXT.tracked_deep_skill_opening_name_parent_active_roots,
            skill.get_name(),
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
            raise RuntimeError(AGENT_TEXT.cannot_close_a_deep_skill_when_none_is_active)

        execution = self.current
        execution.close(end_index=end_index, output=output, history=history)
        self._active.pop()
        logger.debug(
            AGENT_TEXT.tracked_deep_skill_closure_name_remaining_active_messages,
            execution.name,
            len(self._active),
            len(history),
        )
        return execution
