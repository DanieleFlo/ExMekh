from __future__ import annotations

import json
import logging
from typing import Any

from qs.core.agents.deep_skill.deep_skill import CalledDeepSkill, DeepSkill
from qs.core.agents.deep_skill.execution import (
    DeepSkillExecution,
    DeepSkillExecutionTracker,
)
from qs.core.agents.exit_skill.exit_skill import CalledExitSkill, ExitSkill
from qs.core.agents.exit_tool.exit_tool import CalledExitTool
from qs.core.models.client import Client
from qs.core.models.tool import CalledTool
from qs.core.models.tool_message import ToolMessage

from .tool_call_resolver import ResolvedToolCall

logger = logging.getLogger(__name__)


class ToolCallExecutor:
    """Execute resolved tool calls and track nested deep-skill state."""

    def __init__(
        self,
        client: Client,
        exit_skill: ExitSkill,
        initial_depth: int,
    ) -> None:
        self._client = client
        self._exit_skill = exit_skill
        self._initial_depth = initial_depth
        self._depth = initial_depth
        self._skill_executions = DeepSkillExecutionTracker()

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def skill_executions(self) -> tuple[DeepSkillExecution, ...]:
        return self._skill_executions.roots

    def execute_all(self, tool_calls: list[ResolvedToolCall]) -> None:
        logger.info(
            "Executing assistant tool calls: count=%d depth=%d",
            len(tool_calls),
            self._depth,
        )
        for tool_call in tool_calls:
            self.execute(tool_call)

    def execute(self, tool_call: ResolvedToolCall) -> None:
        if isinstance(tool_call, CalledExitSkill):
            self._run_exit_skill(tool_call)
            return
        if isinstance(tool_call, CalledDeepSkill):
            self._run_deep_skill(tool_call)
            return
        if isinstance(tool_call, CalledExitTool):
            self._run_exit_tool(tool_call)
            return
        if isinstance(tool_call, CalledTool):
            self._run_tool(tool_call)
            return
        raise TypeError(
            "Node supports only deep skills, exit skills, exit tools, and tools"
        )

    def _run_deep_skill(self, called_deep_skill: CalledDeepSkill) -> None:
        self._open_deep_skill(called_deep_skill)
        self._client.add_tool_response(
            ToolMessage(
                content=self._render_deep_skill(called_deep_skill.deep_skill),
                tool_call_id=called_deep_skill.tool_call.id,
                name=called_deep_skill.deep_skill.name,
            )
        )

    def _run_tool(self, called_tool: CalledTool) -> None:
        logger.info(
            "Executing tool: name=%s call_id=%s depth=%d",
            called_tool.name,
            called_tool.call_id,
            self._depth,
        )
        self._client.add_tool_response(
            ToolMessage(
                content=self.stringify_result(called_tool.execute()),
                tool_call_id=called_tool.call_id,
                name=called_tool.name,
            )
        )

    def _run_exit_skill(self, called_exit_skill: CalledExitSkill) -> None:
        logger.info(
            "Rendering exit skill: name=%s call_id=%s depth=%d",
            called_exit_skill.exit_skill.name,
            called_exit_skill.tool_call.id,
            self._depth,
        )
        self._client.add_tool_response(
            ToolMessage(
                content=called_exit_skill.exit_skill.get_body(),
                tool_call_id=called_exit_skill.tool_call.id,
                name=called_exit_skill.exit_skill.name,
            )
        )

    def _run_exit_tool(self, called_exit_tool: CalledExitTool) -> None:
        logger.info(
            "Executing exit tool: name=%s call_id=%s active_skills=%d depth=%d",
            called_exit_tool.exit_tool.name,
            called_exit_tool.call_id,
            self._skill_executions.depth,
            self._depth,
        )
        result = called_exit_tool.exit_tool.execute(
            **called_exit_tool.tool_call.get_arguments()
        )
        content = self.get_exit_output(result)
        if self._skill_executions.depth:
            self._close_deep_skill(content)
            return

        self._depth = max(0, self._depth - 1)
        logger.warning(
            "Exit tool executed without an active deep skill: call_id=%s depth=%d",
            called_exit_tool.call_id,
            self._depth,
        )
        self._client.add_tool_response(
            ToolMessage(
                content=content,
                tool_call_id=called_exit_tool.call_id,
                name=called_exit_tool.exit_tool.name,
            )
        )

    def _render_deep_skill(self, deep_skill: DeepSkill) -> str:
        return deep_skill.render_instructions(self._exit_skill)

    def _open_deep_skill(self, called_deep_skill: CalledDeepSkill) -> None:
        start_index = len(self._client.get_history()) - 1
        self._skill_executions.open(
            called_deep_skill.deep_skill,
            called_deep_skill.tool_call.id,
            start_index,
        )
        self._depth += 1
        logger.info(
            "Opened deep skill: name=%s call_id=%s start_index=%d depth=%d",
            called_deep_skill.deep_skill.name,
            called_deep_skill.tool_call.id,
            start_index,
            self._depth,
        )

    def _close_deep_skill(self, output: str) -> None:
        history = self._client.copy_history()
        start_index = self._skill_executions.current.start_index
        execution = self._skill_executions.close(
            end_index=len(history) - 1,
            output=output,
            history=history[start_index:],
        )
        response = ToolMessage(
            content=output,
            tool_call_id=execution.opening_tool_call_id,
            name=execution.skill.name,
        )
        self._client.replace_history_from(execution.start_index + 1, [response])
        self._depth = max(0, self._depth - 1)
        logger.info(
            "Closed deep skill: name=%s opening_call_id=%s start_index=%d "
            "end_index=%d transcript_messages=%d compacted_history=%d depth=%d",
            execution.skill.name,
            execution.opening_tool_call_id,
            execution.start_index,
            execution.end_index,
            len(execution.history),
            len(self._client.get_history()),
            self._depth,
        )

    @staticmethod
    def get_exit_output(result: Any) -> str:
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise TypeError("exit tool result must contain a text string")
        return result["text"]

    @staticmethod
    def stringify_result(result: Any) -> str:
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=True, default=str)
