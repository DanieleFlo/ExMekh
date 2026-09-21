from __future__ import annotations

import json
import logging
from typing import Any, Literal

from qs.core.agents.deep_skill.deep_skill import CalledDeepSkill
from qs.core.agents.deep_skill.execution import (
    DeepSkillExecution,
    DeepSkillExecutionTracker,
)
from qs.core.agents.exit_tool.exit_tool import CalledExitTool, ExitTool
from qs.core.models.client import Client
from qs.core.models.tool import CalledTool
from qs.core.models.tool_message import ToolMessage
from qs.core.models.xml_header import XmlHeader

from .agentic_scope import AgenticScope
from .tool_call_resolver import ResolvedToolCall
from .types import NodeExecutionSnapshot, ToolExecutionResult

logger = logging.getLogger(__name__)


class ToolCallExecutor:
    """Execute resolved tool calls and track nested deep-skill state."""

    def __init__(
        self,
        client: Client,
        exit_tool: ExitTool,
        initial_depth: int,
        agentic_scope: AgenticScope,
    ) -> None:
        self._client = client
        self._exit_tool = exit_tool
        self._depth = initial_depth
        self._agentic_scope = agentic_scope
        self._skill_executions = DeepSkillExecutionTracker()

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def skill_path(self) -> str:
        return self._skill_executions.active_path

    @property
    def skill_executions(self) -> tuple[DeepSkillExecution, ...]:
        return self._skill_executions.roots

    def snapshot(self) -> NodeExecutionSnapshot:
        messages = self._client.copy_history()
        return NodeExecutionSnapshot(
            depth=self._depth,
            skill_path=self._skill_executions.active_path,
            history=[message.message_dump() for message in messages],
            history_tree=self._skill_executions.history_tree(messages),
        )

    def build_agentic_extra(self) -> dict[str, dict[str, object]]:
        extra = self._agentic_scope.build_extra(
            self._skill_executions.active_skills,
            self._depth,
        )
        logger.info(
            "Agentic scope: path=%s depth=%d scope=%s",
            self.skill_path,
            self.depth,
            extra["agentic"],
        )
        return extra

    def acknowledge_agentic_extra(self) -> None:
        self._agentic_scope.acknowledge_extra()

    def execute(
        self, tool_call: ResolvedToolCall, *, close_scope: bool = True
    ) -> ToolExecutionResult:
        supported_types = (CalledDeepSkill, CalledExitTool, CalledTool)
        if not isinstance(tool_call, supported_types):
            raise TypeError("Node supports only deep skills, exit tools, and tools")

        if isinstance(tool_call, CalledExitTool):
            return self._run_exit_tool(tool_call, close_scope=close_scope)
        if isinstance(tool_call, CalledDeepSkill):
            return self._run_deep_skill(tool_call)
        return self._run_tool(tool_call)

    def _run_deep_skill(
        self, called_deep_skill: CalledDeepSkill
    ) -> ToolExecutionResult:
        self._open_deep_skill(called_deep_skill)
        content = called_deep_skill.render_instructions(
            self._exit_tool.render_prompt_section()
        )
        tool_response = ToolMessage(
            content=content,
            tool_call_id=called_deep_skill.call_id,
            name=called_deep_skill.name,
        )
        return ToolExecutionResult(response=tool_response)

    def _run_tool(self, called_tool: CalledTool) -> ToolExecutionResult:
        logger.info(
            "Executing tool: name=%s call_id=%s depth=%d",
            called_tool.name,
            called_tool.call_id,
            self._depth,
        )
        result = called_tool.execute()
        content, output_id = self._format_output(
            called_tool.name, self.stringify_result(result), source="Tool"
        )
        tool_response = ToolMessage(
            content=content,
            tool_call_id=called_tool.call_id,
            name=called_tool.name,
        )
        return ToolExecutionResult(
            response=tool_response,
            raw_result=result,
            save_result=True,
            output_id=output_id,
        )

    def _format_output(
        self, tool_name: str, content: str, *, source: Literal["Tool", "DeepSkill"]
    ) -> tuple[str, str | None]:
        try:
            header = XmlHeader(
                source=source, name=tool_name, id=self._client.generate_id()
            )
            return header.prepend_to(content), header.id
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            logger.error(error)
            return content, None

    def _run_exit_tool(
        self, called_exit_tool: CalledExitTool, *, close_scope: bool = True
    ) -> ToolExecutionResult:
        logger.info(
            "Executing exit tool: name=%s call_id=%s active_skills=%d depth=%d",
            called_exit_tool.name,
            called_exit_tool.call_id,
            self._skill_executions.depth,
            self._depth,
        )
        result = called_exit_tool.execute()
        content = self.get_exit_output(result)

        if not close_scope:
            return ToolExecutionResult(
                response=ToolMessage(
                    content, called_exit_tool.call_id, called_exit_tool.name
                ),
                raw_result=result,
                save_result=True,
            )

        if self._skill_executions.depth:
            content, output_id = self._format_output(
                self._skill_executions.current.skill.name, content, source="DeepSkill"
            )
            return self._close_deep_skill(content, result, output_id)

        self._depth = max(0, self._depth - 1)
        logger.warning(
            "Exit tool executed without an active deep skill: call_id=%s depth=%d",
            called_exit_tool.call_id,
            self._depth,
        )
        tool_response = ToolMessage(
            content=content,
            tool_call_id=called_exit_tool.call_id,
            name=called_exit_tool.name,
        )
        return ToolExecutionResult(
            response=tool_response, raw_result=result, save_result=True
        )

    def _open_deep_skill(self, called_deep_skill: CalledDeepSkill) -> None:
        history = self._client.get_history()
        start_index = max(
            index
            for index, message in enumerate(history)
            if message.get_role_name() == "assistant"
        )
        preceding_tool_responses = tuple(
            message
            for message in history[start_index + 1 :]
            if isinstance(message, ToolMessage)
        )
        called_deep_skill.open_execution(
            self._skill_executions,
            start_index,
            preceding_tool_responses,
        )
        self._depth += 1
        logger.info(
            "Opened deep skill: name=%s call_id=%s start_index=%d depth=%d",
            called_deep_skill.name,
            called_deep_skill.call_id,
            start_index,
            self._depth,
        )

    def _close_deep_skill(
        self, output: str, result: Any, output_id: str | None
    ) -> ToolExecutionResult:
        history = self._client.copy_history()
        start_index = self._skill_executions.get_current_start_index()
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
        self._depth = max(0, self._depth - 1)
        self._agentic_scope.register_skill_return(execution.opening_tool_call_id)
        logger.info(
            "Closed deep skill: name=%s opening_call_id=%s start_index=%d "
            "end_index=%d transcript_messages=%d depth=%d",
            execution.skill.name,
            execution.opening_tool_call_id,
            execution.start_index,
            execution.end_index,
            len(execution.history),
            self._depth,
        )
        return ToolExecutionResult(
            response=response,
            raw_result=result,
            save_result=True,
            replace_history_from=execution.start_index + 1,
            preserved_responses=execution.preceding_tool_responses,
            output_id=output_id,
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
