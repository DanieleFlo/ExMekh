from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from qs.core.agents.deep_skill.deep_skill import CalledDeepSkill
from qs.core.agents.deep_skill.execution import (
    DeepSkillExecution,
    DeepSkillExecutionTracker,
)
from qs.core.agents.exit_skill.exit_skill import CalledExitSkill, ExitSkill
from qs.core.agents.exit_tool.exit_tool import CalledExitTool
from qs.core.models.client import Client
from qs.core.models.tool import CalledTool
from qs.core.models.tool_message import ToolMessage
from qs.core.models.xml_header import XmlHeader

from .agentic_scope import AgenticScope
from .tool_call_resolver import ResolvedToolCall
from .types import NodeExecutionSnapshot

logger = logging.getLogger(__name__)


class ToolCallExecutor:
    """Execute resolved tool calls and track nested deep-skill state."""

    def __init__(
        self,
        client: Client,
        exit_skill: ExitSkill,
        initial_depth: int,
        agentic_scope: AgenticScope,
        on_state: Callable[[str, str, bool], None] | None = None,
    ) -> None:
        self._client = client
        self._exit_skill = exit_skill
        self._depth = initial_depth
        self._agentic_scope = agentic_scope
        self._skill_executions = DeepSkillExecutionTracker()
        self._on_state = on_state

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
        history = self._client.copy_history()
        return NodeExecutionSnapshot(
            depth=self._depth,
            skill_path=self._skill_executions.active_path,
            history=[message.message_dump() for message in history],
            history_tree=self._skill_executions.history_tree(history),
        )

    def build_agentic_extra(self) -> dict[str, dict[str, object]]:
        return self._agentic_scope.build_extra(
            self._skill_executions.active_skills,
            self._depth,
        )

    def acknowledge_agentic_extra(self) -> None:
        self._agentic_scope.acknowledge_extra()

    def execute_all(self, tool_calls: list[ResolvedToolCall]) -> None:
        logger.info(
            "Executing assistant tool calls: count=%d depth=%d",
            len(tool_calls),
            self._depth,
        )
        for tool_call in tool_calls:
            self.execute(tool_call)

    def execute(self, tool_call: ResolvedToolCall) -> None:
        supported_types = (CalledExitSkill, CalledDeepSkill, CalledExitTool, CalledTool)
        if not isinstance(tool_call, supported_types):
            raise TypeError(
                "Node supports only deep skills, exit skills, exit tools, and tools"
            )

        self._emit_state(tool_call, running=True)
        try:
            self._execute(tool_call)
        finally:
            self._emit_state(tool_call, running=False)

    def _execute(self, tool_call: ResolvedToolCall) -> None:
        if isinstance(tool_call, CalledExitSkill):
            self._run_exit_skill(tool_call)
            return
        if isinstance(tool_call, CalledDeepSkill):
            self._run_deep_skill(tool_call)
            return
        if isinstance(tool_call, CalledExitTool):
            self._run_exit_tool(tool_call)
            return
        self._run_tool(tool_call)

    def _emit_state(self, tool_call: ResolvedToolCall, *, running: bool) -> None:
        if self._on_state is not None:
            self._on_state(tool_call.name, tool_call.call_id, running)

    def _run_deep_skill(self, called_deep_skill: CalledDeepSkill) -> None:
        self._open_deep_skill(called_deep_skill)
        content = called_deep_skill.render_instructions(self._exit_skill)
        self._client.add_tool_response(
            ToolMessage(
                content=content,
                tool_call_id=called_deep_skill.call_id,
                name=called_deep_skill.name,
            )
        )

    def _run_tool(self, called_tool: CalledTool) -> None:
        logger.info(
            "Executing tool: name=%s call_id=%s depth=%d",
            called_tool.name,
            called_tool.call_id,
            self._depth,
        )
        content = self.stringify_result(called_tool.execute())
        try:
            xml_header = XmlHeader(
                source="Tool", 
                name=called_tool.name,
                id=self._client.generate_id()
            )
            content = xml_header.prepend_to(content)
        except Exception as e:
            logging.error(e)
        self._client.add_tool_response(
            ToolMessage(
                content=content,
                tool_call_id=called_tool.call_id,
                name=called_tool.name,
            )
        )

    def _run_exit_skill(self, called_exit_skill: CalledExitSkill) -> None:
        logger.info(
            "Rendering exit skill: name=%s call_id=%s depth=%d",
            called_exit_skill.name,
            called_exit_skill.call_id,
            self._depth,
        )
        content = called_exit_skill.render_instructions()
        self._client.add_tool_response(
            ToolMessage(
                content=content,
                tool_call_id=called_exit_skill.call_id,
                name=called_exit_skill.name,
            )
        )

    def _run_exit_tool(self, called_exit_tool: CalledExitTool) -> None:
        logger.info(
            "Executing exit tool: name=%s call_id=%s active_skills=%d depth=%d",
            called_exit_tool.name,
            called_exit_tool.call_id,
            self._skill_executions.depth,
            self._depth,
        )
        result = called_exit_tool.execute()
        content = self.get_exit_output(result)

        if self._skill_executions.depth:
            try:
                xml_header = XmlHeader(
                    source="DeepSkill", 
                    name=called_exit_tool.name,
                    id=self._client.generate_id()
                )
                content = xml_header.prepend_to(content)
            except Exception as e:
                logging.error(e)
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
                name=called_exit_tool.name,
            )
        )

    def _open_deep_skill(self, called_deep_skill: CalledDeepSkill) -> None:
        start_index = len(self._client.get_history()) - 1
        called_deep_skill.open_execution(self._skill_executions, start_index)
        self._depth += 1
        logger.info(
            "Opened deep skill: name=%s call_id=%s start_index=%d depth=%d",
            called_deep_skill.name,
            called_deep_skill.call_id,
            start_index,
            self._depth,
        )

    def _close_deep_skill(self, output: str) -> None:
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
        self._client.replace_history_from(execution.start_index + 1, [response])
        self._depth = max(0, self._depth - 1)
        self._agentic_scope.register_skill_return(execution.opening_tool_call_id)
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
