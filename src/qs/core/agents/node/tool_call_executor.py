from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal, cast

from qs.core.agents.deep_skill.deep_skill import CalledDeepSkill
from qs.core.agents.deep_skill.execution import (
    DeepSkillExecution,
    DeepSkillExecutionTracker,
)
from qs.core.agents.exit_tool.exit_tool import CalledExitTool, ExitTool
from qs.core.agents.exit_tool.items import ExitItemError
from qs.core.models.client import Client
from qs.core.models.tool import CalledTool, PreformattedToolResult
from qs.core.models.tool_message import ToolMessage
from qs.core.models.xml_header import XmlHeader
from qs.utils.json import stringify_tool_result

from .agentic_scope import AgenticScope
from .tool_call_resolver import ResolvedToolCall
from .types import CacheIdentity, NodeExecutionSnapshot, ToolExecutionResult

logger = logging.getLogger(__name__)

type ExitContentResolver = Callable[[str, list[str], str, int | None], str]


class ToolCallExecutor:
    """Execute resolved tool calls and track nested deep-skill state."""

    def __init__(
        self,
        client: Client,
        exit_tool: ExitTool,
        initial_depth: int,
        agentic_scope: AgenticScope,
        exit_content_resolver: ExitContentResolver,
    ) -> None:
        self._client = client
        self._exit_tool = exit_tool
        self._depth = initial_depth
        self._agentic_scope = agentic_scope
        self._exit_content_resolver = exit_content_resolver
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
        self,
        tool_call: ResolvedToolCall,
        *,
        thinking: str | None = None,
        close_scope: bool = True,
        history_position: int = 0,
    ) -> ToolExecutionResult:
        supported_types = (CalledDeepSkill, CalledExitTool, CalledTool)
        if not isinstance(tool_call, supported_types):
            raise TypeError("Node supports only deep skills, exit tools, and tools")

        if isinstance(tool_call, CalledExitTool):
            return self._run_exit_tool(
                tool_call, thinking=thinking, close_scope=close_scope
            )
        if isinstance(tool_call, CalledDeepSkill):
            return self._run_deep_skill(
                tool_call, thinking=thinking, history_position=history_position
            )
        return self._run_tool(tool_call)

    def _run_deep_skill(
        self,
        called_deep_skill: CalledDeepSkill,
        *,
        thinking: str | None,
        history_position: int,
    ) -> ToolExecutionResult:
        self._open_deep_skill(
            called_deep_skill, thinking=thinking, history_position=history_position
        )
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
        if isinstance(result, PreformattedToolResult):
            content = result.content
            raw_result = result.content
            output_id = None
        else:
            raw_result = result
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
            raw_result=raw_result,
            save_result=called_tool.cache_result,
            output_id=output_id,
        )

    def _format_output(
        self,
        tool_name: str,
        content: str,
        *,
        source: Literal["Tool", "DeepSkill"],
        thinking: str | None = None,
    ) -> tuple[str, str | None]:
        excerpt = None
        if source == "DeepSkill" and thinking and thinking.strip():
            tokenizer = self._client.get_tokenizer()
            excerpt = tokenizer.truncate_text(thinking, 100)
        try:
            header = XmlHeader(
                source=source, name=tool_name, id=self._client.generate_id()
            )
            if excerpt and excerpt.strip():
                header.add_field("thinking", excerpt)
            return header.prepend_to(content), header.id
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            logger.error(error)
            return content, None

    def _run_exit_tool(
        self,
        called_exit_tool: CalledExitTool,
        *,
        thinking: str | None,
        close_scope: bool = True,
    ) -> ToolExecutionResult:
        logger.info(
            "Executing exit tool: name=%s call_id=%s active_skills=%d depth=%d",
            called_exit_tool.name,
            called_exit_tool.call_id,
            self._skill_executions.depth,
            self._depth,
        )
        try:
            called_exit_tool.validate_arguments()
        except (TypeError, ValueError) as error:
            tool_name = called_exit_tool.name
            return self._exit_error(
                called_exit_tool,
                f"Argomenti non validi: {error}. "
                f'Usa {tool_name}(text="<risultato>", items=[]) oppure '
                f'{tool_name}(text="… {{{{<id-output>}}}} …", '
                'items=["<id-output>"].',
            )

        result = called_exit_tool.execute()
        content = self.get_exit_output(result)
        items = cast(list[str], result["items"])
        opening_position = (
            self._skill_executions.current.opening_history_position
            if self._skill_executions.depth
            else None
        )
        try:
            content = self._exit_content_resolver(
                content, items, self.skill_path, opening_position
            )
        except ExitItemError as error:
            return self._exit_error(called_exit_tool, str(error))
        result = {**result, "text": content}

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
                self._skill_executions.current.skill.name,
                content,
                source="DeepSkill",
                thinking=thinking,
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

    @staticmethod
    def _exit_error(
        called_exit_tool: CalledExitTool, message: str
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            response=ToolMessage(
                content=(
                    f"Errore {called_exit_tool.name}: {message} "
                    f"Correggi e richiama {called_exit_tool.name}."
                ),
                tool_call_id=called_exit_tool.call_id,
                name=called_exit_tool.name,
            )
        )

    def _open_deep_skill(
        self,
        called_deep_skill: CalledDeepSkill,
        *,
        thinking: str | None,
        history_position: int,
    ) -> None:
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
            thinking,
            preceding_tool_responses,
            history_position,
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
            cache_identity=CacheIdentity(
                name=execution.skill.name,
                source="DeepSkill",
                tool_call_id=execution.opening_tool_call_id,
                args=execution.opening_arguments,
                thinking=execution.opening_thinking,
                history_position=execution.opening_history_position,
            ),
        )

    @staticmethod
    def get_exit_output(result: Any) -> str:
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise TypeError("exit tool result must contain a text string")
        return result["text"]

    @staticmethod
    def stringify_result(result: Any) -> str:
        return stringify_tool_result(result)
