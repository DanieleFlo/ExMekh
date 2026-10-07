from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from time import monotonic
from typing import Any

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.deep_skill.deep_skill import CalledDeepSkill
from exmekh_lib.core.agents.deep_skill.execution import (
    DeepSkillExecution,
    DeepSkillExecutionTracker,
)
from exmekh_lib.core.agents.exit_tool.exit_tool import CalledExitTool, ExitTool
from exmekh_lib.core.agents.exit_tool.items import ExitItemError
from exmekh_lib.core.file_inspection.runtime import FileToolRuntime
from exmekh_lib.core.models.assistant_message import ToolCall
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.tool import CalledTool, PreformattedToolResult
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.tool_calls.execution_manager import (
    ExecutionSnapshot,
    ToolExecutionManager,
)
from exmekh_lib.core.tool_calls.output_formatter import ToolOutputFormatter
from exmekh_lib.utils.tool_errors import ToolArgumentError, tool_failure_result

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
        work_root: Path | None = None,
        execution_manager: ToolExecutionManager | None = None,
        scope_prefix: str = "",
    ) -> None:
        self._client = client
        self._output_formatter = ToolOutputFormatter(client)
        self._exit_tool = exit_tool
        self._depth = initial_depth
        self._agentic_scope = agentic_scope
        self._exit_content_resolver = exit_content_resolver
        self._work_root = work_root if work_root is not None else Path.cwd()
        self._skill_executions = DeepSkillExecutionTracker()
        self._execution_manager = execution_manager
        self._scope_prefix = scope_prefix
        self._file_tools = FileToolRuntime(execution_manager)

    def begin_tool_batch(self) -> None:
        """Freeze prerequisites from context available when this batch was generated."""
        self._file_tools.begin_batch(self._client.get_history())

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def pending_return_call_id(self) -> str | None:
        return self._agentic_scope.pending_return_call_id

    @property
    def skill_path(self) -> str:
        return self._skill_executions.active_path

    @property
    def skill_executions(self) -> tuple[DeepSkillExecution, ...]:
        return self._skill_executions.roots

    @property
    def skill_scope_ids(self) -> tuple[str, ...]:
        return (self._scope_prefix, *self._skill_executions.active_scope_ids)

    def is_directly_in_skill(self, name: str) -> bool:
        return self._skill_executions.is_current_skill(name)

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

    def validate_tool_calls(self, calls: list[ToolCall]) -> None:
        self._agentic_scope.validate_tool_calls(
            calls, self._skill_executions.active_skills, self._depth
        )

    def record_automatic_tool_call(
        self, call: ToolCall
    ) -> AbstractContextManager[None]:
        # Native skill returns consume their latest call batch and final reply.
        return self._client.record_tool_call(
            call, batch_call_id=self._agentic_scope.pending_return_call_id
        )

    def record_steering_output(self) -> str:
        """Expose saved guidance in every enclosing skill's output references."""
        output_id = self._output_formatter.generate_output_id()
        self._skill_executions.record_tool_output(
            RUNTIME.steering_output_name, output_id
        )
        return output_id

    def execute(
        self,
        tool_call: ResolvedToolCall,
        *,
        thinking: str | None = None,
        close_scope: bool = True,
        history_position: int = 0,
        thinking_annotation_provider: Callable[[], str | None] | None = None,
    ) -> ToolExecutionResult:
        supported_types = (CalledDeepSkill, CalledExitTool, CalledTool)
        if not isinstance(tool_call, supported_types):
            raise TypeError("Node supports only deep skills, exit tools, and tools")

        if isinstance(tool_call, CalledExitTool):
            return self._run_exit_tool(
                tool_call,
                thinking=thinking,
                close_scope=close_scope,
                thinking_annotation_provider=thinking_annotation_provider,
            )
        if isinstance(tool_call, CalledDeepSkill):
            return self._run_deep_skill(
                tool_call, thinking=thinking, history_position=history_position
            )
        result = self._run_tool(
            tool_call, thinking_annotation_provider=thinking_annotation_provider
        )
        if result.output_id is not None:
            self._skill_executions.record_tool_output(tool_call.name, result.output_id)
        return result

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

    def _run_tool(
        self,
        called_tool: CalledTool,
        *,
        thinking_annotation_provider: Callable[[], str | None] | None = None,
    ) -> ToolExecutionResult:
        logger.info(
            "Executing tool: name=%s call_id=%s depth=%d",
            called_tool.name,
            called_tool.call_id,
            self._depth,
        )
        snapshot = None
        elapsed_seconds = None
        inspection_path = None
        failed = False
        started_clock = monotonic()
        try:
            if called_tool.background_enabled and self._execution_manager is not None:
                result, snapshot = self._run_managed_tool(
                    called_tool, self._execution_manager
                )
            else:
                with self._file_tools.read_context(
                    self._client.get_history()
                ) as context:
                    try:
                        result = called_tool.execute(
                            work_root=self._work_root,
                            history=tuple(self._client.get_history()),
                        )
                    finally:
                        inspection_path = context.pending_inspection_path()
                elapsed_seconds = monotonic() - started_clock
        except Exception as error:
            result = tool_failure_result(error)
            failed = True
            elapsed_seconds = monotonic() - started_clock
        return self._format_tool_result(
            called_tool,
            result,
            snapshot=snapshot,
            thinking_annotation_provider=thinking_annotation_provider,
            elapsed_seconds=elapsed_seconds,
            inspection_path=inspection_path,
            failed=failed,
        )

    def _run_managed_tool(
        self, called_tool: CalledTool, manager: ToolExecutionManager
    ) -> tuple[Any, ExecutionSnapshot]:
        output_id = self._output_formatter.generate_output_id()
        operation = called_tool.create_execution(
            work_root=self._work_root,
            history=tuple(self._client.get_history()),
        )
        operation.add_run_context(
            self._file_tools.read_context(self._client.get_history())
        )
        try:
            snapshot = manager.start(
                called_tool.call_id,
                called_tool.name,
                called_tool.arguments,
                operation,
                self.skill_scope_ids,
                called_tool.execution_options,
                output_id=output_id,
                skill_path=self.skill_path,
            )
        except (TypeError, ValueError) as error:
            raise ToolArgumentError(f"Invalid arguments: {error}") from error
        result = snapshot.output_result()
        if snapshot.is_finished:
            return result, snapshot
        try:
            partial = manager.partial_output(output_id, 0)
            if partial is not None:
                result = {
                    **result,
                    "partial_output": partial[0],
                    "next_cursor": partial[1],
                }
        except Exception as error:
            result = {**result, "partial_output_error": str(error)}
        return result, snapshot

    def _format_tool_result(
        self,
        called_tool: CalledTool,
        result: Any,
        *,
        snapshot: ExecutionSnapshot | None,
        thinking_annotation_provider: Callable[[], str | None] | None,
        elapsed_seconds: float | None,
        inspection_path: str | None = None,
        failed: bool = False,
    ) -> ToolExecutionResult:
        raw_result = (
            result.content if isinstance(result, PreformattedToolResult) else result
        )
        content, output_id = self._output_formatter.format_tool_result(
            called_tool.name,
            result,
            annotation_provider=thinking_annotation_provider,
            execution=snapshot,
            elapsed_seconds=elapsed_seconds,
            failed=failed,
        )
        tool_response = ToolMessage(
            content=content,
            tool_call_id=called_tool.call_id,
            name=called_tool.name,
            attachments=result.attachments
            if isinstance(result, PreformattedToolResult)
            else None,
        )
        self._file_tools.record_result(called_tool.tool_call, result, tool_response)
        return ToolExecutionResult(
            response=tool_response,
            raw_result=raw_result,
            save_result=called_tool.cache_result,
            output_id=output_id,
            execution=snapshot,
            elapsed_seconds=elapsed_seconds,
            inspection_path=inspection_path,
            failed=failed,
        )

    def _running_tool_outputs(self) -> tuple[ExecutionSnapshot, ...]:
        if self._execution_manager is None:
            return ()
        return self._execution_manager.running_outputs(self.skill_scope_ids)

    def _run_exit_tool(
        self,
        called_exit_tool: CalledExitTool,
        *,
        thinking: str | None,
        close_scope: bool = True,
        thinking_annotation_provider: Callable[[], str | None] | None = None,
    ) -> ToolExecutionResult:
        logger.info(
            "Executing exit tool: name=%s call_id=%s active_skills=%d depth=%d",
            called_exit_tool.name,
            called_exit_tool.call_id,
            self._skill_executions.depth,
            self._depth,
        )
        try:
            result = called_exit_tool.execute(
                work_root=self._work_root,
                history=tuple(self._client.get_history()),
            )
        except (TypeError, ValueError) as error:
            tool_name = called_exit_tool.name
            diagnostic = (
                str(error)
                if isinstance(error, ToolArgumentError)
                else f"Invalid arguments: {error}"
            )
            return self._recoverable_error(
                called_exit_tool,
                f"{diagnostic} "
                f'Use {tool_name}(text="<result>", items=[]) or '
                f'{tool_name}(text="… {{{{<id-output>}}}} …", '
                'items=["<id-output>"].',
            )
        content = result["text"]
        items = result["items"]
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
            return self._recoverable_error(called_exit_tool, str(error))
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
            elapsed_seconds = self._skill_executions.current.elapsed_seconds
            content, output_id = self._output_formatter.format_skill_return(
                self._skill_executions.current.name,
                content,
                thinking=thinking,
                annotation_provider=thinking_annotation_provider,
                background_tools=self._running_tool_outputs(),
                used_tools=self._skill_executions.current.used_tools,
                elapsed_seconds=elapsed_seconds,
            )
            return self._close_deep_skill(content, result, output_id, elapsed_seconds)

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

    def _recoverable_error(
        self, called_tool: CalledTool | CalledExitTool, message: str
    ) -> ToolExecutionResult:
        result = {
            "ok": False,
            "error_kind": "invalid_tool_arguments",
            "error": (
                f"Error {called_tool.name}: {message} "
                f"Correct the error and call {called_tool.name} again."
            ),
        }
        content, output_id = self._output_formatter.format_tool_result(
            called_tool.name, result, failed=True
        )
        return ToolExecutionResult(
            response=ToolMessage(
                content=content,
                tool_call_id=called_tool.call_id,
                name=called_tool.name,
            ),
            raw_result=result,
            save_result=True,
            output_id=output_id,
            failed=True,
        )

    def _open_deep_skill(
        self,
        called_deep_skill: CalledDeepSkill,
        *,
        thinking: str | None,
        history_position: int,
    ) -> None:
        batch = self._client.get_latest_tool_batch()
        called_deep_skill.open_execution(
            self._skill_executions,
            batch.opening_index,
            thinking,
            batch.preceding_responses,
            history_position,
        )
        self._depth += 1
        logger.info(
            "Opened deep skill: name=%s call_id=%s start_index=%d depth=%d",
            called_deep_skill.name,
            called_deep_skill.call_id,
            batch.opening_index,
            self._depth,
        )

    def _close_deep_skill(
        self, output: str, result: Any, output_id: str | None, elapsed_seconds: float
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
            name=execution.name,
        )
        self._depth = max(0, self._depth - 1)
        self._agentic_scope.register_skill_return(execution.opening_tool_call_id)
        logger.info(
            "Closed deep skill: name=%s opening_call_id=%s start_index=%d "
            "end_index=%d transcript_messages=%d depth=%d",
            execution.name,
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
            used_tools=execution.used_tools,
            elapsed_seconds=elapsed_seconds,
            cache_identity=CacheIdentity(
                name=execution.name,
                source="DeepSkill",
                tool_call_id=execution.opening_tool_call_id,
                args=execution.opening_arguments,
                thinking=execution.opening_thinking,
                history_position=execution.opening_history_position,
            ),
        )
