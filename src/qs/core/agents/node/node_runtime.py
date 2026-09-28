from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import partial
from typing import Any, Final, Literal, cast
from uuid import uuid4

from qs.config.runtime import RUNTIME
from qs.core.agents.deep_agent.agent_response import AgentResponse
from qs.core.agents.deep_agent.stream_event import (
    DeepAgentExecutionEvent,
    DeepAgentExecutionEventType,
    DeepAgentStreamEvent,
)
from qs.core.agents.deep_skill.deep_skill import CalledDeepSkill
from qs.core.agents.exit_tool.exit_tool import CalledExitTool
from qs.core.agents.exit_tool.items import ExitItemResolver
from qs.core.cache.node_cache import NodeCache
from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.models.assistant_message import FunctionCall, ToolCall
from qs.core.models.client import Client
from qs.core.models.client_request import ClientRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.provider_stream_event import ProviderStreamEvent
from qs.core.models.tool import PreformattedToolResult
from qs.core.models.user_message import ErrorUserMessage, UserMessage
from qs.core.tool_calls.execution_manager import ToolExecutionManager
from qs.core.tool_calls.execution_tools import ExecutionControlTool
from qs.core.tool_calls.memory_tools import ExecutionHistory, HarnessMemoryTools
from qs.interfaces.event_handler import ProviderEventHandler
from qs.interfaces.message import Message
from qs.utils.json import to_json_value

from .node_runtime_config import NodeRuntimeConfig
from .thinking_annotation import ThinkingAnnotationResolver
from .tool_call_executor import ToolCallExecutor
from .tool_call_resolver import ResolvedToolCall, ToolCallResolver
from .types import NodeUserMessage

logger = logging.getLogger(__name__)


class NodeRuntime(ProviderEventHandler):
    """Own the conversational lifecycle of an initialized node."""

    REQUIRED_TOOL_CHOICE: Final[str] = RUNTIME.required_tool_choice
    EMPTY_RESPONSE_MESSAGE: Final[str] = RUNTIME.empty_response_message

    def __init__(self, config: NodeRuntimeConfig) -> None:
        self._config = config
        self._node_id = f"{config.get_hds_name()}-{uuid4().hex}"
        self._tool_cache: NodeCache | None = None
        self._execution_history = ExecutionHistory()
        self._execution_manager = ToolExecutionManager()
        self._memory_tools = HarnessMemoryTools(
            self._execution_history,
            self._get_tool_cache,
            self._execution_manager,
        )
        self._exit_items = ExitItemResolver(
            self._execution_history,
            self._get_tool_cache,
        )
        self._run_start_position = 0
        self._thinking_annotation_resolver = ThinkingAnnotationResolver(config.provider)
        defaults = config.default_capabilities
        self._execution_control = ExecutionControlTool(self._execution_manager)
        harness_tools = (*self._memory_tools.tools, self._execution_control)
        if defaults is not None:
            harness_tools = tuple(
                (
                    defaults.tool_override(tool.name) or tool
                ).without_background_execution()
                for tool in harness_tools
                if defaults.includes(tool.name)
            )
        self._runtime_tools = (*config.runtime_tools, *harness_tools)
        self._validate_runtime_tool_names()
        self._resolver = ToolCallResolver(
            config.capabilities,
            self._runtime_tools,
        )
        self._client: Client
        self._executor: ToolCallExecutor
        self._reset_session()

    @property
    def depth(self) -> int:
        return self._executor.depth

    @property
    def agent_context_id(self) -> str:
        return self._config.resolved_agent_context_id

    def control_tool_execution(
        self,
        tool_call_id: str,
        action: Literal["wait", "cancel"] = "wait",
        wait_seconds: float = RUNTIME.default_tool_wait_seconds,
    ) -> dict[str, Any]:
        return self._execution_control.control_tool_execution(
            tool_call_id, action, wait_seconds
        )

    def read_tool_output(
        self, output_id: str, cursor: int | None = None
    ) -> PreformattedToolResult:
        return self._memory_tools.read_tool_output(output_id, cursor)

    def _validate_runtime_tool_names(self) -> None:
        capability_names = {
            tool.name for tool in self._config.capabilities.get_all_tools()
        }
        runtime_names = tuple(tool.name for tool in self._runtime_tools)
        duplicate_runtime_names = {
            name for name in runtime_names if runtime_names.count(name) > 1
        }
        if duplicate_runtime_names:
            names = ", ".join(sorted(duplicate_runtime_names))
            raise ValueError(f"Harness runtime tool names must be unique: {names}")

        collisions = capability_names.intersection(runtime_names)
        if collisions:
            names = ", ".join(sorted(collisions))
            raise ValueError(f"Harness runtime tool names are reserved: {names}")

    def run(self, user_message: NodeUserMessage) -> AgentResponse:
        if not self._config.is_persistent_history():
            self._reset_session()

        self._thinking_annotation_resolver.begin_turn()
        self._run_start_position = self._execution_history.count
        logger.info(
            "Starting node run: node=%s depth=%d",
            self._config.get_hds_name(),
            self.depth,
        )
        self._sync_completed_tool_outputs()
        pending_result = self._execution_manager.next_delivery(
            self._executor.skill_scope_ids
        )
        if pending_result is not None and self._has_read_tool():
            self._client.add_user_message(UserMessage(content=user_message))
            self._deliver_completed_tool_outputs()
            response = self._client.continue_after_tools(
                extra=self._executor.build_agentic_extra(),
                tool_choice=self._current_tool_choice(),
                streaming=self._config.is_streaming(),
                on_event=self,
            )
        else:
            response = self._client.call(
                ClientRequest(
                    message=UserMessage(content=user_message),
                    extra=self._executor.build_agentic_extra(),
                    tool_choice=self._current_tool_choice(),
                    streaming=self._config.is_streaming(),
                    on_event=self,
                )
            )
        response = self._route_until_complete(response)
        logger.info(
            "Completed node run: node=%s depth=%d root_executions=%d",
            self._config.get_hds_name(),
            self.depth,
            len(self._executor.skill_executions),
        )
        return AgentResponse(
            response=response,
            usage=response.usage,
            thinking=response.thinking,
            history=self._client.dump_history(),
            skill_executions=self._executor.skill_executions,
        )

    def restore_history(
        self,
        messages: Sequence[Message[Any]],
        restore_execution_state: Callable[[ExecutionHistory, NodeCache], None]
        | None = None,
    ) -> None:
        """Restore messages and tool state for a persistent node."""
        if not self._config.is_persistent_history():
            raise ValueError("Cannot restore history for a non-persistent node")
        self._client.replace_history_from(1, list(messages))
        if restore_execution_state is not None:
            restore_execution_state(
                self._get_execution_history(),
                self._get_tool_cache(),
            )

    def _route_until_complete(self, response: ProviderResponse) -> ProviderResponse:
        while True:
            tool_calls = response.get_tool_calls()
            if tool_calls:
                logger.info(f"Tool calls: {[tool.get_name() for tool in tool_calls]}")
                resolved_tool_calls = self._prepare_tool_batch(tool_calls)
                thinking = response.thinking.thoughts if response.thinking else None
                self._execute_batch(resolved_tool_calls, thinking)
                while (
                    self._pending_batches and self._pending_batches[-1][0] == self.depth
                ):
                    _, pending, pending_thinking = self._pending_batches.pop()
                    self._client.add_tool_calls([call.tool_call for call in pending])
                    self._execute_batch(pending, pending_thinking)

                self._sync_completed_tool_outputs()
                self._deliver_completed_tool_outputs()

                tool_choice = self._current_tool_choice()
                response = self._client.continue_after_tools(
                    extra=self._executor.build_agentic_extra(),
                    tool_choice=tool_choice,
                    streaming=self._config.is_streaming(),
                    on_event=self,
                )
                self._executor.acknowledge_agentic_extra()
                continue

            content = response.get_message()
            if isinstance(content, str) and content.strip():
                self._sync_completed_tool_outputs()
                if self._deliver_completed_tool_outputs():
                    response = self._client.continue_after_tools(
                        extra=self._executor.build_agentic_extra(),
                        tool_choice=self._current_tool_choice(),
                        streaming=self._config.is_streaming(),
                        on_event=self,
                    )
                    continue
                return response

            self._sync_completed_tool_outputs()
            self._deliver_completed_tool_outputs()
            response = self._client.call(
                ClientRequest(
                    message=ErrorUserMessage(self.EMPTY_RESPONSE_MESSAGE),
                    extra=self._executor.build_agentic_extra(),
                    tool_choice=self._current_tool_choice(),
                    streaming=self._config.is_streaming(),
                    on_event=self,
                )
            )

    def _prepare_tool_batch(self, tool_calls: list[ToolCall]) -> list[ResolvedToolCall]:
        """Validate against the generating scope before any execution changes it."""
        extra = self._executor.build_agentic_extra()
        scope = extra.get("agentic")
        if scope:
            context_id = scope.get("agent_context_id")
            if not isinstance(context_id, str) or not context_id.strip():
                raise ValueError(
                    "Agentic scope configuration ('agent_context_id') must be a non-empty string"
                )
            allowed = set(cast(list[str], scope.get("allowed_tools"))) | set(
                cast(list[str], scope["allowed_skills"])
            )
            for call in tool_calls:
                if call.get_name() not in allowed:
                    raise ValueError(
                        f"Capability '{call.get_name()}' is not allowed at "
                        f"{self._executor.skill_path}; allowed_tools={scope['allowed_tools']}, "
                        f"allowed_skills={scope.get('allowed_skills')}"
                    )
            resolved = self._resolver.resolve_all(tool_calls)
            return resolved
        raise ValueError("Agentic scope configuration ('agentic') is missing")

    def _execute_batch(
        self, calls: list[ResolvedToolCall], thinking: str | None
    ) -> None:
        tools = [
            call
            for call in calls
            if not isinstance(call, (CalledDeepSkill, CalledExitTool))
        ]
        skills = [call for call in calls if isinstance(call, CalledDeepSkill)]
        exits = [call for call in calls if isinstance(call, CalledExitTool)]
        active_calls = tools + skills[:1] if skills else tools + exits
        deferred_calls = skills[1:] + exits if skills else []

        annotation_provider = (
            partial(self._thinking_annotation_resolver.resolve, thinking)
            if self._executor.is_directly_in_skill(RUNTIME.deep_thinking_skill_name)
            else None
        )

        if deferred_calls:
            # Deferred siblings must not appear as unanswered calls in the child scope.
            self._client.replace_latest_tool_calls(
                [call.tool_call for call in active_calls]
            )
            self._pending_batches.append((self.depth, deferred_calls, thinking))

        for index, call in enumerate(active_calls):
            self._execute_tool_call(
                call,
                thinking=thinking,
                close_scope=index == len(active_calls) - 1,
                thinking_annotation_provider=annotation_provider,
            )

    def _execute_tool_call(
        self,
        tool_call: ResolvedToolCall,
        *,
        thinking: str | None = None,
        close_scope: bool = True,
        thinking_annotation_provider: Callable[[], str | None] | None = None,
    ) -> None:
        self._sync_completed_tool_outputs()
        skill_path = self._executor.skill_path
        arguments = to_json_value(tool_call.arguments)
        history_entry = self._get_execution_history().record(
            tool_call.name,
            tool_call.call_id,
        )
        self._emit_tool_state(tool_call.name, tool_call.call_id, running=True)
        try:
            logger.info(
                "Starting tool execution: name=%s call_id=%s path=%s",
                tool_call.name,
                tool_call.call_id,
                self._executor.skill_path,
            )
            result = self._executor.execute(
                tool_call,
                thinking=thinking,
                close_scope=close_scope,
                history_position=history_entry.position,
                thinking_annotation_provider=thinking_annotation_provider,
            )
            if result.replace_history_from is None:
                self._client.add_tool_response(result.response)
            else:
                self._client.replace_history_from(
                    result.replace_history_from,
                    messages=[*result.preserved_responses, result.response],
                )
            if result.save_result:
                identity = result.cache_identity
                record = ToolCallRecord(
                    name=identity.name if identity is not None else tool_call.name,
                    id=result.output_id or uuid4().hex,
                    tool_call_id=(
                        identity.tool_call_id
                        if identity is not None
                        else tool_call.call_id
                    ),
                    args=identity.args if identity is not None else arguments,
                    result=result.raw_result,
                    skill_path=skill_path,
                    thinking=(identity.thinking if identity is not None else thinking),
                    source=identity.source if identity is not None else "Tool",
                    execution=(
                        result.execution.status_info()
                        if result.execution is not None
                        else None
                    ),
                )
                record_cache = self._save_tool_record(record)
                if identity is None:
                    history_entry.attach_output(record.id, record_cache.session_dir)
                else:
                    self._get_execution_history().attach_output_at_position(
                        identity.history_position,
                        record.id,
                        record_cache.session_dir,
                    )
            else:
                record = None
            if result.execution is not None:
                if result.output_id is None:
                    raise RuntimeError(
                        "Managed tool execution is missing its output ID"
                    )
                self._execution_manager.attach_output(
                    result.execution.call_id,
                    result.output_id,
                    record,
                )
        finally:
            self._emit_tool_state(tool_call.name, tool_call.call_id, running=False)

    def _save_tool_record(self, record: ToolCallRecord) -> NodeCache:
        cache = self._get_tool_cache()
        cache.store(record)
        return cache

    def _sync_completed_tool_outputs(self) -> None:
        for snapshot, record in self._execution_manager.completed_to_persist():
            result = snapshot.output_result()
            if isinstance(result, PreformattedToolResult):
                result = result.content
            self._save_tool_record(
                replace(record, result=result, execution=snapshot.status_info())
            )
            self._execution_manager.mark_persisted(record.id)

    def _deliver_completed_tool_outputs(self) -> bool:
        if not self._has_read_tool():
            return False
        delivered = False
        while snapshot := self._execution_manager.next_delivery(
            self._executor.skill_scope_ids
        ):
            output_id = snapshot.output_id
            if output_id is None:
                break
            call = ToolCall(
                id=uuid4().hex,
                function=FunctionCall(
                    name=RUNTIME.read_tool_name,
                    arguments={"output_id": output_id},
                ),
            )
            self._execute_automatic_output_read(call)
            self._execution_manager.mark_delivered(output_id)
            delivered = True
        return delivered

    def _execute_automatic_output_read(self, call: ToolCall) -> None:
        with self._executor.record_automatic_tool_call(call):
            self._execute_tool_call(self._resolver.resolve(call))

    def _has_read_tool(self) -> bool:
        return any(tool.name == RUNTIME.read_tool_name for tool in self._runtime_tools)

    def _get_tool_cache(self) -> NodeCache:
        if self._tool_cache is None or not self._tool_cache.is_active:
            self._tool_cache = self._config.build_node_cache(self._node_id)
        return self._tool_cache

    def _get_execution_history(self) -> ExecutionHistory:
        history = getattr(self, "_execution_history", None)
        if history is None:
            history = ExecutionHistory()
            self._execution_history = history
        return history

    def _reset_session(self) -> None:
        self._scope_prefix = uuid4().hex
        self._pending_batches: list[tuple[int, list[ResolvedToolCall], str | None]] = []
        tools = self._config.build_tools(self._runtime_tools)
        self._client = Client(
            provider=self._config.provider,
            system_message=self._config.build_system_message(),
            tools=tools,
        )

        self._executor = self._config.build_executor(
            self._client,
            tuple(tool.name for tool in self._runtime_tools),
            exit_content_resolver=self._compose_exit_content,
            execution_manager=self._execution_manager,
            scope_prefix=self._scope_prefix,
        )

    def _compose_exit_content(
        self,
        content: str,
        items: list[str],
        skill_path: str,
        opening_position: int | None,
    ) -> str:
        return self._exit_items.compose(
            content,
            items,
            skill_path,
            self._run_start_position if opening_position is None else opening_position,
        )

    def _current_tool_choice(self) -> str | None:
        if self.depth > 0:
            return self.REQUIRED_TOOL_CHOICE
        return None

    def handle(self, event: ProviderStreamEvent) -> None:
        self._config.emit_stream(
            DeepAgentStreamEvent(
                provider_event=event,
                depth=self.depth,
                skill_path=self._executor.skill_path,
            )
        )

    def _emit_tool_state(self, name: str, call_id: str, running: bool) -> None:
        if not self._config.is_streaming():
            return

        snapshot = self._executor.snapshot()
        self._config.emit_stream(
            DeepAgentExecutionEvent(
                type=(
                    DeepAgentExecutionEventType.TOOL_START
                    if running
                    else DeepAgentExecutionEventType.TOOL_END
                ),
                depth=snapshot.depth,
                skill_path=snapshot.skill_path,
                history=snapshot.history,
                history_tree=snapshot.history_tree,
                tool_name=name,
                tool_call_id=call_id,
            )
        )
