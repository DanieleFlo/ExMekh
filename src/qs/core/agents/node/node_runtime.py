from __future__ import annotations

import logging
from typing import Final, cast
from uuid import uuid4

from qs.core.agents.deep_agent.agent_response import AgentResponse
from qs.core.agents.deep_agent.stream_event import (
    DeepAgentExecutionEvent,
    DeepAgentExecutionEventType,
    DeepAgentStreamEvent,
)
from qs.core.agents.deep_skill.deep_skill import CalledDeepSkill
from qs.core.agents.exit_tool.exit_tool import CalledExitTool
from qs.core.cache.node_cache import NodeCache
from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.models.assistant_message import ToolCall
from qs.core.models.client import Client
from qs.core.models.client_request import ClientRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.provider_stream_event import ProviderStreamEvent
from qs.core.models.user_message import ErrorUserMessage, UserMessage
from qs.utils.json import to_json_value

from .node_runtime_config import NodeRuntimeConfig
from .tool_call_executor import ToolCallExecutor
from .tool_call_resolver import ResolvedToolCall, ToolCallResolver
from .types import NodeUserMessage

logger = logging.getLogger(__name__)


class NodeRuntime:
    """Own the conversational lifecycle of an initialized node."""

    REQUIRED_TOOL_CHOICE: Final[str] = "required"
    EMPTY_RESPONSE_MESSAGE: Final[str] = (
        "A response without a tool call and without content is not allowed. "
        "Respond with either a tool call or non-empty content."
    )

    def __init__(self, config: NodeRuntimeConfig) -> None:
        self._config = config
        self._node_id = f"{config.get_hds_name()}-{uuid4().hex}"
        self._tool_cache: NodeCache | None = None
        self._resolver = ToolCallResolver(config.capabilities)
        self._client: Client
        self._executor: ToolCallExecutor
        self._reset_session()

    @property
    def depth(self) -> int:
        return self._executor.depth

    def run(self, user_message: NodeUserMessage) -> AgentResponse:
        if not self._config.is_persistent_history():
            self._reset_session()

        logger.info(
            "Starting node run: node=%s depth=%d",
            self._config.get_hds_name(),
            self.depth,
        )
        response = self._client.call(
            ClientRequest(
                message=UserMessage(content=user_message),
                extra=self._executor.build_agentic_extra(),
                tool_choice=self._current_tool_choice(),
                streaming=self._config.is_streaming(),
                on_event=self._emit_stream_event,
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
                    self._execute_batch(pending, pending_thinking)

                tool_choice = self._current_tool_choice()
                response = self._client.continue_after_tools(
                    extra=self._executor.build_agentic_extra(),
                    tool_choice=tool_choice,
                    streaming=self._config.is_streaming(),
                    on_event=self._emit_stream_event,
                )
                self._executor.acknowledge_agentic_extra()
                continue

            content = response.get_message()
            if isinstance(content, str) and content.strip():
                return response

            response = self._client.call(
                ClientRequest(
                    message=ErrorUserMessage(self.EMPTY_RESPONSE_MESSAGE),
                    extra=self._executor.build_agentic_extra(),
                    tool_choice=self._current_tool_choice(),
                    streaming=self._config.is_streaming(),
                    on_event=self._emit_stream_event,
                )
            )

    def _prepare_tool_batch(self, tool_calls: list[ToolCall]) -> list[ResolvedToolCall]:
        """Validate against the generating scope before any execution changes it."""
        extra = self._executor.build_agentic_extra()
        scope = extra.get("agentic")
        if scope:
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
        raise ValueError(
            f"Agentic scope configuration ('agentic') is missing"
        )

    def _execute_batch(
        self, calls: list[ResolvedToolCall], thinking: str | None
    ) -> None:
        origin_depth = self.depth
        for index, call in enumerate(calls):
            self._execute_tool_call(
                call,
                thinking=thinking,
                close_scope=not isinstance(call, CalledExitTool)
                or index == len(calls) - 1,
            )
            if isinstance(call, CalledDeepSkill):
                if index + 1 < len(calls):
                    self._pending_batches.append(
                        (origin_depth, calls[index + 1 :], thinking)
                    )
                return

    def _execute_tool_call(
        self,
        tool_call: ResolvedToolCall,
        *,
        thinking: str | None = None,
        close_scope: bool = True,
    ) -> None:
        skill_path = self._executor.skill_path
        arguments = to_json_value(tool_call.arguments)
        self._emit_tool_state(tool_call.name, tool_call.call_id, running=True)
        try:
            logger.info(
    "Starting tool execution: name=%s call_id=%s path=%s",
    tool_call.name,
    tool_call.call_id,
    self._executor.skill_path,
)
            result = self._executor.execute(tool_call, close_scope=close_scope)
            if result.replace_history_from is None:
                self._client.add_tool_response(result.response)
            else:
                self._client.replace_history_from(
                    result.replace_history_from,
                    messages=[*result.preserved_responses, result.response],
                )
            if result.save_result:
                record = ToolCallRecord(
                    name=tool_call.name,
                    id=result.output_id or uuid4().hex,
                    tool_call_id=tool_call.call_id,
                    args=arguments,
                    result=result.raw_result,
                    skill_path=skill_path,
                    thinking=thinking,
                )
                self._save_tool_record(record)
        finally:
            self._emit_tool_state(tool_call.name, tool_call.call_id, running=False)

    def _save_tool_record(self, record: ToolCallRecord) -> None:
        if self._tool_cache is None or not self._tool_cache.is_active:
            self._tool_cache = self._config.build_node_cache(self._node_id)
        self._tool_cache.store(record)

    def _reset_session(self) -> None:
        self._pending_batches: list[tuple[int, list[ResolvedToolCall], str | None]] = []
        tools = self._config.build_tools()
        self._client = Client(
            provider=self._config.provider,
            system_message=self._config.build_system_message(),
            tools=tools,
        )

        self._executor = self._config.build_executor(self._client)

    def _current_tool_choice(self) -> str | None:
        if self.depth > 0:
            return self.REQUIRED_TOOL_CHOICE
        return None

    def _emit_stream_event(self, event: ProviderStreamEvent) -> None:
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
