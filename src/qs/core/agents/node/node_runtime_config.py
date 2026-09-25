from __future__ import annotations

from dataclasses import dataclass

from qs.config.runtime import RUNTIME
from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.capabilities.defaults import DefaultCapabilities
from qs.core.agents.deep_agent.stream_event import (
    DeepAgentExecutionEvent,
    DeepAgentStreamEvent,
)
from qs.core.agents.exit_tool.exit_tool import ExitTool
from qs.core.agents.hds.hds import HDS
from qs.core.cache.node_cache import NodeCache
from qs.core.models.client import Client
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool import Tool
from qs.core.models.tools import Tools
from qs.interfaces.event_handler import DeepAgentEventHandler
from qs.interfaces.provider import Provider

from .agentic_scope import AgenticScope
from .tool_call_executor import ExitContentResolver, ToolCallExecutor

HARNESS_OUTPUT_GUIDANCE = RUNTIME.harness_output_guidance


@dataclass(frozen=True, slots=True)
class NodeRuntimeConfig:
    provider: Provider
    capabilities: CapabilitiesData
    hds: HDS
    exit_tool: ExitTool
    persistent_history: bool
    initial_depth: int
    agent_context_id: str | None = None
    runtime_tools: tuple[Tool, ...] = ()
    default_capabilities: DefaultCapabilities | None = None
    on_stream: DeepAgentEventHandler | None = None

    @property
    def resolved_agent_context_id(self) -> str:
        if self.agent_context_id is not None:
            return self.agent_context_id
        get_context_id = getattr(self.hds, "get_agent_context_id", None)
        if callable(get_context_id):
            context_id = get_context_id()
            if isinstance(context_id, str) and context_id.strip():
                return context_id
        get_name = getattr(self.hds, "get_name", None)
        if callable(get_name):
            hds_name = get_name()
            if isinstance(hds_name, str) and hds_name.strip():
                return hds_name
        return RUNTIME.default_agent_context_id

    def is_persistent_history(self) -> bool:
        return self.persistent_history

    def is_streaming(self) -> bool:
        return self.on_stream is not None

    def emit_stream(
        self, event: DeepAgentStreamEvent | DeepAgentExecutionEvent
    ) -> None:
        if self.on_stream is not None:
            self.on_stream.handle(event)

    def build_tools(self, runtime_tools: tuple[Tool, ...] = ()) -> Tools:
        return Tools(
            [*self.capabilities.get_contextual_tools(self.hds), *runtime_tools]
        )

    def build_node_cache(self, node_id: str) -> NodeCache:
        session = self.provider.get_session()
        return session.node(node_id)

    def build_executor(
        self,
        client: Client,
        runtime_tool_names: tuple[str, ...] = (),
        *,
        exit_content_resolver: ExitContentResolver,
    ) -> ToolCallExecutor:
        agentic_scope = AgenticScope(
            hds=self.hds,
            exit_tool_name=self.exit_tool.name,
            runtime_tool_names=runtime_tool_names,
            agent_context_id=self.resolved_agent_context_id,
        )
        return ToolCallExecutor(
            client=client,
            exit_tool=self.exit_tool,
            initial_depth=self.initial_depth,
            agentic_scope=agentic_scope,
            exit_content_resolver=exit_content_resolver,
        )

    def get_hds_name(self) -> str:
        return self.hds.get_name()

    def build_system_message(self) -> SystemMessage:
        content = f"{self.hds.get_hds()}\n\n{HARNESS_OUTPUT_GUIDANCE}"
        if self.initial_depth > 0:
            content = f"{content}\n\n{self.exit_tool.render_prompt_section()}"
        return SystemMessage(content=content)
