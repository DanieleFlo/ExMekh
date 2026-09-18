from __future__ import annotations

from dataclasses import dataclass

from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.deep_agent.stream_event import (
    DeepAgentEventHandler,
    DeepAgentExecutionEvent,
    DeepAgentStreamEvent,
)
from qs.core.agents.exit_skill.exit_skill import ExitSkill
from qs.core.agents.hds.hds import HDS
from qs.core.cache.node_cache import NodeCache
from qs.core.models.client import Client
from qs.core.models.system_message import SystemMessage
from qs.core.models.tools import Tools
from qs.interfaces.provider import Provider

from .agentic_scope import AgenticScope
from .tool_call_executor import ToolCallExecutor


@dataclass(frozen=True, slots=True)
class NodeRuntimeConfig:
    provider: Provider
    capabilities: CapabilitiesData
    hds: HDS
    exit_skill: ExitSkill
    persistent_history: bool
    initial_depth: int
    on_stream: DeepAgentEventHandler | None = None

    def is_persistent_history(self) -> bool:
        return self.persistent_history

    def is_streaming(self) -> bool:
        return self.on_stream is not None

    def emit_stream(
        self, event: DeepAgentStreamEvent | DeepAgentExecutionEvent
    ) -> None:
        if self.on_stream is not None:
            self.on_stream(event)

    def build_tools(self) -> Tools:
        return Tools(list(self.capabilities.get_contextual_tools(self.hds)))

    def build_node_cache(self, node_id: str) -> NodeCache:
        session = self.provider.get_session()
        return session.node(node_id)

    def build_executor(
        self,
        client: Client,
    ) -> ToolCallExecutor:
        agentic_scope = AgenticScope(
            hds=self.hds,
            exit_skill_name=self.exit_skill.name,
            exit_tool_names=self.capabilities.get_exit_tool_names(self.exit_skill.name),
        )
        return ToolCallExecutor(
            client=client,
            exit_skill=self.exit_skill,
            initial_depth=self.initial_depth,
            agentic_scope=agentic_scope,
        )

    def get_hds_name(self) -> str:
        return self.hds.get_name()

    def build_system_message(self) -> SystemMessage:
        return SystemMessage(content=self.hds.get_hds())
