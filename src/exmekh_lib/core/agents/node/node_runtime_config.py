from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.capabilities.data import CapabilitiesData
from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.agents.deep_agent.stream_event import (
    DeepAgentEvent,
)
from exmekh_lib.core.agents.exit_tool.exit_tool import ExitTool
from exmekh_lib.core.agents.hds.hds import HDS
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.tool_calls.execution_manager import ToolExecutionManager
from exmekh_lib.core.tool_calls.output_preparer import ToolOutputPreparer
from exmekh_lib.interfaces.event_handler import DeepAgentEventHandler
from exmekh_lib.interfaces.provider import Provider

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
    work_root: Path = field(default_factory=Path.cwd)
    agent_context_id: str | None = None
    runtime_tools: tuple[Tool, ...] = ()
    default_capabilities: DefaultCapabilities | None = None
    on_stream: DeepAgentEventHandler | None = None

    @property
    def resolved_agent_context_id(self) -> str:
        if self.agent_context_id is not None:
            return self.agent_context_id
        return self.hds.get_agent_context_id()

    def is_persistent_history(self) -> bool:
        return self.persistent_history

    def is_streaming(self) -> bool:
        return self.on_stream is not None

    def emit_stream(self, event: DeepAgentEvent) -> None:
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
        execution_manager: ToolExecutionManager | None = None,
        scope_prefix: str = "",
        output_preparer: ToolOutputPreparer | None = None,
    ) -> ToolCallExecutor:
        agentic_scope = AgenticScope(
            hds=self.hds,
            exit_tool_name=self.exit_tool.get_name(),
            runtime_tool_names=runtime_tool_names,
            agent_context_id=self.resolved_agent_context_id,
        )
        return ToolCallExecutor(
            client=client,
            exit_tool=self.exit_tool,
            initial_depth=self.initial_depth,
            agentic_scope=agentic_scope,
            exit_content_resolver=exit_content_resolver,
            work_root=self.work_root,
            execution_manager=execution_manager,
            scope_prefix=scope_prefix,
            output_preparer=output_preparer,
        )

    def get_hds_name(self) -> str:
        return self.hds.get_name()

    def build_system_message(self) -> SystemMessage:
        content = f"{self.hds.get_hds()}\n\n{HARNESS_OUTPUT_GUIDANCE}"
        if self.initial_depth > 0:
            content = f"{content}\n\n{self.exit_tool.render_prompt_section()}"
        return SystemMessage(content=content)
